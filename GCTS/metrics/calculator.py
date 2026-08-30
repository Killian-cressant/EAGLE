import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc
from prts import ts_precision, ts_recall
from timeeval.metrics.vus_metrics import RangePrVUS, RangeRocVUS
import pandas as pd ###################
from gragod.metrics.models import MetricsResult, SystemMetricsResult
from gragod.metrics.visualization import print_all_metrics
from gragod.types import Datasets

N_TH_SAMPLES_DEFAULT = 100
MAX_BUFFER_SIZE_DEFAULT = {Datasets.TELCO: 2, Datasets.SWAT: 3, Datasets.CISCO: 4}


class MetricsCalculator:
    """Calculator for precision, recall, and F1 metrics."""

    def __init__(
        self,
        dataset: Datasets,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        scores: torch.Tensor,
    ):
        """
        Initialize calculator with labels and predictions.

        Args:
            labels: Ground truth labels tensor (n_samples, n_nodes)
            predictions: Predicted labels tensor (n_samples, n_nodes)
        """
        self.dataset = dataset
        self.scores = scores
        self.labels = labels
        self.predictions = predictions
        self.system_scores = torch.sum(scores, dim=1)
        self.system_labels = (torch.sum(labels, dim=1) > 0).int()
        self.system_predictions = (torch.sum(predictions, dim=1) > 0).int()

        self.calculate_only_system_metrics = labels.ndim == 0 or labels.shape[1] in [
            0,
            1,
        ]
    
    def FPR_TPR(self, pred):

        true_positives=0
        false_positives=0
        true_negatives=0
        false_negatives=0
        for k in range(len(pred)):
            if pred[k]== self.labels[k]==1:
                true_positives+=1
            if pred[k]== self.labels[k]==0:
                true_negatives+=1
            if pred[k]==1 and self.labels[k]!=pred[k]:
                false_positives+=1
            if pred[k]==0 and self.labels[k]!=pred[k]:
                false_negatives+=1


        true_negatives=torch.tensor(true_negatives)
        true_positives=torch.tensor(true_positives)
        false_negatives=torch.tensor(false_negatives)
        false_positives=torch.tensor(false_positives)



        per_fpr= torch.where(
            false_positives+true_negatives>0,
            false_positives/(false_positives+true_negatives),
            torch.zeros_like(false_positives, dtype=torch.float),
        )

        per_tpr=torch.where(
            true_positives+false_negatives>0,
            true_positives/(true_positives+false_negatives),
            torch.zeros_like(true_positives, dtype=torch.float),
        )



        return(per_fpr,per_tpr)
        



    def calculate_roc_score(self, Nstep) -> MetricsResult | SystemMetricsResult :
        

        roc_score=0
        mem_tprk=0
        mem_fprk=0
        fpr_list=[]
        tpr_list=[]

        scored_renorm=(self.system_scores-torch.min(self.system_scores))/(torch.max(self.system_scores)-torch.min(self.system_scores))

        ths=np.linspace(0,1,Nstep)

        for k in range(1,Nstep+1):
            scored=(scored_renorm > ths[k-1]).int() 

            fprk,tprk=self.FPR_TPR(scored)

            deltfpr=fprk-mem_fprk
            delttpr=tprk-mem_tprk
            roc_score+=(deltfpr*delttpr)/2


            mem_fprk=fprk
            mem_tprk=tprk
            fpr_list.append(fprk)
            tpr_list.append(tprk)

        roc_score=roc_score
        rc2=auc(fpr_list,tpr_list)
        fc=torch.tensor(rc2)
        per_rc2 = torch.where(
            fc > 0,
            fc,
            torch.zeros_like(fc, dtype=torch.float),
        )
        print(rc2)
        fpr_np = np.array(fpr_list)
        tpr_np = np.array(tpr_list)

      
        df = pd.DataFrame({'FPR': fpr_np, 'TPR': tpr_np, 'ths':ths})

        df.to_csv("fpr_tpr.csv", index=False)
        rc3=torch.tensor([rc2, roc_score])

        return MetricsResult(
            metric_global=rc2,
            metric_mean=float(rc2),
            metric_per_class=rc3,
            metric_system=float(rc2))



    def calculate_vus_pr(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate VUS-PR metrics.
        Based on https://www.paparrizos.org/papers/PaparrizosVLDB22b.pdf.

        Args:
            max_buffer_size: Maximum size of the buffer region around an anomaly.
                We iterate over all buffer sizes from 0 to ``max_buffer_size`` to
                create the surface.
            max_th_samples: Calculating precision and recall for many thresholds is
                quite slow. We, therefore, uniformly sample thresholds from the
                available score space. This parameter controls the maximum number of
                thresholds; too low numbers degrade the metrics' quality.

        Returns:
            MetricsResult | SystemMetricsResult: VUS-PR metrics.
        """
        if max_buffer_size is None:
            max_buffer_size = MAX_BUFFER_SIZE_DEFAULT[self.dataset]

        system_labels_float64 = np.array(self.system_labels, dtype=np.float64)
        system_scores_float64 = np.array(self.system_scores, dtype=np.float64)

        vus_pr = RangePrVUS(
            max_buffer_size=max_buffer_size,
            compatibility_mode=True,
            max_samples=max_th_samples,
        )

        system_vus_pr = (
            vus_pr(
                y_true=system_labels_float64,
                y_score=system_scores_float64,
            )
            if torch.sum(self.system_labels) > 0
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_vus_pr))

        scores_float64 = np.array(self.scores, dtype=np.float64)
        labels_float64 = np.array(self.labels, dtype=np.float64)

        per_class_vus_pr = [
            (
                vus_pr(
                    y_true=labels_float64[:, i],
                    y_score=scores_float64[:, i],
                )
                if not (
                    np.allclose(np.unique(labels_float64[:, i]), np.array([0]))
                    or np.allclose(np.unique(scores_float64[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(labels_float64.shape[1])
        ]
        mean_vus_pr = torch.mean(torch.tensor(per_class_vus_pr))

        global_vus_pr = None

        return MetricsResult(
            metric_global=global_vus_pr,
            metric_mean=float(mean_vus_pr),
            metric_per_class=torch.tensor(per_class_vus_pr),
            metric_system=float(system_vus_pr),
        )

    def get_all_metrics(self, alpha: float = 1.0) -> dict[str, torch.Tensor]:
        """
        Calculate all metrics and return as dictionary.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of metrics.
        """

        auc_score=self.calculate_roc_score(Nstep=40)

        vus_roc = self.calculate_vus_roc()
        vus_pr = self.calculate_vus_pr()

        return {
            **auc_score.model_dump("auc"),
            **vus_roc.model_dump("vus_roc"),
            **vus_pr.model_dump("vus_pr"),
        }


def get_metrics(
    dataset: Datasets,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    range_metrics_alpha: float = 1.0,
) -> dict:
    """
    Calculate and visualize all metrics for given predictions and labels.

    Args:
        predictions: Predicted labels tensor
        labels: Ground truth labels tensor

    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = MetricsCalculator(
        dataset=dataset, labels=labels, predictions=predictions, scores=scores
    )
    metrics = calculator.get_all_metrics(alpha=range_metrics_alpha)

    return metrics


def get_metrics_and_save(
    dataset: Datasets,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    save_dir: Path,
    dataset_split: str,
):
    metrics = get_metrics(dataset, predictions, labels, scores)
    print_all_metrics(metrics, f"------- {dataset_split.capitalize()} -------")
    json.dump(
        metrics,
        open(
            os.path.join(
                save_dir,
                f"{dataset_split}_metrics.json",
            ),
            "w",
        ),
    )
    return metrics
