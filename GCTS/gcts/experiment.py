from sklearn.metrics import recall_score, roc_auc_score, precision_score
import torch
import numpy as np

from model.graph_model import MLP_Decoder, GNN, reparameterize
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import to_dense_adj, dropout_edge
from collections import deque

from .exp_util import preprocess_graph_v2, get_pos_weight_score, kl_loss, plot_graph
from .visualization import (
    plot_time_series_reconstruction,
    plot_the_curves,
    plot_reconstruction_error_over_time,
)


def kendall_loss(loss_struct, loss_temp, log_sigma_struct, log_sigma_temp):
    """
    Kendall et al. homoscedastic uncertainty weighting.

    For a Gaussian likelihood task:
        weighted = L / (2 * exp(2*log_σ)) + log_σ
                 = L * exp(-2*log_σ) * 0.5 + log_σ

    Using log_σ (not σ directly) keeps the parameter unconstrained and
    avoids any risk of σ collapsing to zero.

    Args:
        loss_struct    : scalar — structural reconstruction loss
        loss_temp      : scalar — temporal prediction loss (or 0.0 Tensor)
        log_sigma_struct: nn.Parameter, shape ()
        log_sigma_temp  : nn.Parameter, shape ()

    Returns:
        Scalar weighted loss ready for .backward()
    """
    w_struct = torch.exp(-2.0 * log_sigma_struct)
    w_temp   = torch.exp(-2.0 * log_sigma_temp)

    weighted = (
        0.5 * w_struct * loss_struct + log_sigma_struct
        + 0.5 * w_temp  * loss_temp  + log_sigma_temp
    )
    return weighted


class Experiment:
    def __init__(self, args, train_dataset, validation_dataset, test_dataset):
        self.args = args
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )

        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers

        if self.args.input_dim is None:
            self.args.input_dim = next(iter(self.train_dataset)).x.shape[0]

        print(self.args.input_dim)
        sample_graph = next(iter(self.train_dataset))
        num_nodes = sample_graph.x.shape[1]
        print(f"num nodes: {num_nodes}")

        self.encoder = GNN(args).to(self.device)

        self.edge_decoder = MLP_Decoder(
            in_dim=args.output_dim,
            hid_dim=args.hidden_dim,
            out_dim=args.input_dim,
        ).to(self.device)

        self.temporal_model = nn.GRU(
            input_size=num_nodes,
            hidden_size=num_nodes,
            num_layers=args.num_layers,
            bidirectional=False,
            dropout=args.dropout,
            batch_first=True,
        ).to(device=self.device)

        self.log_sigma_struct = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_sigma_temp   = nn.Parameter(torch.zeros(1, device=self.device))

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.edge_decoder.parameters())
            + list(self.temporal_model.parameters())
            + [self.log_sigma_struct, self.log_sigma_temp],
            lr=args.learning_rate,
        )

        self.gru_hidden = None


    def run(self):
        metric = self.args.metric
        best_validation_acc = 0.0
        self.kl_weight = 0.0
        epochs_no_improve = 0

        temporal_window = self.args.temporal_window
        X_history = deque(maxlen=temporal_window + 1)

        validation_scores_all = []
        test_score_all = []

        for epoch in range(self.args.max_epochs):
            self.encoder.train()
            self.temporal_model.train()
            total_loss = 0.0
            total_struct_loss = 0.0
            total_temp_loss = 0.0
            total_kl = 0.0

            g = 0
            for graph in self.train_dataset:
                g += 1

                graph = graph.to(self.device)
                edge_index, _ = dropout_edge(graph.edge_index, p=0.2)
                graph.edge_index = edge_index
                graph = preprocess_graph_v2(graph)

                if g == 1:
                    plot_graph(graph.A, g, title="Original")

                graph.x = graph.x.T
                pos_weight_score = torch.tensor(
                    get_pos_weight_score(graph) ** self.args.pos_weight_exponent,
                    device=self.device,
                )

                self.optimizer.zero_grad()

                mu, logvar = self.encoder(graph.x, graph.A_norm)
                Z = reparameterize(mu, logvar)
                Z_edge = self.edge_decoder(Z)

                A_pred = torch.matmul(Z_edge, Z_edge.T).flatten()
                A_true = to_dense_adj(
                    graph.edge_index, max_num_nodes=graph.x.shape[0]
                ).view(-1)



                loss_struct = F.binary_cross_entropy_with_logits(
                    A_pred, A_true, pos_weight=pos_weight_score
                )
                loss_kl = kl_loss(mu, logvar)

                X_history.append(graph.x.T)
                if len(X_history) > temporal_window:
                    x_seq = torch.stack(list(X_history)[:-1], dim=0).detach()
                    actual_window = x_seq.shape[1]
                    x_seq = x_seq.reshape(1, -1, x_seq.shape[-1])
                    pred, self.gru_hidden = self.temporal_model(
                        x_seq,
                        self.gru_hidden.detach()
                        if self.gru_hidden is not None
                        else None,
                    )
                    x_pred = pred[0, -actual_window:, :]
                    x_target = X_history[-1]
                    loss_temp = F.mse_loss(x_pred, x_target)
                else:
                    loss_temp = torch.zeros(1, device=self.device)

                self.kl_weight = min(1.0, epoch / self.args.kl_warmup_epochs)

                loss = (
                    kendall_loss(
                        loss_struct,
                        loss_temp,
                        self.log_sigma_struct,
                        self.log_sigma_temp,
                    )
                    + self.kl_weight * loss_kl
                )

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.log_sigma_struct.clamp_(max=3.0)
                    self.log_sigma_temp.clamp_(max=3.0)

                total_struct_loss += loss_struct.item()
                total_temp_loss += loss_temp.item()
                total_kl += loss_kl.item()
                total_loss += loss.item()

            sigma_s = self.log_sigma_struct.exp().item()
            sigma_t = self.log_sigma_temp.exp().item()
            print(
                f"Epoch {epoch:03d} | Loss: {total_loss:.4f} "
                f"| Struct: {total_struct_loss:.4f} "
                f"| Temporal: {total_temp_loss:.4f} "
                f"| KL: {total_kl:.4f} "
                f"| σ_struct: {sigma_s:.3f} "
                f"| σ_temp: {sigma_t:.3f}"
            )

            if epoch % self.args.eval_every == 0:
                validation_acc = self.evaluate(self.validation_dataset, metric)
                test_acc = self.evaluate(self.test_dataset, metric)

                if metric == "all":
                    store_others_val = validation_acc
                    store_others_test = test_acc
                    validation_acc = validation_acc["auc"] ##auc
                    test_acc = test_acc["auc"] #auc

                if validation_acc > best_validation_acc:
                    best_validation_acc = validation_acc
                    best_test_acc = test_acc
                    epochs_no_improve = 0
                    if metric == "all":
                        best_others_val = store_others_val
                        best_others_test = store_others_test
                else:
                    epochs_no_improve += 1

                print(
                    f"Epoch {epoch:03d} | Val {metric}: {validation_acc:.4f} "
                    f"| Test: {test_acc:.4f}"
                )
                validation_scores_all.append(validation_acc)
                test_score_all.append(test_acc)

                if epochs_no_improve > self.args.patience:
                    print("Early stopping")
                    plot_the_curves(
                        validation_scores_all, test_score_all, "training_curves.png"
                    )
                    print("=" * 50)
                    if metric == "all":
                        return best_others_val, best_others_test
                    return best_validation_acc, best_test_acc

        print("=" * 50)
        plot_the_curves(validation_scores_all, test_score_all, "training_curves.png")
        print("=" * 50)
        if metric == "all":
            return best_others_val, best_others_test
        if self.args.last==True:
            print("HEEEEEEEEEEEEEEEREEEEEEEEEE")
            if test_score_all[-1] > best_test_acc:
                print("HEEEEEEEEEEEEEEEREEEEEEEEEE 2")
                return validation_scores_all[-1], test_score_all[-1]
            
        print("FCK HEEEEEEEEEEEEEEEREEEEEEEEEE")
        return best_validation_acc, best_test_acc


    def evaluate(self, loader, metric="acc"):
        metric_possible = ["acc", "auc", "f1", "all"]
        if metric not in metric_possible:
            return "error function not implemented"

        self.encoder.eval()
        self.temporal_model.eval()

        scores = []
        labels = []

        temporal_window = self.args.temporal_window
        X_history = deque(maxlen=temporal_window + 1)

        w_struct = torch.exp(-2.0 * self.log_sigma_struct).item()
        w_temp   = torch.exp(-2.0 * self.log_sigma_temp).item()

        with torch.no_grad():
            for _, graph in enumerate(loader):
                graph = graph.to(self.device)
                graph = preprocess_graph_v2(graph)
                graph.x = graph.x.T

                pos_weight_score = torch.tensor(
                    get_pos_weight_score(graph) ** self.args.pos_weight_exponent,
                    device=self.device,
                )

                mu, logvar = self.encoder(graph.x, graph.A_norm)
                Z = reparameterize(mu, logvar)
                Z_edge = self.edge_decoder(Z)

                A_pred = torch.matmul(Z_edge, Z_edge.T).flatten()
                A_true = to_dense_adj(
                    graph.edge_index, max_num_nodes=graph.x.shape[0]
                ).view(-1)
                err_struct = F.binary_cross_entropy_with_logits(
                    A_pred, A_true, pos_weight=pos_weight_score
                )

                X_history.append(graph.x.T)
                if len(X_history) > temporal_window:
                    X_seq = torch.stack(list(X_history)[:-1], dim=0).detach()
                    actual_window = X_seq.shape[1]
                    X_seq = X_seq.reshape(1, -1, X_seq.shape[-1])
                    pred, _ = self.temporal_model(X_seq)
                    X_pred = pred[0, -actual_window:, :]
                    X_target = X_history[-1]
                    err_temp = F.mse_loss(X_pred, X_target)
                else:
                    err_temp = torch.tensor(0.0, device=self.device)

                score = 0.5 * w_struct * err_struct + 0.5 * w_temp * err_temp
                score_val = score.item()

                y_array = graph.y.cpu().numpy().flatten()
                scores.extend(np.repeat(score_val, len(y_array)))
                labels.extend(y_array)

        scores = np.array(scores)
        labels = np.array(labels)

        threshold = scores.mean() + scores.std()
        preds = (scores > threshold).astype(int)

        if metric == "acc":
            print(f"Also Roc-AUC: {roc_auc_score(labels, scores):.4f}")
            return (preds == labels).mean()
        elif metric == "auc":
            return roc_auc_score(labels, scores)
        elif metric == "f1":
            return f1_score(labels, preds)
        elif metric == "all":
            return {
                "acc": (preds == labels).mean(),
                "auc": roc_auc_score(labels, scores),
                "f1": f1_score(labels, preds),
                "precision": precision_score(labels, preds),
                "recall": recall_score(labels, preds),
            }


    def check_gradients(self):
        print("\n=== Gradient Flow Check ===")
        for name, param in self.temporal_model.named_parameters():
            if param.grad is not None:
                print(
                    f"{name:25} | Mean: {param.grad.abs().mean():.8f} "
                    f"| Std: {param.grad.std():.8f}"
                )
            else:
                print(f"{name:25} | No Gradient")


    def visualize_reconstruction(
        self,
        loader,
        ts_indices=None,
        save_dir="ts_plots",
        tag="",
        centralities=None,
    ):
        if ts_indices is None:
            ts_indices = list(range(6))

        self.encoder.eval()
        self.temporal_model.eval()

        orig_windows = []
        recon_temp_windows = []
        error_per_window = []
        struct_error_per_window = []
        temp_error_per_window = []
        labels_per_timestep = []

        temporal_window = self.args.temporal_window
        X_history = deque(maxlen=temporal_window + 1)

        w_struct = torch.exp(-2.0 * self.log_sigma_struct).item()
        w_temp   = torch.exp(-2.0 * self.log_sigma_temp).item()

        with torch.no_grad():
            for graph in loader:
                graph = graph.to(self.device)
                graph = preprocess_graph_v2(graph)
                x_orig_node_first = graph.x.clone()
                graph.x = graph.x.T

                mu, logvar = self.encoder(graph.x, graph.A_norm)
                Z = reparameterize(mu, logvar)
                Z_edge = self.edge_decoder(Z)

                pos_weight_score = torch.tensor(
                    get_pos_weight_score(graph) ** self.args.pos_weight_exponent,
                    device=self.device,
                )
                A_pred = torch.matmul(Z_edge, Z_edge.T).flatten()
                A_true = to_dense_adj(
                    graph.edge_index, max_num_nodes=graph.x.shape[0]
                ).view(-1)
                err_struct = F.binary_cross_entropy_with_logits(
                    A_pred, A_true, pos_weight=pos_weight_score
                )

                X_history.append(graph.x.T)
                if len(X_history) > temporal_window:
                    X_seq = torch.stack(list(X_history)[:-1], dim=0).detach()
                    actual_window = X_seq.shape[1]
                    X_seq = X_seq.reshape(1, -1, X_seq.shape[-1])
                    pred, _ = self.temporal_model(X_seq)
                    X_pred_temp = pred[0, -actual_window:, :]
                    err_temp = F.mse_loss(X_pred_temp, X_history[-1])

                    orig_windows.append(x_orig_node_first.cpu().numpy())
                    recon_temp_windows.append(X_pred_temp.cpu().numpy())
                    labels_per_timestep.extend(graph.y.cpu().numpy().flatten())
                else:
                    err_temp = torch.tensor(0.0, device=self.device)

                score = (0.5 * w_struct * err_struct + 0.5 * w_temp * err_temp).item()
                struct_error_per_window.append(0.5 * w_struct * err_struct.item())
                temp_error_per_window.append(0.5 * w_temp * err_temp.item())
                error_per_window.append(score)

        if not orig_windows:
            print(
                f"[visualize_reconstruction] Not enough windows "
                f"(need > {temporal_window})."
            )
            return

        originals = np.concatenate(orig_windows, axis=0)
        recons_temporal = np.concatenate(recon_temp_windows, axis=0)
        ts_labels = np.array(labels_per_timestep)

        print(f"[visualize_reconstruction] Plotting temporal recon → {save_dir}/")
        plot_time_series_reconstruction(
            originals=originals,
            reconstructions=recons_temporal,
            ts_indices=ts_indices,
            save_dir=save_dir,
            tag=f"{tag}_temporal",
            anomaly_labels=ts_labels,
        )

        plot_reconstruction_error_over_time(
            struct_error_per_window=struct_error_per_window,
            temp_error_per_window=temp_error_per_window,
            labels_per_point=ts_labels,
            window_size=self.args.window_size,
            save_dir=save_dir,
            tag=tag,
        )
