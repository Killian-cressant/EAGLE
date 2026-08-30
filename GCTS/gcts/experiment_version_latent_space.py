from sklearn.metrics import roc_auc_score
import torch
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import inf
from tqdm.auto import tqdm

from model.graph_model import GIN, MLP_Decoder, GNN, reparameterize
from torch import nn
import matplotlib.pyplot as plt


import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import to_dense_adj, to_dense_batch, dropout_edge
from collections import deque

from .exp_util import preprocess_graph_v2, get_pos_weight_score, kl_loss
from .visualization import plot_time_series_reconstruction, plot_the_curves, plot_reconstruction_error_over_time





class Experiment_Latent:
    def __init__(self, args, train_dataset, validation_dataset, test_dataset):
        self.args = args
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.device = torch.device(
            f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        )


        if self.args.hidden_layers is None:
            self.args.hidden_layers = [self.args.hidden_dim] * self.args.num_layers

        
        if self.args.input_dim is None:
            self.args.input_dim=next(iter(self.train_dataset)).x.shape[0]

        print(self.args.input_dim)
        sample_graph = next(iter(self.train_dataset))
        num_nodes = sample_graph.x.shape[1]
        print(f"num nodes: {num_nodes}")

        
        self.encoder= GNN(args)


        self.edge_decoder = MLP_Decoder(
            in_dim=args.output_dim,
            hid_dim=args.hidden_dim,
            out_dim=args.input_dim
        ).to(self.device)


        self.temporal_model = nn.GRU(
            input_size=self.args.output_dim, 
            hidden_size=self.args.output_dim,
            batch_first=True
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.edge_decoder.parameters()) +
            list(self.temporal_model.parameters()),
            lr=args.learning_rate
        )


    def run(self):


        metric = self.args.metric
        best_validation_acc = 0.0
        self.kl_weight = 0.0

        temporal_window = self.args.temporal_window
        Z_history= deque(maxlen=temporal_window+1)

        validation_scores_all=[]
        test_score_all=[]

        for epoch in range(self.args.max_epochs):

            self.encoder.train()
            total_loss = 0

            for graph in self.train_dataset:

                graph = graph.to(self.device)
                edge_index, _ = dropout_edge(graph.edge_index, p=self.args.dropout)
                graph.edge_index=edge_index

                graph=preprocess_graph_v2(graph)
      
                graph.x=graph.x.T

                pos_weight_score=torch.tensor(get_pos_weight_score(graph), device=self.device)

                self.optimizer.zero_grad()

                mu, logvar = self.encoder(graph.x, graph.A_norm)

                Z = reparameterize(mu, logvar)

                Z=Z+0.1 * torch.randn_like(Z)
                Z_edge = self.edge_decoder(Z)


                A_pred = torch.matmul(Z_edge, Z_edge.T).flatten()
                A_true = to_dense_adj(graph.edge_index, max_num_nodes=graph.x.shape[0]).view(-1)

                loss_struct = F.binary_cross_entropy_with_logits(A_pred, A_true, pos_weight=pos_weight_score)
                loss_kl = kl_loss(mu, logvar)
                Z_history.append(Z.detach())
                Z_history.append(Z)

                if len(Z_history) > temporal_window:

                    Z_seq = torch.stack(list(Z_history)[:-1], dim=0).detach()
                    actual_window = Z_seq.shape[1]
                    Z_seq = Z_seq.reshape(1, -1, Z_seq.shape[-1])
                    pred, _ = self.temporal_model(Z_seq)
                    Z_pred = pred[0, -actual_window:, :] 

                    Z_target = Z_history[-1]

                    loss_temp = F.mse_loss(Z_pred, Z_target)


                else:
                    loss_temp = 0.0

                self.kl_weight = min(1.0, epoch / self.args.kl_warmup_epochs)
                loss = loss_struct +  loss_temp + self.kl_weight * loss_kl

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Loss recons: {loss_struct} | Loss temporal: {loss_temp} | loss KL : {loss_kl}")# loss feat: {loss_feat}")

            if epoch % self.args.eval_every == 0:
                validation_acc = self.evaluate(self.validation_dataset, metric)
                test_acc = self.evaluate(self.test_dataset, metric)

                if validation_acc > best_validation_acc:
                    best_validation_acc = validation_acc
                    best_test_acc = test_acc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1


                print(f"Epoch {epoch} | Val {metric}: {validation_acc:.4f} | Test: {test_acc:.4f}")

                validation_scores_all.append(validation_acc)
                test_score_all.append(test_acc)


                if epochs_no_improve > self.args.patience:
                    print("Early stopping")
                    return best_validation_acc, best_test_acc


        plot_the_curves(validation_scores_all, test_score_all, "training_curves.png")

        return best_validation_acc, best_test_acc


    def evaluate(self,loader, metric='acc'):

        metric_possible = ["acc", "auc", "f1"]
        if metric not in metric_possible:
            return "error function not implemented"

        self.encoder.eval()
        scores = []
        labels = []

        temporal_window = self.args.temporal_window
        Z_history = deque(maxlen=temporal_window + 1)

        with torch.no_grad():
            for _,graph in enumerate(loader):

                graph = graph.to(self.device)
                graph=preprocess_graph_v2(graph)
                graph.x=graph.x.T

                pos_weight_score=torch.tensor(get_pos_weight_score(graph), device=self.device)

                mu, logvar = self.encoder(graph.x, graph.A_norm) 

                Z = reparameterize(mu, logvar)
                Z_edge = self.edge_decoder(Z)


                A_pred = torch.matmul(Z_edge, Z_edge.T).flatten()
                A_true = to_dense_adj(graph.edge_index, max_num_nodes=graph.x.shape[0]).view(-1)


                err_struct = F.binary_cross_entropy_with_logits(
                    A_pred, A_true, pos_weight=pos_weight_score
                )

                Z_history.append(Z)

                if len(Z_history) > temporal_window:
                    Z_seq = torch.stack(list(Z_history)[:-1], dim=0).detach()
                    actual_window = Z_seq.shape[1]
                    Z_seq = Z_seq.reshape(1, -1, Z_seq.shape[-1]) 

                    pred, _ = self.temporal_model(Z_seq)
                    Z_pred = pred[0, -actual_window:, :] 

                    Z_target = Z_history[-1]

                    err_temp = F.mse_loss(Z_pred, Z_target)

                else:
                    err_temp = 0

                score = err_struct  + err_temp 

                scores.append(score.item())
                labels.append(graph.y.item())

        scores = np.array(scores)
        labels = np.array(labels)


        threshold = scores.mean() + scores.std()
        preds = (scores > threshold).astype(int)


        if metric == 'acc':
            print(f"Also Roc-AUC: {roc_auc_score(labels, scores):.4f}")
            return (preds == labels).mean()

        elif metric == 'auc':
            return roc_auc_score(labels, scores)

        elif metric == 'f1':
            return f1_score(labels, preds)


    def visualize_reconstruction(
        self,
        loader,
        ts_indices=None,
        save_dir="ts_plots",
        tag="",
    ):
        if ts_indices is None:
            ts_indices = list(range(6))

        self.encoder.eval()
        self.temporal_model.eval()

        orig_windows        = []
        recon_temp_windows  = []
        error_per_window    = []
        labels_per_window   = []

        temporal_window = self.args.temporal_window
        Z_history = deque(maxlen=temporal_window + 1)

        with torch.no_grad():
            for graph in loader:
                graph = graph.to(self.device)
                graph = preprocess_graph_v2(graph)


                graph.x = graph.x.T  

                mu, logvar = self.encoder(graph.x, graph.A_norm)
                Z          = reparameterize(mu, logvar)
                Z_edge     = self.edge_decoder(Z)

                pos_weight_score = torch.tensor(
                    get_pos_weight_score(graph), device=self.device
                )
                A_pred = torch.matmul(Z_edge, Z_edge.T).flatten()
                A_true = to_dense_adj(
                    graph.edge_index, max_num_nodes=graph.x.shape[0]
                ).view(-1)
                err_struct = F.binary_cross_entropy_with_logits(
                    A_pred, A_true, pos_weight=pos_weight_score
                )

                Z_history.append(Z) 

                if len(Z_history) > temporal_window:
                    Z_seq = torch.stack(list(Z_history)[:-1], dim=0).detach()
                    actual_window = Z_seq.shape[1]
                    Z_seq = Z_seq.reshape(1, -1, Z_seq.shape[-1])
                    pred, _ = self.temporal_model(Z_seq)
                    Z_pred_temp = pred[0, -actual_window:, :]       
                    err_temp    = F.mse_loss(Z_pred_temp, Z_history[-1])

                    orig_np       = Z_history[-1].cpu().numpy() 
                    recon_temp_np = Z_pred_temp.cpu().numpy()


                    orig_windows.append(orig_np)
                    recon_temp_windows.append(recon_temp_np)
                    labels_per_window.append(int(graph.y.item()))

                else:
                    err_temp = torch.tensor(0.0, device=self.device)

                score = (err_struct  + err_temp).item()
                error_per_window.append(score)

        if not orig_windows:
            print("[visualize_reconstruction] Not enough windows to plot "
                f"(need > {temporal_window} windows in loader).")
            return

        originals        = np.concatenate(orig_windows,       axis=0)
        recons_temporal  = np.concatenate(recon_temp_windows, axis=0)

        window_size = orig_windows[0].shape[0]
        ts_labels   = np.repeat(labels_per_window, window_size)

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
            error_per_window=error_per_window,
            labels_per_window=[int(graph.y.item()) for graph in loader],
            save_dir=save_dir,
            tag=tag,
        )
