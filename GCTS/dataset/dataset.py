import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .graph import build_local_graph, build_cosi_graph, build_local_graph_v2
from gcts.types import *
from gcts.config import SWATPaths
from gcts.config import CISCOPaths
from gcts.config import WADIPaths
from gcts.config import TELCOPaths


class SlidingWindowDataset(Dataset):
    """
    Dataset creating sliding windows over time-series data
    and returning PyG Data objects.
    """

    def __init__(
        self,
        dataset_name: Datasets,
        data: torch.Tensor,
        window_size: int,
        edge_index: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        horizon: int = 1,
        drop: bool = False,
        graph_type: str = "corr",
        top_k: int = 200,
        cache_path: str = None
    ):
        self.dataset_name= dataset_name
        self.data = data
        self.labels = labels
        self.edge_index = edge_index
        self.window_size = window_size
        self.horizon = horizon
        self.drop_anomalous_windows = drop
        self.graph_type = graph_type
        self.top_k= top_k
        self.cache_path= cache_path
        
        self.valid_indices = self._get_valid_indices()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):

        idx = self.valid_indices[index]

        x = self.data[idx : idx + self.window_size].float()

        y = self.data[
            idx + self.window_size :
            idx + self.window_size + self.horizon
        ].float()

        if self.labels is not None:
            out_labels = self.labels[
                idx :
                idx + self.window_size
            ].float()

        else:
            out_labels = torch.tensor(0.0, device=self.data.device)


        if self.edge_index is None:
            if self.graph_type == "cosi":
                if self.dataset_name == Datasets.SWAT:
                    name_d= "swat"
                    model_wv=SWATPaths().model_path
                    description_file=SWATPaths().description_file
                    edge_index = build_cosi_graph(x,w2v_path=model_wv, device=self.data.device, description_file=description_file, dataset_name=name_d, top_k=self.top_k, cache_path=self.cache_path)

                if self.dataset_name == Datasets.CISCO:
                    named_d="cisco"
                    model_wv=CISCOPaths().model_path
                    description_file=CISCOPaths().description_file
                    edge_index = build_cosi_graph(x, w2v_path=model_wv, device=self.data.device, description_file=description_file, dataset_name=named_d, top_k=self.top_k, cache_path=self.cache_path)

                if self.dataset_name == Datasets.WADI:
                    named_d="wadi"
                    model_wv=WADIPaths().model_path
                    description_file=WADIPaths.description_file
                    edge_index= build_cosi_graph(x, w2v_path=model_wv, device=self.data.device, description_file=description_file, dataset_name=named_d, top_k=self.top_k, cache_path=self.cache_path)
                if self.dataset_name != Datasets.SWAT and self.dataset_name != Datasets.CISCO and self.dataset_name != Datasets.WADI:
                    raise ValueError(f" COSI graph not implemented YET for {self.dataset_name}")
            elif self.graph_type == "corr":
                edge_index = build_local_graph_v2(x, device=self.data.device, top_k=self.top_k)
            else:
                raise ValueError(f"Unknown graph type: {self.graph_type}")
            edge_index= edge_index.long()
        else:
            edge_index = self.edge_index.long()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=out_labels
        )

        return data

    def _get_valid_indices(self):
        max_start = len(self.data) - self.window_size - self.horizon
        indices = torch.arange(0, max_start, step=self.horizon)

        if self.labels is None or not self.drop_anomalous_windows:
            print(f"Using {len(indices)} windows with stride={self.horizon}")
            return indices

        valid_mask = []
        for i in indices:
            has_anomaly = torch.any(
                self.labels[i :
                            i + self.window_size]
            )
            valid_mask.append(not has_anomaly)

        valid_indices = indices[torch.tensor(valid_mask)]

        print(f"Using {len(valid_indices)} filtered windows (stride={self.horizon})")

        return valid_indices
    


def get_data_loader(
    dataset_name: Datasets,
    X: torch.Tensor,
    edge_index: torch.Tensor | None,
    y: torch.Tensor,
    batch_size: int,
    window_size: int,
    horizon: int,
    shuffle: bool = True,
    graph_type: str = "corr",
    top_k: int = 200,
    cache_path: str = None
):

    dataset = SlidingWindowDataset(
        dataset_name=dataset_name,
        data=X,
        edge_index=edge_index,
        window_size=window_size,
        horizon=horizon,
        labels=y,
        graph_type=graph_type,
        top_k=top_k,
        cache_path=cache_path
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
    )
    return loader