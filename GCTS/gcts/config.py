from dataclasses import dataclass
from typing import Dict, Type

from gcts.types import Datasets


@dataclass
class Paths:
    base_path: str


@dataclass
class SWATPaths(Paths):
    base_path: str = "/"
    name_train: str = ""
    name_val: str = ""
    name_test: str = ""
    edge_index_path: str = ""
    description_file: str = "/"
    model_path: str = "/" 


@dataclass
class WADIPaths(Paths):
    base_path: str = "/"
    name_train: str = ""
    name_val: str = ""
    name_test: str = ""
    edge_index_path: str = ""
    description_file: str = "/"
    model_path: str = "/"




@dataclass
class TELCOPaths(Paths):
    base_path: str = "/"

@dataclass
class CISCOPaths(Paths):
    base_path: str = "/"
    name_train: str = ""
    name_val: str = ""
    name_test: str = ""
    edge_index_path: str = ""
    description_file: str = ""
    model_path: str = ""
    
@dataclass
class SMDPaths(Paths):
    base_path: str = "/"
    name_train: str = ""
    name_test: str = ""
    name_test_label: str = ""


@dataclass
class DatasetConfig:
    normalize: bool
    paths: Type[Paths]


@dataclass
class SWATConfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = SWATPaths

@dataclass
class WADIConfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = WADIPaths

@dataclass
class TELCOConfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = TELCOPaths

@dataclass
class CISCOnfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = CISCOPaths

@dataclass
class SMDconfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = SMDPaths



def get_dataset_config(dataset: Datasets) -> DatasetConfig:
    DATASET_CONFIGS: Dict[Datasets, DatasetConfig] = {
        Datasets.SWAT: SWATConfig(),
        Datasets.TELCO: TELCOConfig(),
        Datasets.CISCO: CISCOnfig(),
        Datasets.WADI: WADIConfig(),
        Datasets.SMD: SMDconfig(),
    }
    return DATASET_CONFIGS[dataset]
