from .input_ksdd import KSDDDataset
from .input_dagm import DagmDataset
from .input_steel import SteelDataset
from .input_ksdd2 import KSDD2Dataset
from .input_crack_segmentation import CrackSegmentationDataset
from .input_sccdnet_dataset import SccdnetDataset
from config import Config
from torch.utils.data import DataLoader
from typing import Optional

from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from data.dataset import HardExamplesBatchSampler

def get_dataset(kind: str, cfg: Config) -> Optional[DataLoader]:
    if kind == "VAL" and not cfg.VALIDATE:
        return None
    if kind == "VAL" and cfg.VALIDATE_ON_TEST:
        kind = "TEST"
    if cfg.DATASET == "KSDD":
        ds = KSDDDataset(kind, cfg)
    elif cfg.DATASET == "DAGM":
        ds = DagmDataset(kind, cfg)
    elif cfg.DATASET == "STEEL":
        ds = SteelDataset(kind, cfg)
    elif cfg.DATASET == "KSDD2":
        ds = KSDD2Dataset(kind, cfg)
    elif cfg.DATASET == "crack_segmentation" or cfg.DATASET == 'CFD' or cfg.DATASET == 'CRACK500' or cfg.DATASET == 'DeepCrack':
        ds = CrackSegmentationDataset(kind, cfg)
    elif cfg.DATASET == "sccdnet":
        ds = SccdnetDataset(kind, cfg)
    else:
        raise Exception(f"Unknown dataset {cfg.DATASET}")

    shuffle = kind == "TRAIN"
    batch_size = cfg.BATCH_SIZE if kind == "TRAIN" else 1
    num_workers = 0
    drop_last = kind == "TRAIN"
    pin_memory = False

    if kind == "TRAIN" and cfg.HARD_NEG_MINING is not None:

        hard_sample_size, hard_samples_selected_min_percent, difficulty_score_type = cfg.HARD_NEG_MINING

        if shuffle:
            default_sampler = RandomSampler(ds)
        else:
            default_sampler = SequentialSampler(ds)

        batch_sampler = HardExamplesBatchSampler(ds,
                                             default_sampler,
                                             batch_size=batch_size,
                                             hard_sample_size=int(hard_sample_size),
                                             drop_last=True,
                                             hard_samples_selected_min_percent=hard_samples_selected_min_percent)

        return DataLoader(dataset=ds, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)
        
    else:
        return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)