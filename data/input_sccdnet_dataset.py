import os
import numpy as np
from data.dataset import Dataset
from config import Config
from datetime import datetime

from tqdm import tqdm
class SccdnetDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(SccdnetDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_samples(self, path_to_samples):

        samples = [i for i in sorted(os.listdir(os.path.join(path_to_samples, 'images')))]

        if self.cfg.TRAIN_SPLIT is not None:
            if self.kind == 'TRAIN':
                samples = []
                for f in range(1, 6):
                    if f != self.cfg.TRAIN_SPLIT:
                        samples += [s.strip() for s in open(os.path.join(self.cfg.DATASET_PATH, "splits", f"split_{f}.txt"), "r").readlines()]
            
            elif self.kind == 'VAL':
                samples = [s.strip() for s in open(os.path.join(self.cfg.DATASET_PATH, "splits", f"split_{self.cfg.TRAIN_SPLIT}.txt"), "r").readlines()]

        for sample in tqdm(samples):
            id, file_type = sample.rsplit(".", 1)
            
            image_path = os.path.join(path_to_samples, 'images', sample)
            seg_mask_path = os.path.join(path_to_samples, 'masks', sample)
            
            seg_mask, _ = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)
            seg_mask = np.array((seg_mask > 0.5), dtype=np.float32)
            positive = seg_mask.max() > 0

            self.pos_pixels += (seg_mask == 1).sum().item()
            self.neg_pixels += (seg_mask == 0).sum().item()

            if not self.cfg.ON_DEMAND_READ:
                image = self.read_img_resize(image_path, self.grayscale, self.image_size)
                image = self.to_tensor(image)
                seg_mask = self.to_tensor(seg_mask)
            else:
                image = None
                seg_mask = None

            if positive:
                self.pos_samples.append((image, seg_mask, True, image_path, seg_mask_path, id, True))
            else:
                self.neg_samples.append((image, seg_mask, True, image_path, seg_mask_path, id, False))
                
    def read_contents(self):

        self.pos_samples = list()
        self.neg_samples = list()

        self.neg_pixels = 0
        self.pos_pixels = 0

        if self.kind == 'TRAIN' or self.kind == 'VAL':
            self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'train'))
        elif self.kind == 'TEST':
            self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'test'))

        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)

        self.len = self.num_pos + self.num_neg
        
        time = datetime.now().strftime("%d-%m-%y %H:%M")

        self.pos_weight_seg = self.neg_pixels / self.pos_pixels if self.pos_pixels else 0
        self.pos_weight_dec = self.num_neg / self.num_pos if self.num_pos else 0

        if self.kind == 'TRAIN' and self.cfg.BCE_LOSS_W:
            print(f"{time} {self.kind}: Number of positives: {self.num_pos}, Number of negatives: {self.num_neg}, Sum: {self.len}, Seg pos_weight: {round(self.pos_weight_seg, 3)}, Dec pos_weight: {round(self.pos_weight_dec, 3)}")
        else:
            print(f"{time} {self.kind}: Number of positives: {self.num_pos}, Number of negatives: {self.num_neg}, Sum: {self.len}")

        self.init_extra()