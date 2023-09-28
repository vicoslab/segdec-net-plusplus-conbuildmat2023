import cv2
import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt
from scipy.signal import convolve2d
from config import Config
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from torch.utils.data import Sampler
import pickle
#from datasets import LockableSeedRandomAccess

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, cfg: Config, kind: str):
        super(Dataset, self).__init__()
        self.path: str = path
        self.cfg: Config = cfg
        self.kind: str = kind
        self.image_size: (int, int) = (self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT)
        self.grayscale: bool = self.cfg.INPUT_CHANNELS == 1

        self.num_negatives_per_one_positive: int = 1
        self.frequency_sampling: bool = self.cfg.FREQUENCY_SAMPLING and self.kind == 'TRAIN'

        if self.cfg.REPRODUCIBLE_RUN is not None:
            torch.random.manual_seed(self.cfg.REPRODUCIBLE_RUN)

    def init_extra(self):
        self.counter = 0
        self.neg_imgs_permutation = np.random.permutation(self.num_neg)

        self.neg_retrieval_freq = np.zeros(shape=self.num_neg)
    
    def count_pixels(self, pixel_type):
        return sum([(s[1] == pixel_type).sum().item() for s in self.pos_samples]) + sum([(s[1] == pixel_type).sum().item() for s in self.neg_samples])

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, bool, str, bool):

        if self.counter >= self.len:
            self.counter = 0
            if self.frequency_sampling:
                sample_probability = 1 - (self.neg_retrieval_freq / np.max(self.neg_retrieval_freq))
                sample_probability = sample_probability - np.median(sample_probability) + 1
                sample_probability = sample_probability ** (np.log(len(sample_probability)) * 4)
                sample_probability = sample_probability / np.sum(sample_probability)

                # use replace=False for to get only unique values
                try:
                    self.neg_imgs_permutation = np.random.choice(range(self.num_neg), size=self.num_negatives_per_one_positive * self.num_pos, p=sample_probability, replace=False)
                except:
                    self.neg_imgs_permutation = np.random.choice(range(self.num_neg), size=self.num_negatives_per_one_positive * self.num_pos, p=sample_probability, replace=True)
            else:
                self.neg_imgs_permutation = np.random.permutation(self.num_neg)


        if self.kind == 'TRAIN':
            if index >= self.num_pos:
                ix = index % self.num_pos
                ix = self.neg_imgs_permutation[ix]
                item = self.neg_samples[ix]
                self.neg_retrieval_freq[ix] = self.neg_retrieval_freq[ix] + 1

            else:
                ix = index
                item = self.pos_samples[ix]
        else:
            if index < self.num_neg:
                ix = index
                item = self.neg_samples[ix]
            else:
                ix = index - self.num_neg
                item = self.pos_samples[ix]

        image, seg_mask, is_segmented, image_path, seg_mask_path, sample_name, is_pos = item

        if self.cfg.ON_DEMAND_READ:
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            image = self.to_tensor(image)
            seg_mask, _ = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)
            seg_mask = np.array((seg_mask > 0.5), dtype=np.float32)
            seg_mask = self.to_tensor(seg_mask)

        self.counter = self.counter + 1

        # Augmentacija
        if self.cfg.AUGMENTATION and self.kind == 'TRAIN' and is_pos:
            p1 = 0.5
            p2 = 0.5
            if torch.rand(1) < p1:
                # Horizontal flip
                if torch.rand(1) < p2:
                    image = F.hflip(image)
                    seg_mask = F.hflip(seg_mask)
                # Vertical flip
                if torch.rand(1) < p2:
                    image = F.vflip(image)
                    seg_mask = F.vflip(seg_mask)
                # 180 Rotation
                if torch.rand(1) < p2:
                    image = F.rotate(image, 180)
                    seg_mask = F.rotate(seg_mask, 180)
                # Color Jittering
                if torch.rand(1) < p2:
                    color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                    image = color_jitter(image)

        return image, seg_mask, is_segmented, sample_name, is_pos, index

    def __len__(self):
        return self.len

    def read_contents(self):
        pass

    def read_img_resize(self, path, grayscale, resize_dim) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if resize_dim is not None:
            img = cv2.resize(img, dsize=resize_dim)
        return np.array(img, dtype=np.float32) / 255.0

    def read_label_resize(self, path, resize_dim, dilate=None) -> (np.ndarray, bool):
        lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if dilate is not None and dilate > 1:
            lbl = cv2.dilate(lbl, np.ones((dilate, dilate)))
        if resize_dim is not None:
            lbl = cv2.resize(lbl, dsize=resize_dim)
        return np.array((lbl / 255.0), dtype=np.float32), np.max(lbl) > 0

    def to_tensor(self, x) -> torch.Tensor:
        if x.dtype != np.float32:
            x = (x / 255.0).astype(np.float32)

        if len(x.shape) == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        else:
            x = np.expand_dims(x, axis=0)

        x = torch.from_numpy(x)
        return x

    def distance_transform(self, mask: np.ndarray, max_val: float, p: float) -> np.ndarray:
        h, w = mask.shape[:2]
        dst_trf = np.zeros((h, w))
        
        num_labels, labels = cv2.connectedComponents((mask * 255.0).astype(np.uint8), connectivity=8)
        for idx in range(1, num_labels):
            mask_roi= np.zeros((h, w))
            k = labels == idx
            mask_roi[k] = 255
            dst_trf_roi = distance_transform_edt(mask_roi)
            if dst_trf_roi.max() > 0:
                dst_trf_roi = (dst_trf_roi / dst_trf_roi.max())
                dst_trf_roi = (dst_trf_roi ** p) * max_val
            dst_trf += dst_trf_roi

        dst_trf[mask == 0] = 1
        return np.array(dst_trf, dtype=np.float32)

    def downsize(self, image: np.ndarray, downsize_factor: int = 8) -> np.ndarray:
        img_t = torch.from_numpy(np.expand_dims(image, 0 if len(image.shape) == 3 else (0, 1)).astype(np.float32))
        img_t = torch.nn.ReflectionPad2d(padding=(downsize_factor))(img_t)
        image_np = torch.nn.AvgPool2d(kernel_size=2 * downsize_factor + 1, stride=downsize_factor)(img_t).detach().numpy()
        return image_np[0] if len(image.shape) == 3 else image_np[0, 0]

    def rle_to_mask(self, rle, image_size):
        if len(rle) % 2 != 0:
            raise Exception('Suspicious')

        w, h = image_size
        mask_label = np.zeros(w * h, dtype=np.float32)

        positions = rle[0::2]
        length = rle[1::2]
        for pos, le in zip(positions, length):
            mask_label[pos - 1:pos + le - 1] = 1
        mask = np.reshape(mask_label, (h, w), order='F').astype(np.uint8)
        return mask

class HardExamplesBatchSampler(Sampler):

    def __init__(self, dataset, default_sampler, batch_size, hard_sample_size, drop_last, hard_samples_selected_min_percent=0.0,
                 device=None, world_size=None, rank=None, is_distributed=False):
        if not isinstance(default_sampler, Sampler):
            raise ValueError("default_sampler should be an instance of "
                             "torch.utils.data.Sampler, but got default_sampler={}"
                             .format(default_sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not (isinstance(hard_sample_size, int) or hard_sample_size is None) or \
                hard_sample_size < 0 or hard_sample_size >= batch_size :
            raise ValueError("hard_sample_size should be a positive integer value smaller than batch_size, "
                             "but got hard_sample_size={}".format(hard_sample_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.is_distributed = is_distributed and world_size > 1
        self.world_size = world_size if self.is_distributed else 1
        self.rank = rank if self.is_distributed else 0
        self.device = device

        self.dataset = dataset
        self.default_sampler = default_sampler
        if self.is_distributed:
            self.hard_sampler = DistributedSubsetRandomSampler(list(range(len(default_sampler))),device=device)
        else:
            self.hard_sampler = torch.utils.data.SubsetRandomSampler(list(range(len(default_sampler))))
        self.hard_sample_size = hard_sample_size if hard_sample_size is not None else 0
        self.hard_samples_selected_min_percent = hard_samples_selected_min_percent if hard_samples_selected_min_percent is not None else 0
        self.batch_size = batch_size
        self.drop_last = drop_last


        self.sample_losses = dict()
        self.sample_storage = dict()
        self.sample_storage_tmp = dict()

    def update_sample_loss_batch(self, gt_sample, losses, index_key='index', storage_keys=[]):
        #assert index_key in gt_sample, "Index key %s is not present in gt_sample" % index_key

        indices = gt_sample[index_key]

        # convert to numpy
        indices = indices.detach().cpu().numpy() if isinstance(indices, torch.Tensor) else indices
        losses = losses.detach().cpu().numpy() if isinstance(losses, torch.Tensor) else losses

        for i,l in enumerate(losses):
            # get id of the sample (i.e. its index key)
            id = indices[i]

            # store its loss value
            self.sample_losses[id] = l
            # store any additional info required to pass along for hard examples
            # (save to temporary array which will be used for next epoch)
            self.sample_storage_tmp[id] = {k:gt_sample[k][i] for k in storage_keys}

    def retrieve_hard_sample_storage_batch(self, ids, key=None):
        # convert to numpy
        ids = ids.detach().cpu().numpy() if isinstance(ids, torch.Tensor) else ids
        # return matching sample_storage value for hard examples (i.e. for first N samples, where N=self.hard_sample_size)
        return [self.sample_storage[id][key] if n < self.hard_sample_size and id in self.sample_storage else None for n,id in enumerate(ids)]

    def _synchronize_dict(self, array):
        return distributed_sync_dict(array, self.world_size, self.rank, self.device)

    def _recompute_hard_samples_list(self):
        if self.is_distributed:
            self.sample_losses = self._synchronize_dict(self.sample_losses)
        if len(self.sample_losses) > 0:
            k = np.array(list(self.sample_losses.keys()))
            v = np.array([self.sample_losses[i] for i in k])
            v = (v - v.mean()) / v.std()
            hard_ids = list(k)
            for std_thr in [2, 1, 0.5, 0]:
                new_hard_ids = list(k[v > std_thr])
                if len(new_hard_ids) > len(v)*self.hard_samples_selected_min_percent:
                    hard_ids = new_hard_ids
                    break
            self.hard_sampler.indices = hard_ids if len(hard_ids) > 0 else list(k)
            if self.rank == 0:
                print('Number of hard samples present: %d/%d' % (len(hard_ids), len(self.sample_losses)))

        """
        if isinstance(self.dataset,LockableSeedRandomAccess):
            # lock seeds for hard samples BUT not for the whole dataset i.e. 90% of the whole dataset
            # (otherwise this will fully lock seeds for all samples and prevent new random augmentation of samples)
            self.dataset.lock_samples_seed(self.hard_sampler.indices if len(self.hard_sampler.indices) < len(self.sample_losses)*0.9 else [])
        """

        # update storage for next iteration
        self.sample_storage = self._synchronize_dict(self.sample_storage_tmp) if self.is_distributed else self.sample_storage_tmp
        self.sample_storage_tmp = dict()

    def __iter__(self):
        from itertools import islice
        self._recompute_hard_samples_list()
        max_index = len(self.default_sampler)
        if self.drop_last:
            total_batch_size = self.batch_size * self.world_size
            max_index = (max_index // total_batch_size) * total_batch_size

        batch = []
        hard_iter = iter(self.hard_sampler)
        self.usage_freq = {i: 0 for i in range(len(self.default_sampler))}
        for idx in islice(self.default_sampler,self.rank,max_index,self.world_size):
            batch.append(idx)
            # stop when spaces for normal samples filled
            if len(batch) == self.batch_size-self.hard_sample_size:
                # fill remaining places with hard examples
                # (does not need to be sync for distributed since sampling is random with replacement)
                while len(batch) < self.batch_size:
                    try:
                        batch.insert(0,next(hard_iter))
                    except StopIteration: # reset iter if no more samples
                        hard_iter = iter(self.hard_sampler)

                for b in batch: self.usage_freq[b] += 1
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            for b in batch: self.usage_freq[b] += 1
            yield batch

    def get_avg_sample_loss(self):
        return np.array(list(self.sample_losses.values())).mean()

    def get_sample_losses(self):
        return self.sample_losses.copy()

    def get_sample_frequency_use(self):
        return self.usage_freq.copy()

    def __len__(self):
        size_default = len(self.default_sampler)

        if self.is_distributed:
            size_default = size_default // self.world_size

        actual_batch_size = self.batch_size-self.hard_sample_size
        if self.drop_last:
            return size_default // actual_batch_size
        else:
            return (size_default + actual_batch_size - 1) // actual_batch_size


import torch.distributed as dist

class DistributedRandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, device=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.device = device

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            iter_order = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).to(self.device)
        else:
            iter_order = torch.randperm(n).to(self.device)

        # ensure order is the same for all processes (use iter from rank-0)
        dist.broadcast(iter_order,0)

        return iter(iter_order.tolist())

    def __len__(self):
        return self.num_samples


class DistributedSubsetRandomSampler(Sampler):
    def __init__(self, indices, device=None):
        self.indices = indices
        self.device = device

    def __iter__(self):
        iter_order = torch.randperm(len(self.indices)).to(self.device)

        # ensure order is the same for all processes (use iter from rank-0)
        dist.broadcast(iter_order,0)

        return (self.indices[i.item()] for i in iter_order)

    def __len__(self):
        return len(self.indices)

def distributed_sync_dict(array, world_size, rank, device, MAX_LENGTH=10*2**20): # default MAX_LENGTH = 10MB
    def _pack_data(_array):
        data = pickle.dumps(_array)
        data_length = int(len(data))
        data = data_length.to_bytes(4, "big") + data
        assert len(data) < MAX_LENGTH
        data += bytes(MAX_LENGTH - len(data))
        data = np.frombuffer(data, dtype=np.uint8)
        assert len(data) == MAX_LENGTH
        return torch.from_numpy(data)
    def _unpack_data(_array):
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        return pickle.loads(data[4:data_length+4])
    def _unpack_size(_array):
        print(_array.shape, _array[:4])
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        print(data_length,data[:4])
        return data_length

    # prepare output buffer
    output_tensors = [torch.zeros(MAX_LENGTH, dtype=torch.uint8, device=device) for _ in range(world_size)]
    # pack data using pickle into input/output
    output_tensors[rank][:] = _pack_data(array)

    # sync data
    dist.all_gather(output_tensors, output_tensors[rank])

    # unpack data and merge into single dict
    return {id:val for array_tensor in output_tensors for id,val in _unpack_data(array_tensor).items()}