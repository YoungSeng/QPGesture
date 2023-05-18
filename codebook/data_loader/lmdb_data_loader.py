import logging
import os
import pdb

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import sys
[sys.path.append(i) for i in ['.', '..']]

from data_loader.data_preprocessor import DataPreprocessor
import pyarrow


class TrinityDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, data_mean, data_std, model=None,
                 file='ag', select='specific'):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None
        self.data_mean = np.array(data_mean).squeeze()
        self.data_std = np.array(data_std).squeeze()
        self.file = file

        logging.info("Reading data '{}'...".format(lmdb_dir))
        if model is not None:
            if 'PAE_' in model:
                preloaded_dir = lmdb_dir + '_cache_' + model.split('_')[-1] + '_1'
        else:
            preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps, file=file, select=select)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        # map_size = 1024 * 20  # in MB
        # map_size <<= 20  # in B
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)      # default 10485760
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            # pose_seq, audio, codes, aux_info = sample
            pose_seq, audio, aux_info = sample

        # # normalize
        # std = np.clip(self.data_std, a_min=0.01, a_max=None)
        # pose_seq = (pose_seq - self.data_mean) / std

        # to tensors
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        # codes = torch.from_numpy(codes).long()
        audio = torch.from_numpy(audio).float()

        # return pose_seq, aux_info, codes, audio
        return pose_seq, aux_info, audio


if __name__ == '__main__':
    '''
    cd codebook/
    python data_loader/lmdb_data_loader.py --config=./configs/codebook.yml --no_cuda 0 --gpu 0
    '''

    from configs.parse_args import parse_args
    import os
    import yaml
    from pprint import pprint
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    args = EasyDict(config)

    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate,
                                   data_mean=args.data_mean, data_std=args.data_std, file='g', select='all_speaker')
    train_loader = DataLoader(dataset=train_dataset, batch_size=128,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    print(len(train_loader))
    for batch_i, batch in enumerate(train_loader, 0):
        target_vec, aux, audio = batch
        print(batch_i)
        pdb.set_trace()
