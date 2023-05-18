import pdb

import logging
logging.getLogger().setLevel(logging.INFO)

from torch.utils.data import DataLoader
from data_loader.lmdb_data_loader import TrinityDataset
import time
import torch
import torch.nn as nn
import yaml
from pprint import pprint
from easydict import EasyDict
from configs.parse_args import parse_args
from models.vqvae import VQVAE as codebook
from generate.generate import Generator_gru as Generator
import os
from torch import optim
import itertools
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


args = parse_args()
mydevice = torch.device('cuda:' + args.gpu)


def evaluate_testset(model, test_data_loader):
    start = time.time()
    model = model.eval()
    tot_error = 0
    tot_eval_nums = 0
    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            _, _, codes, audio = data  # (b, 240, 45), (b, 64000)
            codes = codes.to(mydevice)
            _, loss = model(audio, target=codes)
            tot_error += loss.data.cpu().numpy()
            tot_eval_nums += 1
        print('generation took {:.2} s'.format(time.time() - start))
    model.train()
    return tot_error / (tot_eval_nums * 1.0)


def main(args):
    # dataset

    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate,
                                   data_mean=args.data_mean, data_std=args.data_std)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    val_dataset = TrinityDataset(args.val_data_path,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate,
                                       data_mean=args.data_mean, data_std=args.data_std)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=False)

    logging.info('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(test_loader)))

    model = Generator()
    model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
    model = model.to(mydevice)

    best_val_loss = (1e+2, 0)  # value, epoch

    if not os.path.exists(args.end2end.model_save_path):
        os.mkdir(args.end2end.model_save_path)

    optimizer = optim.Adam(itertools.chain(model.module.parameters()), lr=args.end2end.lr, betas=args.end2end.betas)
    # schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    updates = 0
    total = len(train_loader)

    tb_path = args.end2end.name + '_' + str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    tb_writer = SummaryWriter(log_dir=str(Path(args.end2end.model_save_path).parent / 'tensorboard_runs' / tb_path))


    for epoch in range(1, args.end2end.epochs + 1):
        logging.info(f'Epoch: {epoch}')
        i = 0
        loss_eval = evaluate_testset(model, test_loader)
        logging.info('loss on validation: {:.3f}'.format(loss_eval))

        # tb_writer.add_scalar('loss/validation', loss_eval, epoch)
        is_best = loss_eval < best_val_loss[0]
        if is_best:
            logging.info(' *** BEST VALIDATION LOSS : {:.3f}'.format(loss_eval))
            best_val_loss = (loss_eval, epoch)
        else:
            logging.info(' best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        if is_best or (epoch % args.end2end.save_per_epochs == 0):
            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(args.end2end.model_save_path, args.end2end.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.end2end.model_save_path, args.end2end.name, epoch)

            torch.save({
                'args': args, "epoch": epoch, 'model_dict': model.state_dict()
            }, save_name)
            logging.info('Saved the checkpoint')

        # train model
        model = model.train()
        start = datetime.now()
        for batch_i, batch in enumerate(train_loader, 0):
            _, _, codes, audio = batch
            codes = codes.to(mydevice)

            optimizer.zero_grad()
            _, loss = model(audio, target=codes)

            # write to tensorboard
            # tb_writer.add_scalar('loss' + '/train', loss, updates)
            loss.backward()
            optimizer.step()
            # log
            stats = {'updates': updates, 'loss': loss.item()}
            stats_str = ' '.join(f'{key}[{val:.8f}]' for key, val in stats.items())
            i += 1
            remaining = str((datetime.now() - start) / i * (total - i))
            remaining = remaining.split('.')[0]
            logging.info(f'> epoch [{epoch}] updates[{i}] {stats_str} eta[{remaining}]' + '\r')
            updates += 1
        # schedular.step()
    tb_writer.close()
    # print best losses
    logging.info('--------- Final best loss values ---------')
    logging.info('diff mean: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))


if __name__ == '__main__':

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    main(config)
