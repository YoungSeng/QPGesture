import pdb

import Library.Utility as utility
import Library.Plotting as plot
import Library.AdamWR.adamw as adamw
import Library.AdamWR.cyclic_scheduler as cyclic_scheduler
import math
import numpy as np
import torch        # >=1.8.0 to use torch.fft.rfftfreq
from torch.nn.parameter import Parameter
import torch.nn as nn
import random
from datetime import datetime
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from data_loader.lmdb_data_loader import TrinityDataset
import time
import yaml
from pprint import pprint
from easydict import EasyDict
from configs.parse_args import parse_args



# Start Parameter Section
window = 4.0  # time duration of the time window        # 4.0 -> 2.0
frames = 240  # sample count of the time window (60FPS)     # 241 -> 121
keys = 13  # optional, used to rescale the FT window to resolution for motion controller training afterwards
joints = 15

number_of_channels = 9
input_channels = number_of_channels * joints  # number of channels along time in the input data (here 3*J as XYZ-velocity component of each joint)
phase_channels = 8  # desired number of latent phase channels (usually between 2-10)

batch_size = 1
learning_rate = 1e-4
weight_decay = 1e-4
restart_period = 10
restart_mult = 2

plotting_interval = 600  # update visualization at every n-th batch (visualization only) 100
pca_sequence_count = 10  # number of motion sequences visualized in the PCA (visualization only) 10
test_sequence_length = 160
# End Parameter Section
indices = 128
loss_weight = 300


class Model(nn.Module):
    def __init__(self, input_channels, embedding_channels, time_range, key_range, window):
        super(Model, self).__init__()
        self.input_channels = input_channels  # 3 * joints
        self.embedding_channels = embedding_channels  # 8
        self.time_range = time_range  # 121
        self.key_range = key_range  # 13

        self.window = window  # 2.0
        self.time_scale = key_range / time_range

        self.tpi = Parameter(torch.from_numpy(np.array([2.0 * np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(
            torch.from_numpy(np.linspace(-self.window / 2, self.window / 2, self.time_range, dtype=np.float32)),
            requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * (time_range * self.time_scale) / self.window,
                               requires_grad=False)  # Remove DC frequency

        intermediate_channels = int(input_channels / number_of_channels)

        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1,
                               padding=int((time_range) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.conv2 = nn.Conv1d(intermediate_channels, embedding_channels, time_range, stride=1,
                               padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.bn_conv2 = nn.BatchNorm1d(num_features=embedding_channels)

        self.fc = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))
            self.bn.append(nn.BatchNorm1d(num_features=2))

        self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1,
                                 padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True,
                                 padding_mode='zeros')
        self.bn_deconv1 = nn.BatchNorm1d(num_features=intermediate_channels)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1,
                                 padding=int((time_range) / 2), dilation=1, groups=1, bias=True,
                                 padding_mode='zeros')

    def atan2(self, y, x):
        tpi = self.tpi
        ans = torch.atan(y / x)
        ans = torch.where((x < 0) * (y >= 0), ans + 0.5 * tpi, ans)
        ans = torch.where((x < 0) * (y < 0), ans - 0.5 * tpi, ans)
        return ans

    # Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]  # Spectrum without DC component
        power = spectrum ** 2

        # Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        freq = freq / self.time_scale

        # Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        # Offset
        offset = rfft.real[:, :, 0] / self.time_range  # DC component

        return freq, amp, offset

    def forward(self, x):
        y = x
        # Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)     # [1, 135, 240]
        y = self.conv1(y)
        y = self.bn_conv1(y)
        y = torch.tanh(y)

        y = self.conv2(y)
        y = self.bn_conv2(y)
        y = torch.tanh(y)

        latent = y  # Save latent for returning

        # Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        # Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:, i, :])
            v = self.bn[i](v)
            p[:, i] = self.atan2(v[:, 1], v[:, 0]) / self.tpi

        # Parameters
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b]  # Save parameters for returning

        # Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y  # Save signal for returning

        # Signal Reconstruction
        y = self.deconv1(y)
        y = self.bn_deconv1(y)
        y = torch.tanh(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels * self.time_range)

        return y, latent, signal, params


def visualize_bvh(args, network, model_path, bvh_file_path):
    def Item(value):
        return value.detach().cpu()

    source_pose = np.load(bvh_file_path)['upper']
    # network = utility.ToDevice(Model(
    #     input_channels=input_channels,
    #     embedding_channels=phase_channels,
    #     time_range=frames,
    #     key_range=keys,
    #     window=window
    # ))
    network = network.to(mydevice)
    checkpoint = torch.load(model_path)
    network.load_state_dict(checkpoint['model_dict'])
    network = network.eval()

    data_mean = np.array(args.data_mean).squeeze()
    data_std = np.array(args.data_std).squeeze()
    std = np.clip(data_std, a_min=0.01, a_max=None)
    source_pose = (source_pose - data_mean) / std

    clip_length = source_pose.shape[0]

    # divide into synthesize units and do synthesize
    unit_time = 1

    num_subdivision = clip_length - frames + 1

    print('{}, {}, {}'.format(num_subdivision, unit_time, clip_length))

    pca_indices = []
    pca_batches = []
    pivot = 0
    # result = []

    fig2, ax2 = plt.subplots(phase_channels, 1)

    for j in range(0, pca_sequence_count):
        # for i in range(0, num_subdivision):
        subdivision = np.random.choice(num_subdivision)
        for i in range(subdivision, subdivision + test_sequence_length):
            # prepare pose input
            pose_start = i
            pose_end = pose_start + args.PAE.n_poses
            target_vec = source_pose[pose_start:pose_end]

            target_vec = torch.from_numpy(target_vec).unsqueeze(0).float()

            zero_tensor = torch.zeros(target_vec.shape[0], 1, target_vec.shape[2])
            test_batch = torch.cat(((target_vec[:, 1:, :] - target_vec[:, :-1, :]), zero_tensor), 1).transpose(2,
                                                                                                               1).reshape(
                target_vec.shape[0], -1).to(mydevice)

            if i == subdivision:
                input_tensor = test_batch.to(mydevice)
            else:
                input_tensor = torch.vstack((input_tensor, test_batch)).to(mydevice)

        _, _, _, params = network(input_tensor)
        a = Item(params[2]).squeeze()
        p = Item(params[0]).squeeze()
        # Compute Phase Manifold (2D vectors composed of sin and cos terms)
        m_x = a * np.sin(2.0 * np.pi * p)
        m_y = a * np.cos(2.0 * np.pi * p)
        manifold = torch.hstack((m_x, m_y))
        pca_indices.append(pivot + np.arange(len(input_tensor)))
        pca_batches.append(manifold)
        pivot += len(input_tensor)

        for k in range(phase_channels):
            phase = params[0][:, k]
            amps = params[2][:, k]
            plot.Phase2D(ax2[k], Item(phase), Item(amps), title=("2D Phase Vectors" if k == 0 else None),
                         showAxes=False)

        fig2.tight_layout()
        fig2.savefig(args.PAE.figs_save_path + '/visualize_phase.jpg')

    fig4, ax4 = plt.subplots(1, 1)
    plot.PCA2D(ax4, pca_indices, pca_batches,
               "Phase Manifold (" + " 1 Sequences)")
    fig4.savefig(args.PAE.figs_save_path + '/visualize_bvh.jpg')
    plt.gcf().canvas.draw_idle()


def evaluate_testset(network, dataloader):
    start = time.time()
    network = network.eval()
    loss_function = torch.nn.MSELoss()
    tot_error = 0
    tot_eval_nums = 0
    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader, 0):
            target_vec, _, _ = data  # (b, 240, 45), (b, 64000)
            zero_tensor = torch.zeros(target_vec.shape[0], 1, target_vec.shape[2])
            test_batch = torch.cat(((target_vec[:, 1:, :] - target_vec[:, :-1, :]), zero_tensor), 1).transpose(2,
                                                                                                               1).reshape(
                target_vec.shape[0], -1).to(mydevice)

            yPred, _, _, params = network(test_batch)
            loss = loss_weight * loss_function(yPred, test_batch)
            tot_error += loss.data.cpu().numpy()
            tot_eval_nums += 1
        print('generation took {:.2} s'.format(time.time() - start))
    return tot_error / (tot_eval_nums * 1.0)


def main(args, network):
    def Item(value):
        return value.detach().cpu()

    # Initialize dataloader
    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.PAE.n_poses,
                                   subdivision_stride=args.PAE.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate,
                                   data_mean=args.data_mean, data_std=args.data_std,
                                   model='PAE_' + str(args.PAE.n_poses), file='g', select='all_speaker')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    val_dataset = TrinityDataset(args.val_data_path,
                                 n_poses=args.PAE.n_poses,
                                 subdivision_stride=args.PAE.subdivision_stride,
                                 pose_resampling_fps=args.motion_resampling_framerate,
                                 data_mean=args.data_mean, data_std=args.data_std,
                                 model='PAE_' + str(args.PAE.n_poses), file='g', select='all_speaker')
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    print('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(test_loader)))

    sample_count = len(train_loader)

    if not os.path.exists(args.PAE.model_save_path):
        os.mkdir(args.PAE.model_save_path)

    if not os.path.exists(args.PAE.figs_save_path):
        os.mkdir(args.PAE.figs_save_path)

    # Initialize all seeds
    seed = 23456
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Initialize drawing
    plt.ion()
    fig1, ax1 = plt.subplots(6, 1)
    fig2, ax2 = plt.subplots(phase_channels, 5)
    fig3, ax3 = plt.subplots(1, 2)
    fig4, ax4 = plt.subplots(2, 1)
    dist_amps = []
    dist_freqs = []
    loss_history = utility.PlottingWindow("Loss History", ax=ax4, min=0, max=1, drawInterval=plotting_interval)

    # Setup optimizer and loss function
    optimizer = adamw.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=batch_size,
                                                      epoch_size=sample_count, restart_period=restart_period,
                                                      t_mult=restart_mult, policy="cosine", verbose=True)
    loss_function = torch.nn.MSELoss()

    updates = 0
    best_val_loss = (1e+6, 0)  # value, epoch
    for epoch in range(args.PAE.epochs):

        loss_eval = evaluate_testset(network, test_loader)
        print('loss on validation: {:.3f}'.format(loss_eval))
        is_best = loss_eval < best_val_loss[0]

        if is_best:
            print(' *** BEST VALIDATION LOSS : {:.3f}'.format(loss_eval))
            best_val_loss = (loss_eval, epoch)
        else:
            print(' best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        if is_best:
            save_name = '{}/{}_checkpoint_best.bin'.format(args.PAE.model_save_path, args.PAE.name)

            torch.save({
                'args': args, "epoch": epoch, 'model_dict': network.state_dict()
            }, save_name)
            print('Saved the checkpoint')

        if epoch % args.PAE.save_per_epochs == 0:
            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.PAE.model_save_path, args.PAE.name, epoch)

            torch.save({
                'args': args, "epoch": epoch, 'model_dict': network.state_dict()
            }, save_name)
            print('Saved the checkpoint')

        network = network.train()
        scheduler.step()
        start = datetime.now()
        for i, batch in enumerate(train_loader, 0):
            # Run model prediction
            target_vec, _, _ = batch  # (b, T, J*3)

            zero_tensor = torch.zeros(target_vec.shape[0], 1, target_vec.shape[2])
            train_batch = torch.cat((zero_tensor, (target_vec[:, 1:, :] - target_vec[:, :-1, :])), 1).transpose(2,
                                                                                                                1).reshape(
                target_vec.shape[0], -1).to(mydevice)

            yPred, latent, signal, params = network(train_batch)

            # Compute loss and train
            loss = loss_weight * loss_function(yPred, train_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.batch_step()

            # Start Visualization Section
            _a_ = Item(params[2]).squeeze().numpy()
            for j in range(_a_.shape[0]):
                dist_amps.append(_a_[j, :])
            while len(dist_amps) > 10000:
                dist_amps.pop(0)

            _f_ = Item(params[1]).squeeze().numpy()
            for j in range(_f_.shape[0]):
                dist_freqs.append(_f_[j, :])
            while len(dist_freqs) > 10000:
                dist_freqs.pop(0)

            loss_history.Add(
                (Item(loss).item(), "Reconstruction Loss")
            )

            stats = {'updates': updates, 'loss': loss.item()}
            stats_str = ' '.join(f'{key}[{val:.8f}]' for key, val in stats.items())
            i += 1
            remaining = str((datetime.now() - start) / i * (sample_count - i))
            remaining = remaining.split('.')[0]
            print(f'> epoch [{epoch}] updates[{i}] {stats_str} eta[{remaining}]')

            if loss_history.Counter == 0:
                # # <editor-fold desc='plot'>
                # plot.Functions(ax1[0], Item(train_batch[0]).reshape(network.input_channels,frames), -1.0, 1.0, -5.0, 5.0, title="Motion Curves" + " " + str(network.input_channels) + "x" + str(frames), showAxes=False)
                # plot.Functions(ax1[1], Item(latent[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Convolutional Embedding" + " " + str(phase_channels) + "x" + str(frames), showAxes=False)
                # plot.Circles(ax1[2], Item(params[0][0]).squeeze(), Item(params[2][0]).squeeze(), title="Learned Phase Timing"  + " " + str(phase_channels) + "x" + str(2), showAxes=False)
                # plot.Functions(ax1[3], Item(signal[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Parametrized Signal" + " " + str(phase_channels) + "x" + str(frames), showAxes=False)
                # plot.Functions(ax1[4], Item(yPred[0]).reshape(network.input_channels,frames), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction" + " " + str(network.input_channels) + "x" + str(frames), showAxes=False)
                # plot.Function(ax1[5], [Item(train_batch[0]), Item(yPred[0])], -1.0, 1.0, -5.0, 5.0, colors=[(0, 0, 0), (0, 1, 1)], title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(network.input_channels*frames), showAxes=False)
                # plot.Distribution(ax3[0], dist_amps, title="Amplitude Distribution")
                # plot.Distribution(ax3[1], dist_freqs, title="Frequency Distribution")
                #
                # fig1.tight_layout()
                # fig3.tight_layout()
                # fig1.savefig(args.PAE.figs_save_path + '/{}_1.jpg'.format(updates))
                # fig3.savefig(args.PAE.figs_save_path + '/{}_3.jpg'.format(updates))
                #
                # test_batch = train_batch
                # _, _, _, params = network(test_batch)
                #
                # for j in range(phase_channels):
                #     phase = params[0][:,j]
                #     freq = params[1][:,j]
                #     amps = params[2][:,j]
                #     offs = params[3][:,j]
                #     plot.Phase1D(ax2[j,0], Item(phase), Item(amps), color=(0, 0, 0), title=("1D Phase Values" if j==0 else None), showAxes=False)
                #     plot.Phase2D(ax2[j,1], Item(phase), Item(amps), title=("2D Phase Vectors" if j==0 else None), showAxes=False)
                #     plot.Functions(ax2[j,2], Item(freq).transpose(0,1), -1.0, 1.0, 0.0, 4.0, title=("Frequencies" if j==0 else None), showAxes=False)
                #     plot.Functions(ax2[j,3], Item(amps).transpose(0,1), -1.0, 1.0, 0.0, 1.0, title=("Amplitudes" if j==0 else None), showAxes=False)
                #     plot.Functions(ax2[j,4], Item(offs).transpose(0,1), -1.0, 1.0, -1.0, 1.0, title=("Offsets" if j==0 else None), showAxes=False)
                # fig2.tight_layout()
                # fig2.savefig(args.PAE.figs_save_path + '/{}_2.jpg'.format(updates))
                # # </editor-fold>

                # Manifold Computation and Visualization
                pca_indices = []
                pca_batches = []
                pivot = 0

                network = network.eval()

                with torch.no_grad():
                    for iter_idx, data in enumerate(test_loader, 0):
                        target_vec, _, _ = data  # (b, 240, J*3), (b, 64000)
                        zero_tensor = torch.zeros(target_vec.shape[0], 1, target_vec.shape[2])
                        test_batch = torch.cat((zero_tensor, (target_vec[:, 1:, :] - target_vec[:, :-1, :])),
                                               1).transpose(2, 1).reshape(target_vec.shape[0], -1).to(mydevice)
                        # test_batch = target_vec.reshape(target_vec.shape[0], -1).to(mydevice)  # (b, 240*15*3)
                        _, _, _, params = network(test_batch)
                        a = Item(params[2]).squeeze()
                        p = Item(params[0]).squeeze()
                        # Compute Phase Manifold (2D vectors composed of sin and cos terms)
                        m_x = a * np.sin(2.0 * np.pi * p)
                        m_y = a * np.cos(2.0 * np.pi * p)
                        manifold = torch.hstack((m_x, m_y))
                        pca_indices.append(pivot + np.arange(indices))
                        pca_batches.append(manifold)
                        pivot += indices
                        if iter_idx >= pca_sequence_count:
                            break
                network = network.train()

                plot.PCA2D(ax4[0], pca_indices, pca_batches,
                           "Phase Manifold (" + str(pca_sequence_count) + " Random Sequences)")
                fig4.savefig(args.PAE.figs_save_path + '/{}.jpg'.format(updates))
                updates += 1
                plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(1e-5)
            # End Visualization Section

        print('Epoch', epoch + 1, loss_history.CumulativeValue())


def pose2phase(network, pose, data_mean, std):
    print(pose.shape)
    n_poses = 240
    pose = (pose - data_mean) / std  # norm
    vel = pose[1:, :] - pose[:-1, :]
    vel = np.pad(vel, ((120, 119), (0, 0)), 'constant', constant_values=((0, 0), (0, 0)))  # padding velocity

    clip_length = vel.shape[0]
    unit_time = 1
    num_subdivision = pose.shape[0]
    print('{}, {}, {}'.format(num_subdivision, unit_time, clip_length))
    print(vel.shape)
    result = []

    for i in range(0, num_subdivision):
        print(str(i) + '\r', end='')
        # prepare pose input
        pose_start = i
        pose_end = pose_start + n_poses - 1
        target_vec = vel[pose_start:pose_end]

        target_vec = torch.from_numpy(target_vec).unsqueeze(0).float()

        zero_tensor = torch.zeros(target_vec.shape[0], 1, target_vec.shape[2])
        test_batch = torch.cat((zero_tensor, target_vec), 1).transpose(2, 1).reshape(target_vec.shape[0], -1).to(
            mydevice)

        _, _, _, params = network(test_batch)       # [1, 32400]
        params = [j.detach().cpu() for j in params]
        result.append(params)

    return np.array(result)


if __name__ == '__main__':
    args = parse_args()
    mydevice = torch.device('cuda:' + args.gpu)
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    config.no_cuda = config.gpu

    network = utility.ToDevice(Model(
        input_channels=input_channels,
        embedding_channels=phase_channels,
        time_range=frames,
        key_range=keys,
        window=window
    ))

    if config.stage == 'train':
        print('stage: train')
        main(config, network)
    elif config.stage == 'inference':
        print('stage: inference')
        # Build network model

        model_path = "../pretrained_model/PAE_checkpoint_070.bin"
        # bvh_file_path = '../data/BEAT0909/speaker_1_state_0/Rotation/1_wayne_0_103_110.npz'
        #
        # visualize_bvh(config, network, model_path, bvh_file_path)

        torch.cuda.set_device(int(config.gpu))
        network = network.to(mydevice)
        checkpoint = torch.load(model_path, map_location='cpu')
        network.load_state_dict(checkpoint['model_dict'])
        network = network.eval()
        data_mean = np.array(config.data_mean).squeeze()
        data_std = np.array(config.data_std).squeeze()
        std = np.clip(data_std, a_min=0.01, a_max=None)

        source_rotation = '../dataset/BEAT/speaker_10_state_0/Rotation'
        save_dir = '../dataset/BEAT/speaker_10_state_0/Phase'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for item in os.listdir(source_rotation):
            if os.path.exists(os.path.join(save_dir, item)):
                print(item, 'exists')
                continue
            pose = np.load(os.path.join(source_rotation, item))['upper']
            phase = pose2phase(network, pose, data_mean, std)
            assert phase.shape[0] == pose.shape[0]
            np.savez_compressed((os.path.join(save_dir, item)), phase=phase)

