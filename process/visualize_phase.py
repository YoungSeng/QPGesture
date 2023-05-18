import pdb
import sys
[sys.path.append(i) for i in ['.', '..', '../codebook']]
import codebook.Library.Plotting as plot

import matplotlib.pyplot as plt
import torch

import numpy as np
import os


def read_wav(path_wav):
    import wave
    f = wave.open(path_wav, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]  # 通道数、采样字节数、采样率、采样帧数
    voiceStrData = f.readframes(nframes)
    waveData = np.frombuffer(voiceStrData, dtype=np.short)  # 将原始字符数据转换为整数
    waveData = waveData * 1.0 / max(abs(waveData))  # 音频数据归一化, instead of .fromstring
    waveData = np.reshape(waveData, [nframes, nchannels]).T  # .T 表示转置, 将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
    f.close()
    return waveData, nframes, framerate

def draw_time_domain_image(waveData, nframes, framerate):       # 时域特征
    time = np.arange(0,nframes) * (1.0/framerate)
    plt.plot(time,waveData[0,:],c='b')
    plt.xlabel('time')
    plt.ylabel('am')
    # plt.show()
    plt.savefig('1.jpg')


def draw_1():
    for k in range(phase_channels):
        phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze(1)[:, k]
        amps = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze(1)[:, k]
        plot.Phase2D_mono(ax2[k], phase.detach().cpu(), amps.detach().cpu(), showAxes=False)
    fig2.tight_layout()
    fig2.subplots_adjust(wspace=0, hspace=0.1)  # 调整子图间距
    fig2.savefig('visualize_phase.pdf')

def draw_3():
    for k in range(phase_channels):
        x = []
        y = []
        phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze(1)[:, k]
        amps = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze(1)[:, k]
        x.append(phase.detach().cpu())
        y.append(amps.detach().cpu())
        phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase_2[:, 0]]).squeeze(1)[:, k]
        amps = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase_2[:, 2]]).squeeze(1)[:, k]
        x.append(phase.detach().cpu())
        y.append(amps.detach().cpu())
        phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase_3[:, 0]]).squeeze(1)[:, k]
        amps = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase_3[:, 2]]).squeeze(1)[:, k]
        x.append(phase.detach().cpu())
        y.append(amps.detach().cpu())
        plot.Phase2D_mono(ax2[k], x, y, showAxes=False, topk=True)
    fig2.tight_layout()
    fig2.subplots_adjust(wspace=0, hspace=0.1)  # 调整子图间距
    fig2.savefig('visualize_phase_3.pdf')

# if __name__ == '__main__':
#     phase_channels = 8
#     save_dir = "../data/" + 'BEAT0909'
#     prefix = 'speaker_1_state_0'      # speaker_ID:1, self_talk:0
#     phase_data = np.load('../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_test_240_txt.npz', allow_pickle=True)['phase']
#
#     init_i = np.random.randint(0, 10)
#     init_j = np.random.randint(0, 240 - 32)
#     candidates_phase = phase_data[init_i][init_j:init_j + 32]
#     init_i = np.random.randint(0, 10)
#     init_j = np.random.randint(0, 240 - 32)
#     candidates_phase_2 = phase_data[init_i][init_j:init_j + 32]
#     init_i = np.random.randint(0, 10)
#     init_j = np.random.randint(0, 240 - 32)
#     candidates_phase_3 = phase_data[init_i][init_j:init_j + 32]
#     fig2, ax2 = plt.subplots(phase_channels, 1, figsize = (1.2, 4))
#     pdb.set_trace()
#     draw_1()
#     pdb.set_trace()
#     draw_3()


if __name__ == '__main__':
    path_wav = ".\source.wav"
    waveData, nframes, framerate = read_wav(path_wav)
    draw_time_domain_image(waveData, nframes, framerate)

