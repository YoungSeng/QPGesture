import os
import pdb
import numpy as np
import yaml
from pprint import pprint
from easydict import EasyDict
import torch
import math
import time
import torch.nn as nn
import argparse
import sys

from models.vqvae import VQVAE
[sys.path.append(i) for i in ['.', '..', '../process']]
from process.beat_data_to_lmdb import process_bvh
from process.process_bvh import make_bvh_GENEA2020_BT
from process.bvh_to_position import bvh_to_npy
from process.visualize_bvh import visualize


def main(args, source_pose, model_path, save_path, prefix, normalize=True):
    # source_pose = source_pose[:3600]        # 60s, 60FPS

    # normalize
    if normalize:
        data_mean = np.array(args.data_mean).squeeze()
        data_std = np.array(args.data_std).squeeze()
        std = np.clip(data_std, a_min=0.01, a_max=None)
        source_pose = (source_pose - data_mean) / std

    clip_length = source_pose.shape[0]

    # divide into synthesize units and do synthesize
    unit_time = args.n_poses

    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / unit_time) + 1

    print('{}, {}, {}'.format(num_subdivision, unit_time, clip_length))

    with torch.no_grad():
        # model = VQVAE(args.VQVAE, 15 * 3)  # n_joints * n_chanels
        model = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        # for i in range(0, num_subdivision):
        for i in range(0, 512):
            start_time = i * unit_time

            # prepare pose input
            pose_start = math.floor(start_time)
            pose_end = pose_start + args.n_poses
            in_pose = source_pose[pose_start:pose_end]
            if len(in_pose) < args.n_poses:
                if i == num_subdivision - 1:
                    end_padding_duration = args.n_poses - len(in_pose)
                in_pose = np.pad(in_pose, [(0, args.n_poses - len(in_pose)), (0, 0)], mode='constant')
            in_pose = torch.from_numpy(in_pose).unsqueeze(0).to(mydevice)
            # zs = model.module.encode(in_pose.float())
            # zs = [torch.arange(0, 512).unsqueeze(0).to(mydevice)]
            zs = [torch.tensor([i] * 30).unsqueeze(0).to(mydevice)]
            # zs = [torch.from_numpy(np.load('/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/code8.npy').flatten()).unsqueeze(0).to(mydevice)]
            pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()
            # pose_sample, _, _ = model.module(in_pose.float())
            # pose_sample = pose_sample.squeeze(0).data.cpu().numpy()

            code.append(zs[0].squeeze(0).data.cpu().numpy())
            result.append(pose_sample)
            # break
    # np.savez_compressed('/mnt/nfs7/y50021900/My/codebook/BEAT_output_60fps_rotation/code.npz', code=np.array(code), poses=np.array(result))
    out_code = np.vstack(code)
    out_poses = np.vstack(result)

    if normalize:
        out_poses = np.multiply(out_poses, std) + data_mean
    print(out_poses.shape)
    print(out_code.shape)
    np.save(os.path.join(save_path, 'code' + prefix + '.npy'), out_code)
    np.save(os.path.join(save_path, 'generate' + prefix + '.npy'), out_poses)
    return out_poses, out_code


def cal_distance(args, model_path, save_path, prefix, normalize=True):

    with torch.no_grad():
        # model = VQVAE(args.VQVAE, 15 * 3)  # n_joints * n_chanels
        model = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        # for i in range(0, num_subdivision):
        for i in range(0, 512):
            # prepare pose input
            zs = [torch.tensor([i] * 30).unsqueeze(0).to(mydevice)]
            pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()
            code.append(zs[0].squeeze(0).data.cpu().numpy())
            result.append(pose_sample)
    # code: (512, 30)
    # poses: (512, 240, 135)
    np.savez_compressed('./output/code.npz', code=np.array(code), poses=np.array(result), signature=np.mean(np.array(result), axis=1))


def visualize_code(args, model_path, save_path, prefix, code_source, normalize=True):
    # source_pose = source_pose[:3600]        # 60s, 60FPS

    # normalize
    if normalize:
        data_mean = np.array(args.data_mean).squeeze()
        data_std = np.array(args.data_std).squeeze()
        std = np.clip(data_std, a_min=0.01, a_max=None)

    with torch.no_grad():
        model = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        zs = [torch.from_numpy(code_source.flatten()).unsqueeze(0).to(mydevice)]
        pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()

        code.append(zs[0].squeeze(0).data.cpu().numpy())
        result.append(pose_sample)

    out_code = np.vstack(code)
    out_poses = np.vstack(result)

    if normalize:
        out_poses = np.multiply(out_poses, std) + data_mean
    print(out_poses.shape)
    print(out_code.shape)
    np.save(os.path.join(save_path, 'code' + prefix + '.npy'), out_code)
    np.save(os.path.join(save_path, 'generate' + prefix + '.npy'), out_poses)
    return out_poses, out_code


def visualize_PCA_codebook(signature_path, pic_save_path):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    signature = np.load(signature_path)['signature']
    codebook_size = signature.shape[0]
    c2s = []
    print(codebook_size)
    for i in range(codebook_size):
        c2s.append(signature[i])

    pca = PCA()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca)
    ])
    Xt = pipe.fit_transform(c2s)
    plt.figure()
    plt.scatter(Xt[:, 0], Xt[:, 1], label='code')
    plt.legend()
    plt.title("PCA of Codebook")
    plt.savefig(pic_save_path + 'PCA_w_scaler.jpg')


def visualize_code_freq(code, output_path):
    from collections import Counter
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 2))
    print(code.shape)
    code = code.flatten()

    result = Counter(code)
    result_sorted = sorted(result.items(), key=lambda item: item[1], reverse=True)

    x = []
    y = []
    for d in result_sorted[:15]:
        x.append(str(d[0]))
        y.append(d[1])

    p1 = plt.bar(x[0:len(x)], y[0:len(x)])
    plt.bar_label(p1, label_type='edge')
    plt.tight_layout()
    plt.savefig(output_path + 'visualize_code_freq_top15.jpg')


def clip_code_unit(video_path, save_path):
    import subprocess
    import os

    delta_X = 4  # 每10s切割

    mark = 0

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 获取视频的时长
    def get_length(filename):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", filename],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        return float(result.stdout)

    min = int(get_length(video_path)) // 60  # file_name视频的分钟数
    second = int(get_length(video_path)) % 60  # file_name视频的秒数
    totol_sec = int(get_length(video_path))

    print(min, second, totol_sec)

    for i in range(0, totol_sec, delta_X):

        min_start = str(i // 60)
        start = str(i % 60)
        min_end = str((i + delta_X) // 60)
        end = str((i + delta_X) % 60)

        # crop video
        # 保证两位数
        if len(str(min_start)) == 1:
            min_start = '0' + str(min_start)
        if len(str(min_end)) == 1:
            min_end = '0' + str(min_end)
        if len(str(start)) == 1:
            start = '0' + str(start)
        if len(str(end)) == 1:
            end = '0' + str(end)

        # 设置保存视频的名字

        name = str(mark)
        command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -strict -2 {}'.format(
            video_path,
            min_start, start, min_end, end,
            os.path.join(save_path, name) + '.mp4')
        mark += 1
        os.system(command)


def pick_code_freq(train_code, code_int, topk=10, txt_dataset_path=None):
    from collections import Counter

    print(train_code.shape)

    line_code_count = {}
    for line in range(len(train_code)):
        result = Counter(train_code[line])
        line_code_count[line] = result[code_int]

    dataset = np.load(txt_dataset_path, allow_pickle=True)['aux']

    print([[dataset[i[0]],i[1]] for i in sorted(line_code_count.items(), key=lambda item: item[1], reverse=True)[:topk]])


def pick_code_txt(train_code, code_int=None, txt_dataset_path=None, stride=240, num_frames_code=30, fps=60, codebook_size=512, topk=3):
    from collections import Counter

    dataset = np.load(txt_dataset_path, allow_pickle=True)
    aux = dataset['aux']
    txt = dataset['txt']

    print(train_code.shape)

    code_txt = []

    # reshape txt
    step_sz = int(stride / num_frames_code)
    stride_time = stride//fps
    for line in txt:
        tmp_code_txt = [[] for _ in range(num_frames_code)]
        while line != []:
            tmp = line.pop(0)
            tmp_code_txt[int((tmp[0] % stride_time+ (tmp[1] % stride_time if tmp[1] % stride_time != 0 else stride_time)) * 60 / 2 / step_sz)].append(tmp)      # Prevent n*stride_time from being treated as 0
        code_txt.append(tmp_code_txt)

    # init code txt
    c2txt = {}
    txt2c = {}
    for i in range(codebook_size):
        c2txt[i] = []

    for i in range(train_code.shape[0]):       # for every stride
        for j in range(num_frames_code):        # for every code
            for tmp_code_txt in code_txt[i][(j-3 if j-3 > 0 else 0):(j+4 if j+4 < num_frames_code else num_frames_code)]:
                for tmp in tmp_code_txt:
                    c2txt[train_code[i][j]].append(tmp[2])
                    if tmp[2] not in txt2c:
                        txt2c[tmp[2]] = [train_code[i][j]]
                    else:
                        txt2c[tmp[2]].append(train_code[i][j])

    for i in range(codebook_size):
        count_txt = Counter(c2txt[i])
        count_txt = sorted(count_txt.items(), key=lambda item: item[1], reverse=True)[:topk]
        c2txt[i] = count_txt
        if count_txt == []:
            del c2txt[i]

    c2txt = sorted(c2txt.items(), key=lambda item: item[1][0][1], reverse=True)[:topk]

    # print(c2txt)

    for i in txt2c.keys():
        count_code = Counter(txt2c[i])
        count_code = sorted(count_code.items(), key=lambda item: item[1], reverse=True)[:topk]
        txt2c[i] = count_code

    pdb.set_trace()
    txt2c = sorted(txt2c.items(), key=lambda item: item[1][0][1], reverse=True)[:topk]


def visualizeCodeAndWrite(code_path=None, save_path="./Speech2GestureMatching/output/",
                          prefix=None, pipeline_path="../data/data_pipe_60_rotation.sav",
                          generateGT=True, code_source=None, vis=True):

    bvh_path = '../data/BEAT0909/Motion/1_wayne_0_103_110.bvh'
    model_path = config.VQVAE_model_path
    # bvh_path = "/mnt/nfs7/y50021900/My/data/Trinity_Speech-Gesture_I/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh"
    # model_path = '/mnt/nfs7/y50021900/My/codebook/Trinity_output_60fps_rotation/train_codebook/' + "codebook_checkpoint_best.bin"
    # pipeline_path = '../process/resource/data_pipe_60.sav'
    save_path = os.path.join(save_path, prefix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if generateGT:
        print('process bvh...')
        if not os.path.exists(os.path.join(save_path, 'rotation' + 'GT' + '.npy')):
            poses, _ = process_bvh(bvh_path, modetype='rotation')
            np.save(os.path.join(save_path, 'rotation' + 'GT' + '.npy'), poses)
        else:
            print('npy already exists!')
            poses = np.load(os.path.join(save_path, 'rotation' + 'GT' + '.npy'))
        make_bvh_GENEA2020_BT(save_path, filename_prefix='GT', poses=poses, smoothing=False, pipeline_path=pipeline_path)
    print('inference code and rotation pose...')

    if code_source is None:
        code_source = np.load(code_path)['knn_pred']

    out_poses, out_code = visualize_code(config, model_path, save_path, prefix, code_source, normalize=True)
    print('rotation npy to bvh...')
    make_bvh_GENEA2020_BT(save_path, prefix, out_poses, smoothing=False, pipeline_path=pipeline_path)

    if vis:
        print('bvh to position npy...')
        bvh_path_generated = os.path.join(save_path, prefix + '_generated.bvh')
        bvh_to_npy(bvh_path_generated, save_path)
        print('visualize code...')
        npy_generated = np.load(os.path.join(save_path, prefix + '_generated.npy'))
        out_video = os.path.join(save_path, prefix + '_generated.mp4')
        visualize(npy_generated.reshape((npy_generated.shape[0], -1, 3)), out_video, out_code.flatten(), 'upper')


if __name__ == '__main__':

    from configs.parse_args import parse_args
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    mydevice = torch.device('cuda:' + config.gpu)
    config.no_cuda = config.gpu

    if config.stage == 'train':
        cal_distance(config, model_path=config.VQVAE_model_path, save_path=None,
                     prefix=None, normalize=True)
    elif config.stage == 'inference':
        visualizeCodeAndWrite(code_path=config.code_path, prefix=config.prefix, generateGT=False, save_path=config.save_path)



    # code_source = np.array([34, 34, 34, 34, 34, 34] * 5)      # len=30
    # visualizeCodeAndWrite(code_source=code_source, prefix='code_' + str(code_source[:6])[1:-1].replace(' ', '_'), generateGT=False)

    # code_source = np.array([34, 34, 34, 34, 34, 34] * 5)      # len=30
    # visualizeCodeAndWrite(code_source=code_source, prefix='code_' + str(code_source[:6])[1:-1].replace(' ', '_'), generateGT=False)



    # # signature_path = './BEAT_output_60fps_rotation/code.npz'
    # # pic_save_path = './BEAT_output_60fps_rotation/'
    # # visualize_PCA_codebook(signature_path, pic_save_path)
    #
    # train_code_path = '../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240_code.npz'
    # train_code = np.load(train_code_path)['code']
    # pic_save_path = './BEAT_output_60fps_rotation/'
    # # visualize_code_freq(train_code, pic_save_path)
    # txt_dataset_path = '../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240_txt.npz'
    # # pick_code_freq(train_code, code_int=318, topk=10, txt_dataset_path=txt_dataset_path)
    # pick_code_txt(train_code, code_int=None, txt_dataset_path=txt_dataset_path)
    #
    # # video_path = './BEAT_output_60fps_rotation/0001-122880.mkv'  # 待切割视频存储目录
    # # clip_unit_video_path = './BEAT_output_60fps_rotation/clip_unit/'
    # # clip_code_unit(video_path, clip_unit_video_path)
