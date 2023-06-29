import argparse
import os
import glob
from pathlib import Path
import librosa
import numpy as np
import lmdb
import pyarrow
from sklearn.pipeline import Pipeline
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
from scipy.spatial.transform import Rotation as R
import joblib as jl

target_joints = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
                 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']


def process_bvh(gesture_filename, modetype=None, fps=20, output_data_pipe='data_pipe_20_rotation.sav'):
    p = BVHParser()

    data_all = list()
    try:
        data_all.append(p.parse(gesture_filename))
    except:
        return None

    if modetype == 'position':
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
            ('root', RootTransformer('hip_centric')),
            # ('mir', Mirror(axis='X', append=True)),
            ('jtsel', JointSelector(target_joints, include_root=True)),
            ('param', MocapParameterizer('position')),  # expmap, position
            ('cnst', ConstantsRemover()),
            ('np', Numpyfier())
        ])
        try:
            out_data = data_pipe.fit_transform(data_all)
        except:
            return None

        if not os.path.exists('./resource'):
            os.mkdir('./resource')
        jl.dump(data_pipe, os.path.join('./resource', output_data_pipe))        # 'data_pipe_60_position.sav'

        out_data = out_data[0]
        out_data = np.pad(out_data, ((0, 0), (3, 0)), 'constant', constant_values=(0, 0))
        if out_data.shape[1] != len(target_joints) * 3:
            print(out_data.shape)
            return None
        # assert out_data.shape[1] == 15 * 3
        return out_data

    elif modetype == 'rotation':
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
            ('root', RootTransformer('hip_centric')),
            ('mir', Mirror(axis='X', append=True)),
            ('jtsel', JointSelector(target_joints, include_root=True)),
            ('cnst', ConstantsRemover()),
            ('np', Numpyfier())
        ])
        try:
            out_data = data_pipe.fit_transform(data_all)
        except:
            return None, None

        if out_data[0].shape[1] != len(target_joints) * 3:
            print(out_data.shape)
            return None, None

        if not os.path.exists('./resource'):
            os.mkdir('./resource')
        jl.dump(data_pipe, os.path.join('./resource', output_data_pipe))        # 'data_pipe_60_rotation.sav'

        # euler -> rotation matrix
        out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 3))
        out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 9))
        for i in range(out_data.shape[0]):  # mirror
            for j in range(out_data.shape[1]):  # frames
                r = R.from_euler('ZXY', out_data[i, j], degrees=True)
                out_matrix[i, j] = r.as_matrix().reshape(out_data.shape[2], 9)
        out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

        return out_matrix[0], out_matrix[1]


def make_lmdb_gesture_dataset(base_path, lmdb_name='lmdb', modetyepe=None, remake=True, recode=False, n_frames=240):
    gesture_path = os.path.join(base_path, 'Motion')
    audio_path = os.path.join(base_path, 'Audio')
    rotation_path = os.path.join(base_path, 'Rotation')
    wav_path = os.path.join(base_path, 'Wav')
    mfcc_path = os.path.join(base_path, 'MFCC')

    # text_path = os.path.join(base_path, 'Transcripts')
    out_path = os.path.join(base_path, lmdb_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    map_size = 1024 * 200  # in MB
    map_size <<= 20  # in B
    db = [lmdb.open(os.path.join(out_path, 'lmdb_train'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_valid'), map_size=map_size),
          lmdb.open(os.path.join(out_path, 'lmdb_test'), map_size=map_size)]

    # delete existing files
    for i in range(3):
        with db[i].begin(write=True) as txn:
            txn.drop(db[i].open_db())

    all_poses = []
    bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
    v_i = 0
    e_bvh_i = 0
    e_bvh = []
    e_wav_i = 0
    e_wav = []
    for _, bvh_file in enumerate(bvh_files):
        name = os.path.split(bvh_file)[1][:-4]
        if name.split('_')[2] == '0' or name.split('_')[2] == '1':
            print('process: ' + name)
        else:
            print('skip: ' + name)
            continue

        if remake:
            # load skeletons and subtitles
            if modetyepe == 'rotation':
                poses, poses_mirror = process_bvh(bvh_file, modetyepe, fps=60)
            elif modetyepe == 'position':
                poses = process_bvh(bvh_file, modetyepe)
        else:
            poses = np.load(os.path.join(rotation_path, name + '.npz'))['upper']
        if poses is None:
            # print('gesture error: ' + name)
            e_bvh.append(name)
            e_bvh_i += 1
            continue

        # subtitle = SubtitleWrapper(os.path.join(text_path, name + '.json')).get()

        if remake:
            # load audio
            try:
                audio_raw, audio_sr = librosa.load(os.path.join(audio_path, '{}.wav'.format(name)),
                                                   mono=True, sr=16000, res_type='kaiser_fast')
            except:
                # print('audio error: ' + name)
                e_wav.append(name)
                e_wav_i += 1
                continue
        else:
            audio_raw = np.load(os.path.join(wav_path, name + '.npz'))['wav']
            mfcc_raw = np.load(os.path.join(mfcc_path, name + '.npz'))['mfcc']

        # process
        clips = [{'vid': name, 'clips': []},    # train
                 {'vid': name, 'clips': []},    # validation
                 {'vid': name, 'clips': []}]    # test

        if remake:
            # split
            if v_i % 10 == 0:        # 80% for training, 10% for validation and 10% for testing
                dataset_idx = 2  # test
            elif v_i % 10 == 1:
                dataset_idx = 1  # validation
            else:
                dataset_idx = 0  # train

        else:
            if '81_86' in bvh_file:
                continue
            elif '103' in bvh_file:
                dataset_idx = 2  # pick test data
            elif '111' in bvh_file:
                dataset_idx = 1  # validation
            else:
                dataset_idx = 0  # train

        if not remake:
            normalize = True
            data_mean = [0.96776, 0.03511, -0.00725, -0.03507, 0.96983, -0.00267, 0.00705, 0.00363, 0.99770, 0.99930, 0.01376, 0.00255, -0.01374, 0.99938, -0.00253, -0.00263, 0.00236, 0.99987, 0.99216, 0.01860, -0.00709, -0.01882, 0.98965, -0.02039, 0.00531, 0.02222, 0.99601, 0.99612, -0.00993, -0.00998, 0.00991, 0.99691, -0.00743, 0.00991, 0.00820, 0.99890, 0.97768, 0.03605, 0.00750, -0.03840, 0.98164, 0.01540, -0.00722, -0.01537, 0.98648, 0.97946, 0.01763, 0.03590, -0.01997, 0.97667, 0.02363, -0.03636, -0.02139, 0.97316, 0.99365, 0.00565, 0.00280, -0.00546, 0.99544, -0.00042, -0.00264, 0.00288, 0.99802, 0.96583, -0.00000, 0.03884, -0.01122, 0.93705, 0.25010, -0.03542, -0.24257, 0.94337, 0.42426, -0.00898, 0.03954, 0.00077, 0.78974, -0.20711, -0.03733, 0.36827, 0.39807, 0.77377, -0.03384, -0.00000, 0.02925, 0.01928, -0.84132, 0.03604, 0.63766, 0.00386, 0.92148, -0.00060, 0.02677, 0.00000, 0.96406, 0.08089, -0.02854, -0.07688, 0.91554, 0.97416, 0.00000, 0.01114, 0.00433, 0.96865, -0.19804, -0.01083, 0.19909, 0.95347, 0.40921, -0.00936, -0.05575, -0.00286, 0.82451, 0.12718, 0.04597, -0.30625, 0.40531, 0.79050, 0.02049, 0.00000, 0.01737, -0.00978, 0.82851, 0.00535, -0.66105, -0.02535, 0.91757, -0.00631, 0.02182, -0.00000, 0.95817, -0.09851, -0.01829, 0.09236, 0.91430]
            data_std = [0.02841, 0.23917, 0.06431, 0.23880, 0.02782, 0.01986, 0.06575, 0.01430, 0.00336, 0.00108, 0.03157, 0.01468, 0.03155, 0.00106, 0.00621, 0.01473, 0.00612, 0.00018, 0.01512, 0.11668, 0.03705, 0.11614, 0.01536, 0.07804, 0.03890, 0.07677, 0.00516, 0.00416, 0.07577, 0.04228, 0.07578, 0.00407, 0.01605, 0.04228, 0.01571, 0.00136, 0.02789, 0.15938, 0.12886, 0.15728, 0.02378, 0.09677, 0.13076, 0.09421, 0.02422, 0.03618, 0.12665, 0.14733, 0.12803, 0.07007, 0.15448, 0.14572, 0.15622, 0.07448, 0.02085, 0.09297, 0.05959, 0.09228, 0.01465, 0.01851, 0.06068, 0.01429, 0.00739, 0.20466, 0.00001, 0.15420, 0.04271, 0.19514, 0.13914, 0.14860, 0.15817, 0.05350, 0.16375, 0.29095, 0.84077, 0.46422, 0.24234, 0.24331, 0.75914, 0.31183, 0.17603, 0.28407, 0.56519, 0.00013, 0.23171, 0.44004, 0.20889, 0.51453, 0.28072, 0.49853, 0.18833, 0.07924, 0.32926, 0.00007, 0.16923, 0.18816, 0.33851, 0.17250, 0.10350, 0.10469, 0.00000, 0.19984, 0.04231, 0.09819, 0.10514, 0.19528, 0.11157, 0.02373, 0.15367, 0.26760, 0.85681, 0.43448, 0.24673, 0.23313, 0.78615, 0.30631, 0.16722, 0.28200, 0.54330, 0.00016, 0.22882, 0.44907, 0.24319, 0.49285, 0.25625, 0.50377, 0.20667, 0.07828, 0.32970, 0.00034, 0.19347, 0.18646, 0.33914, 0.17255, 0.10199]

            if normalize:
                data_mean = np.array(data_mean).squeeze()
                data_std = np.array(data_std).squeeze()
                std = np.clip(data_std, a_min=0.01, a_max=None)
                poses = (poses - data_mean) / std

            if recode:
                model_path = '../codebook/BEAT_output_60fps_rotation/train_codebook/' + "codebook_checkpoint_best.bin"
                codes_raw = pose2code(poses[:n_frames * int(len(poses) / n_frames)], model_path=model_path)
                print(poses.shape, codes_raw.shape)

        if remake:
            # save subtitles and skeletons
            poses = np.asarray(poses)
            clips[dataset_idx]['clips'].append(
                {  # 'words': word_list,
                    'poses': poses,
                    'audio_raw': audio_raw
                })
        elif recode:
            poses = np.asarray(poses)
            clips[dataset_idx]['clips'].append(
                {  # 'words': word_list,
                    'poses': poses,
                    'audio_raw': audio_raw,
                    'mfcc_raw': mfcc_raw,
                    'code_raw': codes_raw
                })
        else:
            poses = np.asarray(poses)
            clips[dataset_idx]['clips'].append(
                {  # 'words': word_list,
                    'poses': poses,
                    'audio_raw': audio_raw,
                    'mfcc_raw': mfcc_raw,      # for debug
                    'code_raw': np.array([0])       # for debug
                })

        if remake and modetyepe == 'rotation':
            poses_mirror = np.asarray(poses_mirror)
            clips[dataset_idx]['clips'].append(
                {  # 'words': word_list,
                    'poses': poses_mirror,
                    'audio_raw': audio_raw
                })

        # write to db
        for i in range(3):
            with db[i].begin(write=True) as txn:
                if len(clips[i]['clips']) > 0:
                    k = '{:010}'.format(v_i).encode('ascii')
                    v = pyarrow.serialize(clips[i]).to_buffer()
                    txn.put(k, v)

        all_poses.append(poses)
        v_i += 1

    print('total length of training set: ' + str(v_i), '\terror gesture files: ' + str(e_bvh_i), '\terror audio files: ' + str(e_wav_i))
    print('error gesture:')
    print(e_bvh)
    print('error audio:')
    print(e_wav)
    # close db
    for i in range(3):
        db[i].sync()
        db[i].close()

    # calculate data mean
    all_poses = np.vstack(all_poses)
    pose_mean = np.mean(all_poses, axis=0)
    pose_std = np.std(all_poses, axis=0)

    print('data mean/std')
    print(str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print(str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))


def pose2code(poses, model_path, n_frames=240):

    mydevice = torch.device('cuda:' + args.gpu)
    poses = poses.reshape(-1, n_frames, poses.shape[-1])
    with torch.no_grad():
        model = VQVAE(config.VQVAE, 15 * 9)  # n_joints * n_chanels
        model = nn.DataParallel(model, device_ids=[3])
        model = model.to(mydevice)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        code = []

        i = 0
        for seq in poses:
            print(i)
            in_pose = torch.from_numpy(seq).unsqueeze(0).to(mydevice)
            zs = model.module.encode(in_pose.float())
            # pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()
            code.append(zs[0].squeeze(0).data.cpu().numpy())
            i += 1
        codes_raw = np.array(code)
    return codes_raw.reshape(-1, codes_raw.shape[-1])


if __name__ == '__main__':

    import torch.nn as nn
    import sys
    [sys.path.append(i) for i in ['.', '..', '../codebook']]
    from codebook.models.vqvae import VQVAE
    import torch
    from codebook.configs.parse_args import parse_args
    args = parse_args()
    from easydict import EasyDict
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v

    config = EasyDict(config)
    config.no_cuda = config.gpu

    make_lmdb_gesture_dataset(config.beat_data_to_lmdb.path, lmdb_name=config.beat_data_to_lmdb.lmdb_name,
                              modetyepe=config.beat_data_to_lmdb.mode, remake=True, recode=False)
