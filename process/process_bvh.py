import os
import pdb

import numpy as np

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *

from scipy.spatial.transform import Rotation as R
from sklearn.pipeline import Pipeline
import joblib as jl
from scipy.signal import savgol_filter


target_joints = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
                 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']


def process_bvh_GENEA2020_BT(gesture_filename):
    '''
    reference from https://github.com/youngwoo-yoon/Co-Speech_Gesture_Generation/blob/3b4e641659623377da9e69add6ca13d7c3d35a49/scripts/trinity_data_to_lmdb.py
    '''
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=20, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('mir', Mirror(axis='X', append=True)),
        ('jtsel', JointSelector(target_joints, include_root=True)),
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    if not os.path.exists('./resource'):
        os.mkdir('./resource')
    jl.dump(data_pipe, os.path.join('./resource', 'data_pipe_BT.sav'))

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 3))
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 9))
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            r = R.from_euler('ZXY', out_data[i, j], degrees=True)
            out_matrix[i, j] = r.as_matrix().reshape(out_data.shape[2], 9)
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0], out_matrix[1]


def make_bvh_GENEA2020_BT(save_path, filename_prefix, poses, smoothing=True, pipeline_path='./resource/data_pipe_60.sav'):
    writer = BVHWriter()
    # pipeline = jl.load('./resource/data_pipe_BT.sav')
    pipeline = jl.load(pipeline_path)

    # smoothing
    if smoothing:
        n_poses = poses.shape[0]
        out_poses = np.zeros((n_poses, poses.shape[1]))
        for i in range(poses.shape[1]):
            out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal
    else:
        out_poses = poses

    # rotation matrix to euler angles
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 9))
    out_poses = out_poses.reshape((out_poses.shape[0], out_poses.shape[1], 3, 3))
    out_euler = np.zeros((out_poses.shape[0], out_poses.shape[1] * 3))
    for i in range(out_poses.shape[0]):  # frames
        r = R.from_matrix(out_poses[i])
        out_euler[i] = r.as_euler('ZXY', degrees=True).flatten()

    bvh_data = pipeline.inverse_transform([out_euler])

    out_bvh_path = os.path.join(save_path, filename_prefix + '_generated.bvh')
    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)


def process_bvh_GENEA2020_BA(bvh_dir, dest_dir, fps=20):
    '''
    Modified from https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder/tree/GENEA_2020/data_processing
    '''
    p = BVHParser()
    data_all = list()
    for f in sorted(os.listdir(bvh_dir)):
        print(f)
        data_all.append(p.parse(os.path.join(bvh_dir, f)))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('mir', Mirror(axis='X', append=True)),
        ('jtsel', JointSelector(target_joints, include_root=True)),
        ('exp', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    assert len(out_data) == 2 * len(os.listdir(bvh_dir))

    if not os.path.exists('./resource'):
        os.mkdir('./resource')
    jl.dump(data_pipe, os.path.join('./resource' , 'data_pipe_AT.sav'))

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    fi = 0
    for f in sorted(os.listdir(bvh_dir)):
        print(f)
        np.save(os.path.join(dest_dir, f[:-4] + '.npy'), out_data[fi])
        np.save(os.path.join(dest_dir, f[:-4] + '_mirrored.npy'), out_data[len(os.listdir(bvh_dir))+fi])
        fi += 1


def make_bvh_GENEA2020_BA(save_path, feat_file):
    features = np.load(feat_file)
    print("Original features shape: ", features.shape)

    pipeline = jl.load('./resource/data_pipe_AT.sav')
    bvh_data = pipeline.inverse_transform([features])
    writer = BVHWriter()
    with open(save_path, 'w') as f:
        writer.write(bvh_data[0], f)


def process_bvh_BEAT(gesture_filename):
    pose_each_file = []
    with open(gesture_filename, "r") as pose_data:
        for j, line in enumerate(pose_data.readlines()):
            data = np.fromstring(line, dtype=float, sep=" ")  # 1*27 e.g., 27 rotation
            pose_each_file.append(data)
    pose_each_file = np.array(pose_each_file)  # n frames * 27
    return pose_each_file


if __name__ == '__main__':
    '''
    cd process/
    python process_bvh.py
    '''

    '''
    bvh_file = "/mnt/nfs7/y50021900/My/data/Trinity_Speech-Gesture_I/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh"
    poses, poses_mirror = process_bvh_GENEA2020_BT(bvh_file)
    # 7730 frames 60fps -> (2577, 135(15*9)) 20fps, 5640000/48000=117.5 117.5*20=2350 117.5*60=7050
    
    save_path = "/mnt/nfs7/y50021900/My/tmp"
    filename_prefix = 'tmp_BT'
    make_bvh_GENEA2020_BT(save_path, filename_prefix, poses, smoothing=False)
    # make_npy()
    '''


    '''
    bvh_dir = "/mnt/nfs7/y50021900/My/tmp/TEST/bvh"
    dest_dir = "/mnt/nfs7/y50021900/My/tmp/TEST/npy"
    process_bvh_GENEA2020_BA(bvh_dir, dest_dir)
    # -> (2577, 45(15*3)) 20fps

    save_path = "/mnt/nfs7/y50021900/My/tmp"
    filename_prefix = 'tmp_BA'
    make_bvh_GENEA2020_BA(save_path=os.path.join(save_path, filename_prefix+'_generated.bvh'),
                          feat_file=os.path.join(dest_dir, 'TestSeq001.npy'))
    '''
    # bvh_file = "/mnt/nfs7/y50021900/My/tmp/tmp_BT_generated.bvh"
    # p = BVHParser()
    # gesture_filename = p.parse(bvh_file)
    # print_skel(gesture_filename)

    # bvh_file = "/mnt/nfs7/y50021900/My/beat-main/datasets/speakers/1/1_wayne_0_103_110.bvh"
    # poses = process_bvh_GENEA2020_BT(bvh_file)
    # pdb.set_trace()
    # np.save("/mnt/nfs7/y50021900/My/tmp/TEST/rotation_matrix.npy", poses)       # (2577, 135)

    # bvh_file = "/mnt/nfs7/y50021900/My/data/Trinity_Speech-Gesture_I/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh"
    # poses, poses_mirror = process_bvh_GENEA2020_BT(bvh_file)
    poses = np.load("/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/11/generate11.npy")
    # np.save("/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/11/TestSeq001_rotation_matrix.npy", poses)
    save_path = "/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/11"
    filename_prefix = 'generate11'
    make_bvh_GENEA2020_BT(save_path, filename_prefix, poses, smoothing=False)
