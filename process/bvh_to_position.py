import glob
import pdb
import joblib as jl
from sklearn.pipeline import Pipeline
import os
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.writers import *

target_joints = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
                 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

# target_joints = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftForeFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightForeFoot', 'RightToeBase', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'pCube4', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'Neck', 'Neck1', 'Head']

# target_joints = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftForeFoot', 'LeftToeBase', 'LeftToeBaseEnd',
#                  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightForeFoot', 'RightToeBase', 'RightToeBaseEnd', 'Spine',
#                  'Spine1', 'Spine2', 'Spine3', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'RightShoulder', 'RightArm',
#                  'RightForeArm', 'Neck', 'Neck1', 'Head', 'HeadEnd']        # BEAT 27

# joint_name_to_idx = {'Hips': 0, 'LeftUpLeg': 1, 'LeftLeg': 2, 'LeftFoot': 3, 'LeftForeFoot': 4, 'LeftToeBase': 5, 'LeftToeBaseEnd': 6, 'LeftToeBaseEnd_Nub': 7, 'RightUpLeg': 8, 'RightLeg': 9, 'RightFoot': 10, 'RightForeFoot': 11, 'RightToeBase': 12, 'RightToeBaseEnd': 13, 'RightToeBaseEnd_Nub': 14, 'Spine': 15, 'Spine1': 16, 'Spine2': 17, 'Spine3': 18, 'LeftShoulder': 19, 'LeftArm': 20, 'LeftForeArm': 21, 'LeftHand': 22, 'LeftHandIndex': 23, 'LeftHandThumb1': 24, 'LeftHandThumb2': 25, 'LeftHandThumb3': 26, 'LeftHandThumb4': 27, 'LeftHandThumb4_Nub': 28, 'LeftHandIndex1': 29, 'LeftHandIndex2': 30, 'LeftHandIndex3': 31, 'LeftHandIndex4': 32, 'LeftHandIndex4_Nub': 33, 'LeftHandRing': 34, 'LeftHandPinky': 35, 'LeftHandPinky1': 36, 'LeftHandPinky2': 37, 'LeftHandPinky3': 38, 'LeftHandPinky4': 39, 'LeftHandPinky4_Nub': 40, 'LeftHandRing1': 41, 'LeftHandRing2': 42, 'LeftHandRing3': 43, 'LeftHandRing4': 44, 'LeftHandRing4_Nub': 45, 'LeftHandMiddle1': 46, 'LeftHandMiddle2': 47, 'LeftHandMiddle3': 48, 'LeftHandMiddle4': 49, 'LeftHandMiddle4_Nub': 50, 'RightShoulder': 51, 'RightArm': 52, 'RightForeArm': 53, 'RightHand': 54, 'RightHandIndex': 55, 'RightHandThumb1': 56, 'RightHandThumb2': 57, 'RightHandThumb3': 58, 'RightHandThumb4': 59, 'RightHandThumb4_Nub': 60, 'RightHandIndex1': 61, 'RightHandIndex2': 62, 'RightHandIndex3': 63, 'RightHandIndex4': 64, 'RightHandIndex4_Nub': 65, 'RightHandRing': 66, 'RightHandPinky': 67, 'RightHandPinky1': 68, 'RightHandPinky2': 69, 'RightHandPinky3': 70, 'RightHandPinky4': 71, 'RightHandPinky4_Nub': 72, 'RightHandRing1': 73, 'RightHandRing2': 74, 'RightHandRing3': 75, 'RightHandRing4': 76, 'RightHandRing4_Nub': 77, 'RightHandMiddle1': 78, 'RightHandMiddle2': 79, 'RightHandMiddle3': 80, 'RightHandMiddle4': 81, 'RightHandMiddle4_Nub': 82, 'Neck': 83, 'Neck1': 84, 'Head': 85, 'HeadEnd': 86, 'HeadEnd_Nub': 87}
# target_joints = list(joint_name_to_idx.keys())      # BEAT ALL

# locomotion debug
# joint_name_to_idx = {'Hips': 0, 'LowerBack': 1, 'Spine': 2, 'Spine1': 3, 'RightShoulder': 4, 'RightArm': 5, 'RightForeArm': 6, 'RightHand': 7, 'RThumb': 8, 'RightFingerBase': 9, 'RightHandIndex1': 10, 'LeftShoulder': 11, 'LeftArm': 12, 'LeftForeArm': 13, 'LeftHand': 14, 'LThumb': 15, 'LeftFingerBase': 16, 'LeftHandIndex1': 17, 'Neck': 18, 'Neck1': 19, 'Head': 20, 'RHipJoint': 21, 'RightUpLeg': 22, 'RightLeg': 23, 'RightFoot': 24, 'RightToeBase': 25, 'LHipJoint': 26, 'LeftUpLeg': 27, 'LeftLeg': 28, 'LeftFoot': 29, 'LeftToeBase': 30}
# target_joints = list(joint_name_to_idx.keys())


def get_joint_tree(path, Nub=True, select=True):
    p = BVHParser()
    X = p.parse(path)

    joint_name_to_idx = {}
    index = 0
    for _, joint in enumerate(X.traverse()):
        if select:
            if joint not in target_joints: continue
        if not Nub:
            if '_Nub' in joint: continue
        joint_name_to_idx[joint] = index
        index += 1

    # if select and X.root_name not in joint_name_to_idx:
    #     joint_name_to_idx[X.root_name] = index
    #     index += 1

    # traverse tree
    joint_links = []
    stack = [X.root_name]
    while stack:
        joint = stack.pop()
        if not Nub:
            if '_Nub' in joint: continue
        parent = X.skeleton[joint]['parent']
        # tab = len(stack)
        # print('%s- %s (%s)'%('| '*tab, joint, parent))
        if parent:
            if select:
                if joint in target_joints and parent in target_joints:
                    joint_links.append((joint_name_to_idx[parent], joint_name_to_idx[joint]))
        for c in X.skeleton[joint]['children']:
            stack.append(c)

    print(joint_name_to_idx)
    print(joint_links)


def process_bvh(gesture_filename):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=60, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('jtsel', JointSelector(target_joints, include_root=True)),
        ('param', MocapParameterizer('position')),        # expmap, position
        ('cnst', ConstantsRemover(mode='position')),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    out_data = out_data[0]

    return out_data


def bvh_to_npy(bvh_path, sav_path):
    print(bvh_path)
    pos_data = process_bvh(bvh_path)
    # pdb.set_trace()
    # pos_data = np.pad(pos_data, ((0, 0), (3, 0)), 'constant', constant_values=(0, 0))
    print(pos_data.shape)
    npy_path = os.path.join(sav_path, bvh_path.split('/')[-1].replace('.bvh', '.npy'))
    np.save(npy_path, pos_data)


def process_bvh_fullbody(gesture_filename):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=60, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        ('jtsel', JointSelector(target_joints, include_root=False)),     # (225) -> 171
        ('pos', MocapParameterizer('position')),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)

    return out_data[0]


def bvh_to_npy_fullbody(bvh_path, sav_path):
    print(bvh_path)
    pos_data = process_bvh_fullbody(bvh_path)
    print(pos_data.shape)
    npy_path = os.path.join(sav_path, bvh_path.split('/')[-1].replace('.bvh', '.npy'))
    np.save(npy_path, pos_data)


def npy_to_bvh_upper(npy_path, sav_path, pipe_path, sav_path2):
    # if not os.path.exists(sav_path):
    #     os.mkdir(sav_path)
    x = np.load(npy_path)[:3600, 3:]        # 1min
    print(npy_path, x.shape)
    pipeline = jl.load(pipe_path)
    bvh_data = pipeline.inverse_transform([x])
    writer = BVHWriter()
    with open(sav_path, 'w') as f:
        writer.write(bvh_data[0], f)
    lines = []
    with open(sav_path2, 'w') as f2:
        with open(sav_path, 'r') as f:
            for j, line in enumerate(f.readlines()):
                if j <= 430:
                    if 'JOINT' in line:
                        if line.split(' ')[-1][:-1] in target_joints:
                            lines[-1] = (lines[-1].split(' ')[0] + ' 3 Xpositon Yposition Zposition\n')
                    lines.append(line)
        for line in lines:
            f2.write(line)
        for i in x:
            line_data = np.array2string(i, max_line_width=np.inf, separator=' ')
            f2.write(line_data[1:-2] + '\n')
    pdb.set_trace()


if __name__ == '__main__':
    '''
    cd process/
    python bvh_to_position.py
    '''
    # print joint tree information
    # bvh_file = "/mnt/nfs7/y50021900/My/beat-main/datasets/speakers/1/1_wayne_0_103_110.bvh"
    # get_joint_tree(bvh_file, Nub=True, select=False)


    # bvh_dir = "/mnt/nfs7/y50021900/My/tmp/TEST/bvh_BEAT2/"
    # save_dir = "/mnt/nfs7/y50021900/My/tmp/TEST/npy_position_BEAT_2/"
    #
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    #
    # # parse bvh
    # use_parallel_processing = False
    # files = sorted([f for f in glob.iglob(bvh_dir + '*.bvh')])
    # if use_parallel_processing:
    #     from joblib import Parallel, delayed
    #     Parallel(n_jobs=8)(delayed(bvh_to_npy)(bvh_path) for bvh_path in files)
    # else:
    #     for bvh_path in files:
    #         bvh_to_npy(bvh_path, save_dir)
            # bvh_to_npy_fullbody(bvh_path, save_dir)

    # bvh_to_npy('/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/11/generate11_generated.bvh', '/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/11/')

    source_bvh_path = '/mnt/nfs7/y50021900/My/data/BEAT0909/Motion/1_wayne_0_103_110.bvh'
    target_posititon_path = '../SIGGRAPH_2022/Dataset/'
    bvh_to_npy(source_bvh_path, target_posititon_path)

    # npy_path = "/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/10/generate10.npy"
    # pipe_path = os.path.join('./resource', 'data_pipe_60_position.sav')
    # sav_path = "/mnt/nfs7/y50021900/My/tmp/TEST/bvh_BEAT/generate10.bvh"
    # sav_path2 = "/mnt/nfs7/y50021900/My/tmp/TEST/bvh_BEAT/generate10_copy.bvh"
    # npy_to_bvh_upper(npy_path, sav_path, pipe_path, sav_path2)
