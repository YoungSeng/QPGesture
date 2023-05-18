import math
import pdb

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from utils import normalize_data
from constant import UPPERBODY_PARENT, NUM_AUDIO_FEAT_FRAMES, NUM_BODY_FEAT_FRAMES, BODY_FEAT_IDX, NUM_MFCC_FEAT, NUM_JOINTS, WAV_TEST_SIZE, FRAME_INTERVAL, num_frames, num_frames_code


def load_train_db(data_file):
    print('read training dataset...')
    data = np.load(data_file)

    # mfcc shape: (num_seq, num_frames=64, num_feat=NUM_MFCC_FEAT)
    # motion shape: (num_seq, num_frames=64, num_feat=NUM_JOINTS)

    mfcc = data['mfcc'][:, :, :NUM_MFCC_FEAT]
    motion = data['body']  # (934, 64, 165)

    n_b, n_t = motion.shape[0], motion.shape[1]  # num_seq, 64
    n_mfcc_feat = NUM_MFCC_FEAT

    # motion = motion.reshape((n_b, n_t, 9, -1))  # (934, 64, 9, 55)
    # motion = motion.transpose((0, 1, 3, 2))  # (934, 64, 55, 9)

    motion = motion.reshape((n_b, n_t, -1, 9))

    # Thumb, Index, Middle, Ring, Pinky (https://ts1.cn.mm.bing.net/th/id/R-C.c2b42cc4bc071eff96d9ad041349d1af?rik=HYG%2fn%2bgJX1Pxgg&riu=http%3a%2f%2fastrogurukul.com%2fweb%2fwp-content%2fuploads%2f2013%2f02%2ffnger-names.jpg&ehk=D%2biw5zn33M%2f8xUEeWTLsMXJ7%2b8siGDki%2fLYIBspncCI%3d&risl=&pid=ImgRaw&r=0&sres=1&sresct=1)
    # [3, 4, 6, 7, 14, 20, 35, 41] -> right_elbow, right_wrist, left_elbow, left_wrist, right_index_0, right_little_0, left_index_0, left_little_0

    slc_body_kpts = np.take(motion, BODY_FEAT_IDX, axis=2)  # (934, 64, 4, 9)
    slc_body_kpts = slc_body_kpts.reshape((n_b, n_t, -1))  # (934, 64, 36)

    body_feat = np.zeros((n_b, n_t, NUM_BODY_FEAT_FRAMES, 9 * len(BODY_FEAT_IDX)))  # (934, 64, 4, 36)

    for i in range(NUM_BODY_FEAT_FRAMES):  # 4
        post_pad = np.zeros((n_b, i * FRAME_INTERVAL, 9 * len(BODY_FEAT_IDX)))  # (934, i * 2, 3*8)
        body_feat[:, :, i, :] = np.concatenate((slc_body_kpts[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
        # [(934, 64, 3*8), (934, 0, 3*8)] [(934, 62, 3*8), (934, 2, 3*8)]

    body_feat = body_feat.reshape((n_b, n_t, -1))
    audio_feat = np.zeros((n_b, n_t, NUM_AUDIO_FEAT_FRAMES, n_mfcc_feat))

    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((n_b, i * FRAME_INTERVAL, n_mfcc_feat))
        audio_feat[:, :, i, :] = np.concatenate((mfcc[:, (i * FRAME_INTERVAL):, :n_mfcc_feat], post_pad), axis=1)

    audio_feat = audio_feat.reshape((n_b, n_t, -1))

    features = np.concatenate((audio_feat, body_feat), axis=2)

    motion = motion.reshape((n_b, n_t, -1))

    features = features.transpose((0, 2, 1))
    mfcc = mfcc.transpose((0, 2, 1))
    motion = motion.transpose((0, 2, 1))

    # features shape: (num_seq, num_feat=(NUM_AUDIO_FEAT+NUM_BODY_FEAT), num_frames=64)
    # mfcc shape: (num_seq, num_feat=NUM_MFCC_FEAT, num_frames=64)
    # motion shape: (num_seq, num_feat=NUM_JOINTS, num_frames=64)

    return features.astype(np.float32), mfcc.astype(np.float32), motion.astype(np.float32)


def load_test_db(data_file):
    # mfcc shape: (num_seq, num_frames=64, num_feat=NUM_MFCC_FEAT)
    # audio shape: (num_seq, num_feat=WAV_TEST_SIZE)
    print('read testing dataset...')
    data = np.load(data_file)
    mfcc = data['mfcc'][:, :, :NUM_MFCC_FEAT]
    # audio = data['wav']

    n_b, n_t = mfcc.shape[0], mfcc.shape[1]
    n_mfcc_feat = NUM_MFCC_FEAT

    audio_feat = np.zeros((n_b, n_t, NUM_AUDIO_FEAT_FRAMES, n_mfcc_feat))

    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((n_b, i * FRAME_INTERVAL, n_mfcc_feat))
        audio_feat[:, :, i, :] = np.concatenate((mfcc[:, (i * FRAME_INTERVAL):, :n_mfcc_feat], post_pad), axis=1)

    audio_feat = audio_feat.reshape((n_b, n_t, -1))
    features = audio_feat.transpose((0, 2, 1))

    # mfcc = mfcc.reshape(-1, num_frames, mfcc.shape[-1])     # 20221012

    mfcc = mfcc.transpose((0, 2, 1))

    # features shape: (num_seq, num_feat=(NUM_AUDIO_FEAT), num_frames=64)
    # mfcc shape: (num_seq, num_feat=NUM_MFCC_FEAT, num_frames=64)
    # audio shape: (num_seq, num_feat=WAV_TEST_SIZE)

    return features.astype(np.float32), mfcc.astype(np.float32), None


def prep_train_resync_data(train_path):
    print('read training resync dataset...')
    training_data = np.load(train_path)     # 'body', 'face', 'wav', 'imgs'

    # mfcc_train shape          : (num_seq, num_frame=64, num_feat=NUM_MFCC_FEAT)
    # gesture_knn_train shape   : (num_seq, num_frame=64, num_feat=NUM_JOINTS)
    # gesture_real_train shape  : (num_seq, num_frame=64, num_feat=NUM_JOINTS)

    mfcc_train, gesture_real_train = training_data['mfcc'][:, :, :NUM_MFCC_FEAT], training_data['body']        # ?, ?, (4748, 64, 165)

    gesture_knn_train = gesture_real_train.copy().reshape(-1, gesture_real_train.shape[-1])
    np.random.shuffle(gesture_knn_train)
    gesture_knn_train = gesture_knn_train.reshape(gesture_real_train.shape[0], gesture_real_train.shape[1], gesture_real_train.shape[2])

    # mfcc_train = np.random.rand(2, 112 + 96, 14)
    # gesture_knn_train = np.random.rand(2, 112 + 96, 165)
    # gesture_real_train = training_data['body']

    # mfcc_train shape          : (num_seq, num_feat=NUM_MFCC_FEAT, num_frame=64)
    # gesture_knn_train shape   : (num_seq, num_feat=NUM_JOINTS, num_frame=64)
    # gesture_real_train shape  : (num_seq, num_feat=NUM_JOINTS, num_frame=64)

    mfcc_train = mfcc_train.transpose((0, 2, 1))
    gesture_knn_train = gesture_knn_train.transpose((0, 2, 1))
    gesture_real_train = gesture_real_train.transpose((0, 2, 1))

    # gesture_knn_train shape: (num_seq, num_feat, num_frame)
    # gesture_real_train shape: (num_seq, num_feat, num_frame)

    # mfcc_mean shape: (1, num_feat=NUM_MFCC_FEAT, 1)
    # mfcc_std shape: (1, num_feat=NUM_MFCC_FEAT, 1)

    # gesture_knn_mean shape: (1, num_feat=NUM_JOINTS, 1)
    # gesture_knn_std shape: (1, num_feat=NUM_JOINTS, 1)

    mfcc_mean, mfcc_std, gesture_knn_mean, gesture_knn_std = calc_data_stats(
        mfcc_train.transpose((0, 2, 1)), 
        gesture_knn_train.transpose((0, 2, 1))
    )

    # gesture_real_mean shape: (1, num_feat=NUM_JOINTS, 1)
    # gesture_real_std shape: (1, num_feat=NUM_JOINTS, 1)

    _, _, gesture_real_mean, gesture_real_std = calc_data_stats(
        mfcc_train.transpose((0, 2, 1)), 
        gesture_real_train.transpose((0, 2, 1))
    )

    # shuffle the sequences
    train_len = mfcc_train.shape[0]
    rand_idx = np.arange(train_len)
    np.random.shuffle(rand_idx)
    
    mfcc_train = mfcc_train[rand_idx]
    gesture_knn_train = gesture_knn_train[rand_idx]
    gesture_real_train = gesture_real_train[rand_idx]

    # normalize
    mfcc_train = normalize_data(mfcc_train, mfcc_mean, mfcc_std)

    motion_knn_train = normalize_data(gesture_knn_train, gesture_knn_mean, gesture_knn_std)
    x_knn_train = np.concatenate((mfcc_train, motion_knn_train), axis=1)
    motion_real_train = normalize_data(gesture_real_train, gesture_real_mean, gesture_real_std)
    x_real_train = np.concatenate((mfcc_train, motion_real_train), axis=1)

    x_knn_train = x_knn_train[0::FRAME_INTERVAL]
    x_real_train = x_real_train[0::FRAME_INTERVAL]

    # x_knn_train shape: (num_seq, num_feat=(NUM_MFCC_FEAT+NUM_BODY_FEAT), num_frames=64)
    # x_real_train shape: (num_seq, num_feat=(NUM_MFCC_FEAT+NUM_BODY_FEAT), num_frames=64)

    return torch.tensor(x_knn_train), torch.tensor(x_real_train)


def calc_data_stats(feat1, feat2=None):
    feat1_mean = np.expand_dims(feat1.mean(axis=(1, 0)), axis=(0, -1))
    feat1_std = np.expand_dims(feat1.std(axis=(1, 0)), axis=(0, -1))

    if feat2 is None:
        return feat1_mean, feat1_std

    else:
        feat2_mean = np.expand_dims(feat2.mean(axis=(1, 0)), axis=(0, -1))
        feat2_std = np.expand_dims(feat2.std(axis=(1, 0)), axis=(0, -1))
        return feat1_mean, feat1_std, feat2_mean, feat2_std


def convert_abswise_to_parwise(motion):     # (num_seq, num_feat=NUM_JOINTS, num_frames=64)
    motion = motion.transpose((0, 2, 1))        # -> (num_seq, 64, NUM_JOINTS)
    N, T, _ = motion.shape

    motion = motion.reshape((N, T, -1, 3))      # -> (num_seq, 64, NUM_JOINTS//3, 3)
    motion = motion - motion[:, :, UPPERBODY_PARENT]

    motion = motion.reshape((N, T, -1))     # -> (num_seq, 64, NUM_JOINTS)

    return motion.transpose((0, 2, 1))      # -> (num_seq, NUM_JOINTS, 64)


def load_db_codebook(data_file, codepath, test_data_path, train_wavlm, test_wavlm, train_wavvq, test_wavvq):
    import torch.nn.functional as F
    print('read training dataset...')
    data = np.load(data_file)
    mfcc = data['mfcc'][:, :, :NUM_MFCC_FEAT]
    energy = np.expand_dims(data['energy'], axis=-1)        # (n, 240, 1)
    pitch = np.expand_dims(data['pitch'], axis=-1)
    volume = np.expand_dims(data['volume'], axis=-1)
    speech_features = np.concatenate((energy, pitch, volume), axis=2)
    code = np.load(codepath)['code']

    audio_feat = np.zeros((mfcc.shape[0], mfcc.shape[1], NUM_AUDIO_FEAT_FRAMES, NUM_MFCC_FEAT))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((mfcc.shape[0], i * FRAME_INTERVAL, NUM_MFCC_FEAT))
        audio_feat[:, :, i, :] = np.concatenate((mfcc[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
    train_feat = audio_feat.reshape((mfcc.shape[0], mfcc.shape[1], -1))

    speech_features_feat = np.zeros((speech_features.shape[0], speech_features.shape[1], NUM_AUDIO_FEAT_FRAMES, 3))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((speech_features.shape[0], i * FRAME_INTERVAL, 3))
        speech_features_feat[:, :, i, :] = np.concatenate((speech_features[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
    train_speech_features_feat = speech_features_feat.reshape((speech_features.shape[0], speech_features.shape[1], -1))

    print('mfcc shape', mfcc.shape, 'code shape', code.shape, 'train_feat shape', train_feat.shape,
          'speech_features shape', speech_features.shape, 'train speech features feat shape', train_speech_features_feat.shape)

    print('read testing dataset...')
    test_data = np.load(test_data_path)
    test_mfcc = test_data['mfcc'][:, :, :NUM_MFCC_FEAT]
    test_energy = np.expand_dims(test_data['energy'], axis=-1)
    test_pitch = np.expand_dims(test_data['pitch'], axis=-1)
    test_volume = np.expand_dims(test_data['volume'], axis=-1)
    test_speech_features = np.concatenate((test_energy, test_pitch, test_volume), axis=2)

    test_audio_feat = np.zeros((test_mfcc.shape[0], test_mfcc.shape[1], NUM_AUDIO_FEAT_FRAMES, NUM_MFCC_FEAT))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((test_mfcc.shape[0], i * FRAME_INTERVAL, NUM_MFCC_FEAT))
        test_audio_feat[:, :, i, :] = np.concatenate((test_mfcc[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
    test_feat = test_audio_feat.reshape((test_mfcc.shape[0], test_mfcc.shape[1], -1))

    speech_features_feat = np.zeros((test_speech_features.shape[0], test_speech_features.shape[1], NUM_AUDIO_FEAT_FRAMES, 3))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((test_speech_features.shape[0], i * FRAME_INTERVAL, 3))
        speech_features_feat[:, :, i, :] = np.concatenate((test_speech_features[:, (i * FRAME_INTERVAL):], post_pad), axis=1)
    test_speech_features_feat = speech_features_feat.reshape((test_speech_features.shape[0], test_speech_features.shape[1], -1))

    print('test_mfcc shape', test_mfcc.shape, 'test_feat shape', test_feat.shape,
          'test_speech_features shape', test_speech_features.shape,
          'test speech features feat shape', test_speech_features_feat.shape)

    # debug
    # train_wavlm = np.zeros((2, 2, 2))
    # test_wavlm = np.zeros((2, 2, 2))
    # train_wavlm_feat = np.zeros((2, 2, 2))
    # test_wavlm_feat = np.zeros((2, 2, 2))
    # train_wavlm_interpolate = np.zeros((2, 2, 2))
    # test_wavlm_interpolate = np.zeros((2, 2, 2))

    train_wavlm = np.load(train_wavlm)['wavlm']
    test_wavlm = np.load(test_wavlm)['wavlm']

    nums_wavlm_frames = train_wavlm.shape[1]
    new_wavlm_frames = nums_wavlm_frames // code.shape[-1] * code.shape[-1]
    train_wavlm_interpolate = F.interpolate(torch.from_numpy(train_wavlm).transpose(1, 2), size=new_wavlm_frames, align_corners=True, mode='linear').transpose(1, 2).numpy()
    test_wavlm_interpolate = F.interpolate(torch.from_numpy(test_wavlm).transpose(1, 2), size=new_wavlm_frames, align_corners=True, mode='linear').transpose(1, 2).numpy()
    print(train_wavlm.shape, test_wavlm.shape, train_wavlm_interpolate.shape, test_wavlm_interpolate.shape)

    train_wavlm_feat = np.zeros((train_wavlm_interpolate.shape[0], train_wavlm_interpolate.shape[1], NUM_AUDIO_FEAT_FRAMES, train_wavlm_interpolate.shape[-1]))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((train_wavlm_interpolate.shape[0], i * (FRAME_INTERVAL-2), train_wavlm_interpolate.shape[-1]))
        train_wavlm_feat[:, :, i, :] = np.concatenate((train_wavlm_interpolate[:, (i * (FRAME_INTERVAL-2)):], post_pad), axis=1)
    train_wavlm_feat = train_wavlm_feat.reshape((train_wavlm_interpolate.shape[0], train_wavlm_interpolate.shape[1], -1))

    test_wavlm_feat = np.zeros((test_wavlm_interpolate.shape[0], test_wavlm_interpolate.shape[1], NUM_AUDIO_FEAT_FRAMES, test_wavlm_interpolate.shape[-1]))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((test_wavlm_interpolate.shape[0], i * (FRAME_INTERVAL-2), test_wavlm_interpolate.shape[-1]))
        test_wavlm_feat[:, :, i, :] = np.concatenate((test_wavlm_interpolate[:, (i * (FRAME_INTERVAL-2)):], post_pad), axis=1)
    test_wavlm_feat = test_wavlm_feat.reshape((test_wavlm_interpolate.shape[0], test_wavlm_interpolate.shape[1], -1))

    print('train_wavlm_feat shape', train_wavlm_feat.shape, 'test_wavlm_feat shape', test_wavlm_feat.shape)

    train_wavvq = np.load(train_wavvq)['wavvq']
    test_wavvq = np.load(test_wavvq)['wavvq']

    FRAME_INTERVAL_vq = train_wavvq.shape[1] / num_frames_code

    '''
    train_wavvq_feat = np.zeros((train_wavvq.shape[0], train_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, train_wavvq.shape[-1]))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((train_wavvq.shape[0], int(i * FRAME_INTERVAL_vq), train_wavvq.shape[-1]))
        train_wavvq_feat[:, :, i, :] = np.concatenate((train_wavvq[:, int(i * FRAME_INTERVAL_vq):], post_pad), axis=1)
    train_wavvq_feat = train_wavvq_feat.reshape((train_wavvq.shape[0], train_wavvq.shape[1], -1))

    test_wavvq_feat = np.zeros((test_wavvq.shape[0], test_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, test_wavvq.shape[-1]))
    for i in range(NUM_AUDIO_FEAT_FRAMES):
        post_pad = np.zeros((test_wavvq.shape[0], int(i * FRAME_INTERVAL_vq), test_wavvq.shape[-1]))
        test_wavvq_feat[:, :, i, :] = np.concatenate((test_wavvq[:, int(i * FRAME_INTERVAL_vq):], post_pad), axis=1)
    test_wavvq_feat = test_wavvq_feat.reshape((test_wavvq.shape[0], test_wavvq.shape[1], -1))
    '''
    # 20221101
    train_wavvq_feat = np.zeros((train_wavvq.shape[0], train_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, train_wavvq.shape[-1]))
    for i in range(0, NUM_AUDIO_FEAT_FRAMES):
        pre_pad_len = int((NUM_AUDIO_FEAT_FRAMES - i - 1) * FRAME_INTERVAL_vq)
        post_pad_len = int(i * FRAME_INTERVAL_vq)
        pre_pad = np.zeros((train_wavvq.shape[0], pre_pad_len, train_wavvq.shape[-1]))
        post_pad = np.zeros((train_wavvq.shape[0], post_pad_len, train_wavvq.shape[-1]))
        train_wavvq_feat[:, :, i, :] = np.concatenate((pre_pad, train_wavvq[:, :(train_wavvq.shape[1]-pre_pad_len)]), axis=1)

    train_wavvq_feat1 = train_wavvq_feat.reshape((train_wavvq.shape[0], train_wavvq.shape[1], -1))

    train_wavvq_feat = np.zeros((train_wavvq.shape[0], train_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, train_wavvq.shape[-1]))
    for i in range(0, NUM_AUDIO_FEAT_FRAMES):
        pre_pad_len = int((NUM_AUDIO_FEAT_FRAMES - i - 1) * FRAME_INTERVAL_vq)
        post_pad_len = int(i * FRAME_INTERVAL_vq)
        pre_pad = np.zeros((train_wavvq.shape[0], pre_pad_len, train_wavvq.shape[-1]))
        post_pad = np.zeros((train_wavvq.shape[0], post_pad_len, train_wavvq.shape[-1]))
        train_wavvq_feat[:, :, i, :] = np.concatenate((train_wavvq[:, post_pad_len:], post_pad), axis=1)
    train_wavvq_feat = np.delete(train_wavvq_feat, 0, axis=2)
    train_wavvq_feat2 = train_wavvq_feat.reshape((train_wavvq.shape[0], train_wavvq.shape[1], -1))

    train_wavvq_feat = np.concatenate((train_wavvq_feat1, train_wavvq_feat2), axis=-1)

    test_wavvq_feat = np.zeros((test_wavvq.shape[0], test_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, test_wavvq.shape[-1]))
    for i in range(0, NUM_AUDIO_FEAT_FRAMES):
        pre_pad_len = int((NUM_AUDIO_FEAT_FRAMES - i - 1) * FRAME_INTERVAL_vq)
        pre_pad = np.zeros((test_wavvq.shape[0], pre_pad_len, test_wavvq.shape[-1]))
        test_wavvq_feat[:, :, i, :] = np.concatenate((pre_pad, test_wavvq[:, :(test_wavvq.shape[1]-pre_pad_len)]), axis=1)

    test_wavvq_feat1 = test_wavvq_feat.reshape((test_wavvq.shape[0], test_wavvq.shape[1], -1))

    test_wavvq_feat = np.zeros((test_wavvq.shape[0], test_wavvq.shape[1], NUM_AUDIO_FEAT_FRAMES, test_wavvq.shape[-1]))
    for i in range(0, NUM_AUDIO_FEAT_FRAMES):
        post_pad_len = int(i * FRAME_INTERVAL_vq)
        post_pad = np.zeros((test_wavvq.shape[0], post_pad_len, test_wavvq.shape[-1]))
        test_wavvq_feat[:, :, i, :] = np.concatenate((test_wavvq[:, post_pad_len:], post_pad), axis=1)
    test_wavvq_feat = np.delete(test_wavvq_feat, 0, axis=2)
    test_wavvq_feat2 = test_wavvq_feat.reshape((test_wavvq.shape[0], test_wavvq.shape[1], -1))

    test_wavvq_feat = np.concatenate((test_wavvq_feat1, test_wavvq_feat2), axis=-1)

    print('train_wavvq_feat shape', train_wavvq_feat.shape, 'test_wavvq_feat shape', test_wavvq_feat.shape)

    train_phase = np.load(data_file, allow_pickle=True)['phase']       # n, len, 4 (1 * 8 * 1)
    test_phase = np.load(test_data_path, allow_pickle=True)['phase']

    train_context = np.load(data_file)['context'].squeeze(2)  # n, len, 384
    test_context = np.load(test_data_path)['context'].squeeze(2)  # n, len, 384

    return mfcc.transpose((0, 2, 1)), code, test_mfcc.transpose((0, 2, 1)), \
           train_feat.transpose((0, 2, 1)), test_feat.transpose((0, 2, 1)), \
           train_wavlm_interpolate.transpose((0, 2, 1)), test_wavlm_interpolate.transpose((0, 2, 1)), \
           train_wavlm_feat.transpose((0, 2, 1)), test_wavlm_feat.transpose((0, 2, 1)), \
           speech_features.transpose((0, 2, 1)), test_speech_features.transpose((0, 2, 1)),\
           train_speech_features_feat.transpose((0, 2, 1)), test_speech_features_feat.transpose((0, 2, 1)),\
           train_wavvq_feat.transpose((0, 2, 1)), test_wavvq_feat.transpose((0, 2, 1)), \
           train_phase.transpose((0, 2, 1)), test_phase.transpose((0, 2, 1)), \
           train_context.transpose((0, 2, 1)), test_context.transpose((0, 2, 1))


# def interpolate_data(matrix, size):
#     from scipy import interpolate
#
#     x = []
#     y = []
#     for i in range(1, 1 + len(matrix)):
#         x.append(i)
#         y.append(matrix[i])
#
#     f = interpolate.interp1d(x, y, kind='linear')
#
#     a = [i for i in range(1, 1 + size)]
#
#     a = a / (size/len(matrix))
#
#     return f(a)

