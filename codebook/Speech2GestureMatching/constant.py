import numpy as np

SR = 16000
WAV_TEST_SIZE = 409600

UPPERBODY_PARENT = np.array([1, 11, 1, 2, 3, 1, 5, 6, 10, 10, 10, 10, 1, 13, 13, 14, 15, 13, 17, 18, 13, 20, \
    21, 13, 23, 24, 13, 26, 27, 16, 19, 22, 25, 28, 34, 34, 35, 36, 34, 38, 39, 34, 41, 42, 34, 44, 45, 34, \
    47, 48, 37, 40, 43, 46, 49])

PARWISE_ORDER = [10, 11, 1, 12, 0, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(13, 61))

FILTER_SMOOTH_STD = 1.5

NUM_AUDIO_FEAT_FRAMES = 6       # 6 or 8
NUM_BODY_FEAT_FRAMES = 4

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
BODY_FEAT_IDX = [0, 8, 9, 12, 13]        # 'RightArm': 8, 'RightForeArm': 9, 'LeftArm': 12, 'LeftForeArm': 13, 'Spine3': 3?, 'Spine': 0

NUM_MFCC_FEAT=13        # notice that 14th always be zero nearly
NUM_AUDIO_FEAT=13*8      # 14*8
NUM_BODY_FEAT=144+36        # 3*8*4 -> 9*4*4
NUM_JOINTS=135      # 3*55 -> 9 * 15
STEP_SZ=4      # 8 -> 30

FRAME_INTERVAL = 4      # 2, 4, 8

BATCH_SIZE = 100
LR = 1e-4
MAX_ITERS = 300000      # 300000 -> 30
BURNIN_ITER = 10000     # 10000 -> 1

WEIGHT_GEN = 1
WEIGHT_RECON = 0.1
LAMBDA_GP = 100
GEN_HOP = 5

num_frames = 240
num_frames_code = 30
codebook_size = 512
