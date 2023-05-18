import pdb

import numpy as np
import random
import os
import argparse
import torch

from torch.utils.data import TensorDataset
from tqdm import tqdm

from utils import normalize_data, inv_normalize_data
from train_resync_gestureknn import train_resync_model, load_resync_model
from data_processing import load_train_db, calc_data_stats, load_test_db, \
    prep_train_resync_data, convert_abswise_to_parwise
from visualization import generate_seq_videos
from constant import BATCH_SIZE, LR, MAX_ITERS, UPPERBODY_PARENT, PARWISE_ORDER, NUM_BODY_FEAT, NUM_MFCC_FEAT, NUM_JOINTS, WAV_TEST_SIZE, num_frames


seed_value = 0 
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)

torch.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_value)


parser = argparse.ArgumentParser()
parser.add_argument('-t','--train', help="Train the model.", action='store_true')
parser.add_argument('-d','--train_database', help="Path to the file that stores the training database.", type=str, default="/path/to/training_db_data.npz")
parser.add_argument('-e','--test_data', help="Path to test data.", type=str, default="/path/to/test_data.npz")
parser.add_argument('-s','--train_searched_motion', help="Path to file that stores k-NN reconstruction gesture sequences.", type=str, default="/path/to/training_searched_motion.npz")
parser.add_argument('-ok','--out_knn_filename_1', help="Path to the file that stores the gesture sequences composed by the k-NN.", type=str, default="/path/to/knn_pred.npz")
parser.add_argument('--out_knn_filename_2', help="Path to the file that stores the gesture sequences composed by the k-NN.", type=str, default="/path/to/knn_pred.npz")
parser.add_argument('-om','--out_resync_model', help="Output path of the trained model.", type=str, default="/path/to/resync_result/")
parser.add_argument('-ov','--out_video_path', help="Output path of the video.", type=str, default="/path/to/video/")
args = parser.parse_args()


@torch.no_grad()
def predict_resynced_gesture(mfcc_test, knn_pred, model_resync, motion_mean_parwise, motion_std_parwise, device, frames=0):
    # mfcc_test shape: (num_seq, num_feat=NUM_MFCC_FEAT, num_frames)
    # knn_pred shape: (num_seq, num_feats=NUM_JOINTS, num_frames)

    motion_mean_parwise = motion_mean_parwise.squeeze()
    motion_std_parwise = motion_std_parwise.squeeze()

    n_test_seq, _, n_frames = mfcc_test.shape
    motion_output = []

    n_test_seq = frames if frames != 0 else mfcc_test.shape[0]

    for i in tqdm(range(0, n_test_seq)):
        curr_knn = knn_pred[i:(i+1), :, :]

        # curr_knn = convert_abswise_to_parwise(curr_knn)
        curr_knn = curr_knn.squeeze().transpose()
        curr_knn = normalize_data(curr_knn, motion_mean_parwise, motion_std_parwise)

        curr_knn = np.expand_dims(curr_knn, axis=0)
        curr_knn = curr_knn.transpose((0, 2, 1))

        curr_mfcc = mfcc_test[i:i+1, :, :]

        resync_input = np.concatenate((curr_mfcc, curr_knn), axis=1)
        resync_input = torch.tensor(resync_input).to(device, dtype=torch.float)

        resync_output = model_resync(resync_input)
        resync_output = resync_output.cpu().detach().numpy().squeeze()

        pred_motion = resync_output.transpose()
        pred_motion = inv_normalize_data(pred_motion, motion_mean_parwise, motion_std_parwise)

        pred_motion = pred_motion.reshape((n_frames, -1, 3))

        # for j in range(len(UPPERBODY_PARENT)):
        #     pred_motion[:, PARWISE_ORDER[j]] += pred_motion[:, UPPERBODY_PARENT[PARWISE_ORDER[j]]]

        print("pred_motion.mean()", pred_motion.mean())

        pred_motion = pred_motion.reshape((n_frames, -1))
        motion_output.append(pred_motion.transpose((1, 0)))

    return np.array(motion_output)


def main():
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(args.out_resync_model):
        os.makedirs(args.out_resync_model)

    # train model
    if args.train:
        # The training_searched_motion.npz file should contain the following variables:
        # "mfcc" with shape (num_seq, seq_len=64, num_feat=NUM_MFCC_FEAT)
        # "motion" with shape (num_seq, seq_len=64, num_joints=NUM_JOINTS)
        # "motion_searched" with shape (num_seq, seq_len=64, num_joints=NUM_JOINTS)
        #
        # where "motion" stores the ground truth 3D gesture data and 
        # "motion_searched" stores the 're-interpreted' 3D gesture by running GestureKNN.
        #
        # x_knn_train shape: (num_seq, num_feat=(NUM_MFCC_FEAT+NUM_BODY_FEAT), num_frames=64)
        # x_real_train shape: (num_seq, num_feat=(NUM_MFCC_FEAT+NUM_BODY_FEAT), num_frames=64)


        x_knn_train, x_real_train = prep_train_resync_data(args.train_searched_motion)
        # x_knn_train = torch.tensor(np.random.rand(128, (NUM_MFCC_FEAT+NUM_JOINTS), num_frames))
        # x_real_train = torch.tensor(np.random.rand(128, (NUM_MFCC_FEAT+NUM_JOINTS), num_frames))
        print(x_knn_train.shape, x_real_train.shape)
        train_ds = TensorDataset(x_knn_train, x_real_train)
        
        train_resync_model(train_ds, BATCH_SIZE, MAX_ITERS, LR, args.out_resync_model, device)

    print('Finished training!')

    # The training_db_data.npz file should contain the following variables:
    # "mfcc" with shape (num_seq, seq_len=64, num_feat=NUM_MFCC_FEAT)
    # "motion" with shape (num_seq, seq_len=64, num_joints=NUM_JOINTS)
    #
    # train_db_feats shape: (num_seq, num_feat=(NUM_AUDIO_FEAT+NUM_BODY_FEAT), num_frames=64)
    # train_db_mfcc shape: (num_seq, num_feat=NUM_MFCC_FEAT, num_frames=64)
    # train_db_motion shape: (num_seq, num_feat=NUM_JOINTS, num_frames=64)
    train_db_feats, train_db_mfcc, train_db_motion = load_train_db(args.train_database)

    # train_db_mfcc = np.random.rand(128, NUM_MFCC_FEAT, num_frames)
    # train_db_motion = np.random.rand(128, NUM_JOINTS, num_frames)

    # train_db_motion_parwise = convert_abswise_to_parwise(train_db_motion)
    train_db_motion_parwise = train_db_motion

    mfcc_mean, mfcc_std, motion_parwise_mean, motion_parwise_std = \
        calc_data_stats(
            train_db_mfcc.transpose((0, 2, 1)),
            train_db_motion_parwise.transpose((0, 2, 1)))

    # The test_data.npz file should contain the following variables:
    # "mfcc" with shape (num_seq, seq_len=64, num_feat=NUM_MFCC_FEAT)
    # "motion" with shape (num_seq, seq_len=64, num_joints=NUM_JOINTS)
    # "wav" with shape (num_seq, num_feat=WAV_TEST_SIZE)
    #
    # feat_test shape: (num_seq, num_feat=NUM_AUDIO_FEAT, num_frames)
    # mfcc_test shape: (num_seq, num_feat=NUM_MFCC_FEAT, num_frames)
    # audio_test shape: (num_seq, num_feat=WAV_TEST_SIZE)
    _, mfcc_test, _ = load_test_db(args.test_data)

    # feat_test = np.random.rand(128, NUM_AUDIO_FEAT, num_frames)
    # mfcc_test = np.random.rand(2, NUM_MFCC_FEAT, num_frames)
    # audio_test = np.random.rand(2, WAV_TEST_SIZE)

    mfcc_test = normalize_data(mfcc_test, mfcc_mean, mfcc_std)

    # load trained model
    resync_mode_fn = os.path.join(args.out_resync_model, 'best_model.pth')

    # here we provide the k-NN searched motion from the test set 
    knn_pred_fn = args.out_knn_filename_1
    # knn_pred shape: (num_seq, num_feats=NUM_JOINTS, num_frames)
    knn_pred = np.load(knn_pred_fn)['knn_pred']

    # knn_pred = np.random.rand(2, NUM_JOINTS, num_frames)

    model_resync, _ = load_resync_model(resync_mode_fn, device)
    model_resync.eval()

    # pred_seqs shape: (num_seq, num_feats=NUM_JOINTS, num_frames)
    pred_seqs = predict_resynced_gesture(mfcc_test, knn_pred, model_resync, motion_parwise_mean, motion_parwise_std, device,
                                         frames=1)

    if not os.path.exists(args.out_video_path):
        os.makedirs(args.out_video_path)

    np.savez_compressed(args.out_knn_filename_2, knn_pred=pred_seqs)  # (2, 165, 240)

    # generate_seq_videos(pred_seqs, audio_test, UPPERBODY_PARENT, args.out_video_path, "resyncnet")


if __name__ == "__main__":
    '''
cd Speech2GestureMatching/
python ResyncGestureKNN.py --train_database=../../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240.npz --train_searched_motion=../../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240.npz --out_knn_filename_1=./output/knn_pred.npz --out_knn_filename_2=./output/knn_pred_stage2.npz --out_resync_model=./output/trained_model_folder/ --out_video_path=./output/output_video_folder/ --test_data=../../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_test_240.npz
 --train 
    '''
    # main()
    pred_seqs = np.load(args.out_knn_filename_1)['knn_pred']
    prefix = 'knn_pred_3600'
    generate_seq_videos(pred_seqs, prefix, gaussian_smooth=True, Savitzky_Golay_smooth=False, vis=True)
    pred_seqs = np.load(args.out_knn_filename_2)['knn_pred']
    prefix = 'knn_pred_stage2_3600'
    generate_seq_videos(pred_seqs, prefix, gaussian_smooth=True, Savitzky_Golay_smooth=False, vis=True)
