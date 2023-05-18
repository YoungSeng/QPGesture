import pdb

import numpy as np
import random
import os
import argparse
from data_processing import load_train_db, load_test_db, calc_data_stats
from control import create_control_filter
from utils import normalize_data, normalize_feat
from tqdm import tqdm
# from scipy import spatial
from sklearn.metrics.pairwise import paired_distances
from visualization import generate_seq_videos
from constant import UPPERBODY_PARENT, NUM_AUDIO_FEAT, NUM_BODY_FEAT, \
    NUM_MFCC_FEAT, NUM_JOINTS, STEP_SZ, WAV_TEST_SIZE, num_frames_code, num_frames, codebook_size, NUM_AUDIO_FEAT_FRAMES
import Levenshtein
import torch

seed_value= 123456
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)


parser = argparse.ArgumentParser()
parser.add_argument('-d','--train_database', help="Path to training database.", type=str, default="/path/to/training_db_data.npz")
parser.add_argument('-c','--train_codebook', help="Path to training database.", type=str, default="/path/to/training_db_data.npz")
parser.add_argument('-w','--train_wavlm', help="Path to training database.", type=str, default="/path/to/training_db_data.npz")
parser.add_argument('-wvq','--train_wavvq', help="Path to training database.", type=str, default="/path/to/training_db_data.npz")
parser.add_argument('-s','--codebook_signature', help="Path to training database.", type=str, default="/path/to/training_db_data.npz")
parser.add_argument('-e','--test_data', help="Path to test data.", type=str, default="/path/to/test_data.npz")
parser.add_argument('-tw','--test_wavlm', help="Path to training database.", type=str, default="/path/to/training_db_data.npz")
parser.add_argument('-twvq','--test_wavvq', help="Path to training database.", type=str, default="/path/to/training_db_data.npz")
parser.add_argument('-om','--out_knn_filename', help="Output filename of the k-NN searched motion.", type=str, default="/path/to/knn_pred.npz")
parser.add_argument('-ov','--out_video_path', help="Output path of the video.", type=str, default="/path/to/video/")
parser.add_argument('-k','--desired_k', help="The desired k-value for the k-NN (starts from 0).", type=int, default=0)
parser.add_argument('-f','--fake', type=bool, default=False)
parser.add_argument('-of','--out_fake_knn_filename', help="Output filename of the k-NN searched motion.", type=str, default="/path/to/knn_pred.npz")
parser.add_argument('--max_frames', type=int, default=0)

args = parser.parse_args()


def wavvq_distances(ls1, ls2, mode='sum'):
    if mode == 'sum':
        def ls2str(ls):
            ls = ls.reshape(NUM_AUDIO_FEAT_FRAMES, -1).transpose()  # (NUM_AUDIO_FEAT_FRAMES, groups=2)
            str1 = ''.join([chr(int(i)) for i in ls[0]])
            str2 = ''.join([chr(int(i)) for i in ls[1]])
            return str1, str2

        ls1_group1_str, ls1_group2_str = ls2str(ls1)
        ls2_group1_str, ls2_group2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_group1_str, ls2_group1_str) + Levenshtein.distance(ls1_group2_str, ls2_group2_str)

    elif mode == 'combine':
        def ls2str(ls):
                ls = ls.reshape(-1, 2).transpose()      # (NUM_AUDIO_FEAT_FRAMES * 2, groups=2)
                ls = ls[0] * 320 + ls[1]
                str = ''.join([chr(int(i)) for i in ls])
                return str

        ls1_str = ls2str(ls1)
        ls2_str = ls2str(ls2)

        return Levenshtein.distance(ls1_str, ls2_str)


class GestureKNN(object):
    def __init__(self, feat_train, motn_train, control_mask, n_aud_feat=112, n_body_feat=96, n_joints=165, step_sz=8):
        super(GestureKNN, self).__init__()
        
        # feat_train shape    : (num_seq, num_frames, (n_aud_feat + n_body_feat))
        # control_mask shape  : (num_seq, num_frames)
        # motn_train shape    : (num_seq, num_frames, n_joints)

        self.n_aud_feat = n_aud_feat
        self.n_body_feat = n_body_feat
        self.n_joints = n_joints
        self.step_sz = step_sz

        self.feat_train = feat_train
        self.motn_train = motn_train
        
        self.control_mask = control_mask
        self.n_db_seq = feat_train.shape[0]
        self.n_db_frm = feat_train.shape[1]


    def init_frame(self):
        init_seq = np.random.randint(0, self.n_db_seq)
        init_frm = np.random.randint(0, self.n_db_frm)

        while self.control_mask[init_seq, init_frm] != 1:
            init_seq = np.random.randint(0, self.n_db_seq)
            init_frm = np.random.randint(0, self.n_db_frm)

        return init_seq, init_frm


    def search_motion(self, feat_test, desired_k):
        # feat_test shape    : (self.n_aud_feat, num_frames)), (112, 64)

        n_frames = feat_test.shape[-1]      # 64
        feat_test = np.concatenate((feat_test[:, 0:1], feat_test), axis=1)      # (112, 1+64)
        pose_feat = np.zeros((self.n_body_feat, feat_test.shape[1]))        # (96， 1+64)
        feat_test = np.concatenate((feat_test, pose_feat), axis=0)      # (96+112, 1+64)

        # initialize pose feature
        init_seq, init_frm = self.init_frame()      # 21, 147
        feat_test[self.n_aud_feat:, 0] = self.feat_train[init_seq, init_frm, self.n_aud_feat:]      # (96, )
        pred_motion = np.zeros((self.n_joints, n_frames + 1))       # (165, 1+64)

        # start from j = 1 (j = 0 is just a duplicate of j = 1 so that we can initialize pose feature) 
        j = 1
        while j < n_frames:
            pos_dist_cands, pose_cands, feat_cands = self.search_pose_cands(feat_test[self.n_aud_feat:, j-1])
            # search_pose_cands( (96,) ) -> (num_seq,) (num_seq, 165, 8) (num_seq, 208, 8)
                
            n_retained = pos_dist_cands.shape[0]        # (num_seq, )

            # compute distance between audio pose feature and the pre-selected feature candidates
            audio_test_feat = feat_test[:self.n_aud_feat, j]        # (112, )
            
            aud_dist_cands = []
            for k in range(n_retained):
                # audio_sim_score = spatial.distance.cosine(audio_test_feat, feat_cands[k, :self.n_aud_feat, 0])
                # This library is not precise enough, if the input is the same two 1D matrices, the output is a number of order 1e-8 instead of 0, which will lead to incorrect sorting from smallest to largest.

                audio_sim_score = paired_distances([audio_test_feat], [feat_cands[k, :self.n_aud_feat, 0]], metric='cosine')[0]
                aud_dist_cands.append(audio_sim_score)

            # len(aud_dist_cands) = num_seq
            pos_score = np.array(pos_dist_cands).argsort().argsort()
            aud_score = np.array(aud_dist_cands).argsort().argsort()
            
            combined_score = pos_score + aud_score
            combined_sorted_idx = np.argsort(combined_score).tolist()       # len=num_seq

            feat_cands = feat_cands[combined_sorted_idx]        # (num_seq, 208, 8)
            pose_cands = pose_cands[combined_sorted_idx]        # (num_seq, 165, 8)
            
            feat_test[self.n_aud_feat:, j:(j+self.step_sz)] = feat_cands[desired_k, self.n_aud_feat:, :self.step_sz]        # (96, 8)
            pred_motion[:, j:(j+self.step_sz)] = pose_cands[desired_k, :, :self.step_sz]        # (165, 8)
            
            j += self.step_sz
        
        # pred_motion shape    : (self.n_joints, num_frames))
        return pred_motion[:, 1:]
        
        
    def search_pose_cands(self, body_test_feat):
        pos_dist_cands = []
        pose_cands = []
        feat_cands = []

        for k in range(self.feat_train.shape[0]):       # num_seq
            if self.control_mask[k].sum() == 0:
                continue

            body_dist_list = []
            body_train_feat = self.feat_train[k, :, self.n_aud_feat:]       # (num_seq, 64, 112+96) -> (64, 96)

            for l in range(body_train_feat.shape[0]):       # num_frames
                body_dist = np.linalg.norm(body_test_feat - body_train_feat[l])     # for every frame
                body_dist_list.append(body_dist)

            sorted_idx_list = np.argsort(body_dist_list)

            pose_cand_ctr = 0
            pose_cand_found = False

            while pose_cand_ctr < len(sorted_idx_list) - 1:     # for every frame
                f = sorted_idx_list[pose_cand_ctr]      # index
                d = body_dist_list[f]       # distance

                pose_cand_ctr += 1
                
                # skip the same sequence
                if d == 0.:
                    continue
                
                # skip frames with padded features
                if f > self.n_db_frm - self.step_sz:       # num_frames-8
                    continue
                
                # check if control condition is satisfied, self.control_mask: default ones like (num_seq, num_frames)
                if (self.control_mask[k, f] + self.control_mask[k, f + self.step_sz - 1]) != 2:
                    continue
                else:
                    pose_cand_found = True
                    break

            if pose_cand_found == False:
                continue
            
            # feat_cand shape: (num_feat_dim, step_sz)
            # pose_cand shape: (num_feat_dim, step_sz)
            feat_cand = self.feat_train[k, f:(f+self.step_sz), :].transpose()       # (8, 112+96).transpose()
            pose_cand = self.motn_train[k, f:(f+self.step_sz), :].transpose()       # (8, 165).transpose()

            pos_dist_cands.append(d)
            pose_cands.append(pose_cand)
            feat_cands.append(feat_cand)
        
        pos_dist_cands = np.array(pos_dist_cands)
        pose_cands = np.array(pose_cands)
        feat_cands = np.array(feat_cands)
        
        return pos_dist_cands, pose_cands, feat_cands


    def search_fake_motion(self, feat_test, desired_k):
        # feat_test shape    : (self.n_aud_feat, num_frames)), (112, 64)

        n_frames = feat_test.shape[-1]  # 64
        pose_feat = np.zeros((self.n_body_feat, feat_test.shape[1]))  # (96， 1+64)
        feat_test = np.concatenate((feat_test, pose_feat), axis=0)  # (96+112, 1+64)

        # initialize pose feature
        pred_motion = np.zeros((self.n_joints, n_frames))  # (165, 1+64)

        # start from j = 1 (j = 0 is just a duplicate of j = 1 so that we can initialize pose feature)
        j = 0
        while j < n_frames:
            pos_dist_cands, pose_cands = self.search_fake_pose_cands(feat_test[:self.n_aud_feat, j])     # 20221010

            pos_score = np.array(pos_dist_cands).argsort().argsort()

            combined_sorted_idx = np.argsort(pos_score).tolist()  # len=num_seq

            pose_cands = pose_cands[combined_sorted_idx]  # (num_seq, 165, 8)

            pred_motion[:, j:(j + self.step_sz)] = pose_cands[desired_k, :, :self.step_sz]  # (165, 8)

            j += self.step_sz

        # pred_motion shape    : (self.n_joints, num_frames))
        return pred_motion


    def search_fake_pose_cands(self, body_test_feat):
        pos_dist_cands = []
        pose_cands = []

        for k in range(self.feat_train.shape[0]):  # num_seq
            if self.control_mask[k].sum() == 0:
                continue

            body_dist_list = []
            body_train_feat = self.feat_train[k, :, :self.n_aud_feat]  # (num_seq, 64, 112+96) -> (64, 96)

            for l in range(body_train_feat.shape[0]):  # num_frames
                body_dist = paired_distances([body_test_feat], [body_train_feat[l]], metric='cosine')[0]
                body_dist_list.append(body_dist)

            sorted_idx_list = np.argsort(body_dist_list)

            pose_cand_ctr = 0
            pose_cand_found = False

            while pose_cand_ctr < len(sorted_idx_list) - 1:  # for every frame
                f = sorted_idx_list[pose_cand_ctr]  # index
                d = body_dist_list[f]  # distance

                pose_cand_ctr += 1

                # skip the same sequence
                if d == 0.:
                    continue

                # skip frames with padded features
                if f > self.n_db_frm - self.step_sz:  # num_frames-8
                    continue

                # check if control condition is satisfied, self.control_mask: default ones like (num_seq, num_frames)
                if (self.control_mask[k, f] + self.control_mask[k, f + self.step_sz - 1]) != 2:
                    continue
                else:
                    pose_cand_found = True
                    break

            if pose_cand_found == False:
                continue

            # pose_cand shape: (num_feat_dim, step_sz)
            pose_cand = self.motn_train[k, f:(f + self.step_sz), :].transpose()  # (8, 165).transpose()

            pos_dist_cands.append(d)
            pose_cands.append(pose_cand)

        pos_dist_cands = np.array(pos_dist_cands)
        pose_cands = np.array(pose_cands)

        return pos_dist_cands, pose_cands


def predict_gesture_from_audio(feat_train, pose_train, feat_test, control_mask, data_stats, \
                    k=0, n_aud_feat=112, n_body_feat=96, n_joints=165, step_sz=8, frames=0):
    # feat_train shape: (num_seq, num_feat=(NUM_AUDIO_FEAT+NUM_BODY_FEAT), num_frames)
    # pose_train shape: (num_seq, num_feat=NUM_JOINTS, num_frames)
    # feat_test shape: (num_seq, num_feat=NUM_AUDIO_FEAT, num_frames)
    # control_mask shape: (num_seq, num_frames)

    feat_mean = data_stats['feat_mean']
    feat_std = data_stats['feat_std']

    aud_mean_test = feat_mean[:, :n_aud_feat]
    aud_std_test = feat_std[:, :n_aud_feat]

    norm_feat_test = normalize_data(feat_test, aud_mean_test, aud_std_test)
    norm_feat_train = normalize_data(feat_train, feat_mean, feat_std)    
    norm_feat_train = norm_feat_train.transpose((0, 2, 1))
    pose_train = pose_train.transpose((0, 2, 1))

    n_test_seq = frames if frames != 0 else feat_test.shape[0]
    print('init knn...')
    gesture_knn = GestureKNN(feat_train=norm_feat_train, 
                            motn_train=pose_train, 
                            control_mask=control_mask, 
                            n_aud_feat=n_aud_feat, 
                            n_body_feat=n_body_feat, 
                            n_joints=n_joints,
                            step_sz=step_sz)

    motion_output = []

    print('begin search...')
    desired_k = np.random.choice(15, n_test_seq, p=[0.5, 0.5/14, 0.5/14, 0.5/14, 0.5/14, 0.5/14, 0.5/14, 0.5/14, 0.5/14,
                                                    0.5/14, 0.5/14, 0.5/14, 0.5/14, 0.5/14, 0.5/14])
    for i in tqdm(range(0, n_test_seq)):
        # pred_motion shape    : (NUM_JOINTS, num_frames))
        if args.fake:
            pred_motion = gesture_knn.search_fake_motion(feat_test=norm_feat_test[i], desired_k=desired_k[i])
        else:
            pred_motion = gesture_knn.search_motion(feat_test=norm_feat_test[i], desired_k=k)
        motion_output.append(pred_motion)

    # motion_output shape    : (num_seqs, num_feat=NUM_JOINTS, num_frames))
    return np.array(motion_output)


def main():
    # The training_db_data.npz file should contain the following variables:
    # "mfcc" with shape (num_seq, seq_len=64, num_feat=NUM_MFCC_FEAT)
    # "motion" with shape (num_seq, seq_len=64, num_joints=NUM_JOINTS)
    #
    # train_feats shape: (num_seq, num_feat=(NUM_AUDIO_FEAT+NUM_BODY_FEAT), num_frames=64)
    # train_motion shape: (num_seq, num_feat=NUM_JOINTS, num_frames=64)

    train_feats, _, train_motion = load_train_db(args.train_database)
    # train_feats = np.random.rand(128, 112+96, num_frames)
    # train_motion = np.random.rand(128, 165, num_frames)

    feat_mean, feat_std, motion_mean, motion_std = \
        calc_data_stats(
            train_feats.transpose((0, 2, 1)),
            train_motion.transpose((0, 2, 1)))

    data_stats = {
        'feat_mean' : feat_mean,
        'feat_std' : feat_std,
        'motion_mean' : motion_mean,
        'motion_std' : motion_std,
    }

    # The test_data.npz file should contain the following variables:
    # "mfcc" with shape (num_seq, seq_len=64, num_feat=NUM_MFCC_FEAT)
    # "motion" with shape (num_seq, seq_len=64, num_joints=NUM_JOINTS)
    # "wav" with shape (num_seq, num_feat=WAV_TEST_SIZE)
    #
    # feat_test shape: (num_seq, num_feat=NUM_AUDIO_FEAT, num_frames)
    # mfcc_test shape: (num_seq, num_feat=NUM_MFCC_FEAT, num_frames)
    # audio_test shape: (num_seq, num_feat=WAV_TEST_SIZE)

    feat_test, mfcc_test, audio_test = load_test_db(args.test_data)
    # feat_test = np.random.rand(2, 112, num_frames)
    # audio_test = np.random.rand(2, 409600)      # 96000HZ

    # control_mask shape: (num_seq, num_frames)
    control_mask = create_control_filter(train_feats.copy(), None)

    if args.fake:
        pred_seqs = predict_gesture_from_audio(
            feat_train=train_feats,
            pose_train=train_motion,
            feat_test=feat_test,
            control_mask=control_mask,
            data_stats=data_stats,
            k=args.desired_k,
            n_aud_feat=NUM_AUDIO_FEAT,
            n_body_feat=NUM_BODY_FEAT,
            n_joints=NUM_JOINTS,
            step_sz=STEP_SZ)
        print(pred_seqs.shape)
        np.savez_compressed(args.out_fake_knn_filename, knn_pred=pred_seqs.transpose((0, 2, 1)))  # (2, 165, 240)
    else:
        # pred_seqs shape: (num_seq, num_feats=NUM_JOINTS, num_frames)
        pred_seqs = predict_gesture_from_audio(
                        feat_train=train_feats,
                        pose_train=train_motion,
                        feat_test=feat_test,
                        control_mask=control_mask,
                        data_stats=data_stats,
                        k=args.desired_k,
                        n_aud_feat=NUM_AUDIO_FEAT,
                        n_body_feat=NUM_BODY_FEAT,
                        n_joints=NUM_JOINTS,
                        step_sz=STEP_SZ)      # for speaker 10,  for 0, 1 condition, 185 seqs takes 1h 58min, 15 seqs takes 9 min 52 s

        # save output motion as input to the ResyncGestureKNN module later
        print(pred_seqs.shape)
        np.savez_compressed(args.out_knn_filename, knn_pred=pred_seqs)      # (2, 165, 240)

        # pred_seqs = np.load(args.out_knn_filename)['knn_pred']
        # if not os.path.exists(args.out_video_path):
        #     os.makedirs(args.out_video_path)
        # generate_seq_videos(pred_seqs, audio_test, UPPERBODY_PARENT, args.out_video_path)


class CodeKNN(object):
    def __init__(self, mfcc_train, code_train, feat_train, wavlm_train, wavlm_train_feat, speech_features,
                 speech_features_feat, wavvq_train_feat, phase_train, context_train,
                 use_wavlm=False, use_wavvq=False, use_phase=False, use_txt=False):
        super(CodeKNN, self).__init__()

        # mfcc_train shape    : (num_seq, num_frames=240, NUM_MFCC_FEAT=13)
        # code_train shape    : (num_seq, num_frames_code=30)

        if use_wavlm:
            self.step_sz = wavlm_train.shape[1] // num_frames_code
            self.n_db_seq = wavlm_train.shape[0]
            self.n_db_frm = wavlm_train.shape[1]
        elif use_wavvq:
            self.step_sz = 398 / num_frames_code
            self.n_db_seq = wavvq_train_feat.shape[0]
            self.n_db_frm = 398
        else:
            self.step_sz = num_frames // num_frames_code  # 8
            self.n_db_seq = mfcc_train.shape[0]
            self.n_db_frm = mfcc_train.shape[1]
        print('step_sz is ', self.step_sz)
        self.mfcc_train = mfcc_train
        self.code_train = code_train
        self.feat_train = feat_train
        self.wavlm_train = wavlm_train
        self.wavlm_train_feat = wavlm_train_feat
        self.wavvq_train_feat = wavvq_train_feat
        self.phase_train = phase_train
        self.phase_channels = 8
        self.context_train = context_train
        # self.energy = energy
        # self.pitch = pitch
        # self.volume = volume
        self.speech_features = speech_features
        self.speech_features_feat = speech_features_feat
        self.code_to_signature()
        self.code_to_freq()
        self.use_phase = use_phase

    def init_code_phase(self):
            init_i = np.random.randint(0, self.n_db_seq)
            init_j = np.random.randint(0, self.n_db_frm - int(num_frames/num_frames_code))
            init_code = self.code_train[init_i, init_j//num_frames_code]
            if not self.use_phase:
                return init_code
            else:
                init_phase = self.phase_train[init_i, init_j:init_j + int(num_frames/num_frames_code)]
                phase = torch.tensor([j.detach().cpu().numpy() for j in init_phase[:, 0]]).squeeze().squeeze().numpy()  # 32, 8
                amps = torch.tensor([j.detach().cpu().numpy() for j in init_phase[:, 2]]).squeeze().squeeze().numpy()  # 32, 8
                phase_amp = np.concatenate((phase, amps), axis=1)
                return init_code, phase_amp

    def code_to_signature(self):
        x = np.load(args.codebook_signature)['signature']
        self.c2s = {}
        for i in range(codebook_size):
            self.c2s[i] = x[i]

    def code_to_freq(self):
        from collections import Counter
        train_code = np.load(args.train_codebook)['code']
        code = train_code.flatten()
        result = Counter(code)
        result_sorted = sorted(result.items(), key=lambda item: item[1], reverse=True)
        x = []
        y = []
        for d in result_sorted:
            x.append(d[0])
            y.append(d[1])
        y = 1 - np.array(y) / sum(y)
        self.c2f = {}
        for i in range(codebook_size):
            if i in x:
                self.c2f[i] = y[x.index(i)]
            else:
                self.c2f[i] = 1
        self.freq_dist_cands = list(self.c2f.values())

    def search_code_knn(self, clip_test, desired_k, use_feature=False, use_wavlm=False, use_freq=False, seed_code=None,
                        use_wavvq=False, use_phase=False, seed_phase=None, use_txt=False, clip_context=None, use_aud=False):
        # mfcc_test shape : (num_frames=3600, NUM_MFCC_FEAT=13)
        pose_cands = []
        result = []
        result_phase = []
        vote = []
        if use_phase:
            if seed_code != None:
                init_code = seed_code
                init_phase_amp = seed_phase
            else:
                init_code, init_phase_amp = self.init_code_phase()
            result_phase.append(init_phase_amp)
            print(init_code, init_phase_amp.shape)
        else:
            if seed_code != None:
                init_code = seed_code
            else:
                init_code = self.init_code_phase()
                print(init_code)
        result.append(init_code)

        for code in self.c2s.keys():
            pose_cands.append(code)

        i = 0
        while i < len(clip_test):
        # for i in range(0, len(clip_test), STEP_SZ * self.step_sz):
            print(str(i) + '\r', end='')
            pos_dist_cands = []
            for code in self.c2s.keys():
                if code == result[-1]:      # avoid still in the same code, optical
                    pos_dist_cands.append(1e10000)
                    continue
                pos_dist_cands.append(np.linalg.norm(self.c2s[result[-1]] - self.c2s[code]))
            # pose_cands shape: (codebook_size, )       pos_dist_cands shape: (codebook_size, )

            # len(aud_dist_cands) = codebook_size
            pos_score = np.array(pos_dist_cands).argsort().argsort()

            use_freq = True
            if use_freq:  # control signal
                freq_score = np.array(self.freq_dist_cands).argsort().argsort()
                pos_score = pos_score + freq_score * 0.05       # 0.1

            if use_txt:
                if use_wavlm:
                    clip_context_ = clip_context[int(i/self.wavlm_train.shape[1]*30)]
                elif use_wavvq:
                    clip_context_ = clip_context[int(i / 398 * 30)]
                txt_dist_cands, txt_index_cands, aux_ = self.search_text_cands(clip_context_)
                txt_score = np.array(txt_dist_cands).argsort().argsort()
                combined_score_ = pos_score + txt_score
                combined_sorted_idx_ = np.argsort(combined_score_).tolist()  # len=num_seq

            if use_aud:
                if use_wavvq and use_feature:
                    clip_wavvq = clip_test[int(i)]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_wavvq, mode='wavvq_feat')
                elif use_wavlm and not use_feature:
                    clip_wavlm = clip_test[i:i + self.step_sz]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_wavlm, mode='wavlm')
                elif use_wavlm and use_feature:
                    clip_wavlm_feat = clip_test[i]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_wavlm_feat, mode='wavlm_feat')
                elif not use_wavlm and use_feature:
                    clip_feat = clip_test[i]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_feat, mode='feat')
                elif not use_wavlm and not use_feature:
                    clip_mfcc = clip_test[i:i + self.step_sz]
                    aud_dist_cands, aud_index_cands, aux = self.search_audio_cands(clip_mfcc, mode='audio')

                aud_score = np.array(aud_dist_cands).argsort().argsort()
                combined_score = pos_score + aud_score
                combined_sorted_idx = np.argsort(combined_score).tolist()  # len=num_seq

            if not use_phase and use_txt and use_aud:
                combined_score = pos_score + aud_score + txt_score
                combined_sorted_idx = np.argsort(combined_score).tolist()
                if np.random.rand() > 0.5:
                    for ii in aud_index_cands[combined_sorted_idx[desired_k]]:
                        result.append(ii)
                else:
                    for ii in txt_index_cands[combined_sorted_idx[desired_k]]:
                        result.append(ii)
            elif not use_phase and use_aud:
                for ii in aud_index_cands[combined_sorted_idx[desired_k]]:
                    result.append(ii)
            elif not use_phase and use_txt:
                for ii in aud_index_cands[combined_sorted_idx_[desired_k]]:
                    result.append(ii)
            elif use_phase and use_aud and not use_txt:
                tmp_distance = []
                tmp_phase_amp = []
                for index in combined_sorted_idx[:2]:
                    candidates_index = aux[index]
                    candidates_phase = self.phase_train[candidates_index[0]][int(candidates_index[1]/398*240):int(candidates_index[1]/398*240) + 32]      # (32, 4, (1, 8, 1))
                    phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze().squeeze().numpy()        # 32, 8
                    amp = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze().squeeze().numpy()
                    phase_amp = np.concatenate((phase[:8], amp[:8]), axis=1)        # 32, 16
                    tmp_distance.append(paired_distances([np.concatenate((result_phase[-1][-5:], phase_amp[:3]), axis=0).reshape(-1)], [np.concatenate((result_phase[-1][-3:], phase_amp[:5]), axis=0).reshape(-1)], metric='cosine')[0])
                    tmp_phase_amp.append(np.concatenate((phase[-8:], amp[-8:]), axis=1))
                final_index = tmp_distance.index(min(tmp_distance))
                # print(final_index)
                for ii in aud_index_cands[combined_sorted_idx[final_index]]:
                    result.append(ii)
                result_phase.append(tmp_phase_amp[final_index])

            elif use_phase and not use_aud and use_txt:
                tmp_distance = []
                tmp_phase_amp = []
                for index in combined_sorted_idx_[:2]:
                    candidates_index = aux_[index]
                    candidates_phase = self.phase_train[candidates_index[0]][int(candidates_index[1]/398*240):int(candidates_index[1]/398*240) + 32]      # (32, 4, (1, 8, 1))
                    phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze().squeeze().numpy()        # 32, 8
                    amp = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze().squeeze().numpy()
                    phase_amp = np.concatenate((phase[:8], amp[:8]), axis=1)        # 32, 16
                    tmp_distance.append(paired_distances([np.concatenate((result_phase[-1][-5:], phase_amp[:3]), axis=0).reshape(-1)], [np.concatenate((result_phase[-1][-3:], phase_amp[:5]), axis=0).reshape(-1)], metric='cosine')[0])
                    tmp_phase_amp.append(np.concatenate((phase[-8:], amp[-8:]), axis=1))
                final_index = tmp_distance.index(min(tmp_distance))
                # print(final_index)
                for ii in txt_index_cands[combined_sorted_idx_[final_index]]:
                    result.append(ii)
                result_phase.append(tmp_phase_amp[final_index])

            elif use_phase and use_aud and use_txt:
                tmp_distance = []
                tmp_phase_amp = []
                for index in combined_sorted_idx[:1]:
                    candidates_index = aux[index]
                    candidates_phase = self.phase_train[candidates_index[0]][int(candidates_index[1]/398*240):int(candidates_index[1]/398*240) + 32]      # (32, 4, (1, 8, 1))
                    phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze().squeeze().numpy()        # 32, 8
                    amp = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze().squeeze().numpy()
                    phase_amp = np.concatenate((phase[:8], amp[:8]), axis=1)        # 32, 16
                    tmp_distance.append(paired_distances([np.concatenate((result_phase[-1][-5:], phase_amp[:3]), axis=0).reshape(-1)], [np.concatenate((result_phase[-1][-3:], phase_amp[:5]), axis=0).reshape(-1)], metric='cosine')[0])
                    tmp_phase_amp.append(np.concatenate((phase[-8:], amp[-8:]), axis=1))
                for index in combined_sorted_idx_[:1]:
                    candidates_index = aux_[index]
                    candidates_phase = self.phase_train[candidates_index[0]][int(candidates_index[1]/398*240):int(candidates_index[1]/398*240) + 32]      # (32, 4, (1, 8, 1))
                    phase = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 0]]).squeeze().squeeze().numpy()        # 32, 8
                    amp = torch.tensor([j.detach().cpu().numpy() for j in candidates_phase[:, 2]]).squeeze().squeeze().numpy()
                    phase_amp = np.concatenate((phase[:8], amp[:8]), axis=1)        # 32, 16
                    tmp_distance.append(paired_distances([np.concatenate((result_phase[-1][-5:], phase_amp[:3]), axis=0).reshape(-1)], [np.concatenate((result_phase[-1][-3:], phase_amp[:5]), axis=0).reshape(-1)], metric='cosine')[0])
                    tmp_phase_amp.append(np.concatenate((phase[-8:], amp[-8:]), axis=1))
                final_index = tmp_distance.index(min(tmp_distance))
                # print(final_index)
                if final_index in [0]:
                    for ii in aud_index_cands[combined_sorted_idx[final_index]]:
                        result.append(ii)
                elif final_index in [1]:
                    for ii in txt_index_cands[combined_sorted_idx_[final_index - 1]]:
                        result.append(ii)
                else:
                    raise ValueError("wrong final index")
                result_phase.append(tmp_phase_amp[final_index])
                vote.append(final_index)
                # print(final_index)
            i += STEP_SZ * self.step_sz

        if use_phase:
            return np.array(result)[1:1 + num_frames_code], np.array(result_phase)[1:], np.array(vote)
        else:
            return np.array(result)[1:1+num_frames_code], np.array(result_phase)[1:]

    def search_audio_cands(self, clip_input, mode='audio'):
        # mfcc_test:(num_frames=3600, NUM_MFCC_FEAT=13)
        aud_dist_cands = [1e+3] * codebook_size
        aud_index_cands = [[] for _ in range(codebook_size)]
        aux = [[] for _ in range(codebook_size)]
        for j in range(self.n_db_seq):
            k = 0
            while k < self.n_db_frm - STEP_SZ * self.step_sz:
            # for k in range(0, self.n_db_frm-STEP_SZ*self.step_sz, self.step_sz):
                code = self.code_train[j, int(k / self.step_sz)]
                if mode == 'wavvq_feat':
                    audio_sim_score = wavvq_distances(clip_input, self.wavvq_train_feat[j, int(k)], mode='combine')
                elif mode == 'audio':
                    audio_sim_score = paired_distances([clip_input.reshape(-1)], [self.mfcc_train[j, k:k+self.step_sz].reshape(-1)], metric='cosine')[0]
                elif mode == 'feat':
                    audio_sim_score = paired_distances([clip_input], [self.feat_train[j, k]], metric='cosine')[0]
                elif mode == 'wavlm':
                    audio_sim_score = paired_distances([clip_input.reshape(-1)], [self.wavlm_train[j, k:k + self.step_sz].reshape(-1)], metric='cosine')[0]
                elif mode == 'wavlm_feat':
                    audio_sim_score = paired_distances([clip_input], [self.wavlm_train_feat[j, k]], metric='cosine')[0]
                if audio_sim_score < aud_dist_cands[code]:
                    aud_dist_cands[code] = audio_sim_score
                    aud_index_cands[code] = self.code_train[j, int(k / self.step_sz):int(k / self.step_sz) + STEP_SZ]
                    aux[code] = [j, int(k)]
                k += self.step_sz
        return aud_dist_cands, aud_index_cands, aux

    def search_code_change(self, clip_test):
        '''
        energy  (3600,)
        pitch   (3600,)
        volume  (3600,)
        '''
        result = []
        # init_code = self.init_code()
        init_code = 34
        result.append(init_code)
        test_energy = clip_test[0]
        test_pitch = clip_test[1]
        test_volume = clip_test[2]
        pdb.set_trace()

    def search_text_cands(self, clip_input, mode='wavvq_feat'):
        txt_dist_cands = [1e+3] * codebook_size
        txt_index_cands = [[] for _ in range(codebook_size)]
        aux = [[] for _ in range(codebook_size)]
        for j in range(self.n_db_seq):
            for k in range(0, 240 - STEP_SZ * 8, 8):
                code = self.code_train[j, k // 8]
                if mode == 'wavvq_feat':
                    text_sim_score = paired_distances([clip_input], [self.context_train[j, k // 8]], metric='cosine')[0]
                if text_sim_score < txt_dist_cands[code]:
                    txt_dist_cands[code] = text_sim_score
                    txt_index_cands[code] = self.code_train[j, (k // 8):(k // 8) + STEP_SZ]
                    aux[code] = [j, k]
        return txt_dist_cands, txt_index_cands, aux


def predict_code_from_audio(train_mfcc, train_code, test_mfcc, data_stats, train_feat, test_feat, train_wavlm, test_wavlm,
                            train_wavlm_feat, test_wavlm_feat, speech_features, test_speech_features,
                            train_speech_features_feat, test_speech_features_feat, train_wavvq_feat, test_wavvq_feat,
                            train_phase, test_phase, train_context, test_context,
                            use_feature=False, use_wavlm=False, use_freq=False, use_speechfeat=False, use_wavvq=False,
                            use_phase=False, use_txt=False, use_aud=False, frames=0):
    norm_mfcc_train = normalize_data(train_mfcc, data_stats['mfcc_train_mean'], data_stats['mfcc_train_std'])
    norm_mfcc_train = norm_mfcc_train.transpose((0, 2, 1))
    norm_mfcc_test = normalize_data(test_mfcc, data_stats['mfcc_train_mean'], data_stats['mfcc_train_std'])
    norm_mfcc_test = norm_mfcc_test.transpose((0, 2, 1))

    norm_feat_train = normalize_data(train_feat, data_stats['feat_train_mean'], data_stats['feat_train_std'])
    norm_feat_train = norm_feat_train.transpose((0, 2, 1))
    norm_feat_test = normalize_data(test_feat, data_stats['feat_train_mean'], data_stats['feat_train_std'])
    norm_feat_test = norm_feat_test.transpose((0, 2, 1))

    n_test_seq = frames if frames != 0 else test_wavvq_feat.shape[0]      # test_mfcc.shape[0]

    train_wavlm = train_wavlm.transpose((0, 2, 1))
    test_wavlm = test_wavlm.transpose((0, 2, 1))
    train_wavlm_feat = train_wavlm_feat.transpose((0, 2, 1))
    test_wavlm_feat = test_wavlm_feat.transpose((0, 2, 1))

    # norm_energy, _, _ = normalize_feat(energy)
    # norm_pitch, _, _ = normalize_feat(pitch)
    # norm_volume, _, _ = normalize_feat(volume)
    #
    # norm_energy_test, _, _ = normalize_feat(test_energy)
    # norm_pitch_test, _, _ = normalize_feat(test_pitch)
    # norm_volume_test, _, _ = normalize_feat(test_volume)

    norm_speech_features = normalize_data(speech_features, data_stats['speech_features_train_mean'], data_stats['speech_features_train_std'])
    norm_speech_features = norm_speech_features.transpose((0, 2, 1))
    norm_test_speech_features = normalize_data(test_speech_features, data_stats['speech_features_train_mean'], data_stats['speech_features_train_std'])
    norm_test_speech_features = norm_test_speech_features.transpose((0, 2, 1))

    norm_speech_features_feat = normalize_data(train_speech_features_feat, data_stats['speech_features_feat_train_mean'], data_stats['speech_features_feat_train_std'])
    norm_speech_features_feat = norm_speech_features_feat.transpose((0, 2, 1))
    norm_test_speech_features_feat = normalize_data(test_speech_features_feat, data_stats['speech_features_feat_train_mean'], data_stats['speech_features_feat_train_std'])
    norm_test_speech_features = norm_test_speech_features_feat.transpose((0, 2, 1))

    train_wavvq_feat = train_wavvq_feat.transpose((0, 2, 1))
    test_wavvq_feat = test_wavvq_feat.transpose((0, 2, 1))

    train_phase = train_phase.transpose((0, 2, 1))
    # test_phase = test_phase.transpose((0, 2, 1))

    train_context = train_context.transpose((0, 2, 1))
    test_context = test_context.transpose((0, 2, 1))

    gesture_knn = CodeKNN(mfcc_train=norm_mfcc_train, code_train=train_code, feat_train=norm_feat_train,
                          wavlm_train=train_wavlm, wavlm_train_feat=train_wavlm_feat, speech_features=norm_speech_features,
                          speech_features_feat=norm_speech_features_feat, wavvq_train_feat=train_wavvq_feat,
                          phase_train=train_phase, context_train=train_context,
                          use_wavlm=use_wavlm, use_wavvq=use_wavvq, use_phase=use_phase, use_txt=use_txt)

    motion_output = []
    phase_output = []
    vote_output = []

    print('begin search...')
    for i in tqdm(range(0, n_test_seq)):
        # if use_speechfeat:
        #     pred_motion = gesture_knn.search_code_change(clip_test=[norm_energy_test[i], norm_pitch_test[i], norm_volume_test[i]])
        clip_context = test_context[i] if use_txt else None
        if use_wavvq and use_feature:
            if use_phase and use_aud:
                pred_motion, pred_phase, vote = gesture_knn.search_code_knn(clip_test=test_wavvq_feat[i], desired_k=args.desired_k, use_wavlm=False, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None, use_wavvq=True, use_phase=use_phase, seed_phase=phase_output[-1][-1] if i > 0 else None, use_txt=use_txt, clip_context=clip_context, use_aud=use_aud)
            elif not use_phase and use_aud:
                pred_motion, pred_phase = gesture_knn.search_code_knn(clip_test=test_wavvq_feat[i], desired_k=args.desired_k, use_wavlm=False, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None, use_wavvq=True, use_phase=use_phase, use_txt=use_txt, clip_context=clip_context, use_aud=use_aud)
            elif use_phase and not use_aud:
                pred_motion, pred_phase, vote = gesture_knn.search_code_knn(clip_test=test_wavvq_feat[i], desired_k=args.desired_k, use_wavlm=False, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None, use_wavvq=True, use_phase=use_phase, seed_phase=phase_output[-1][-1] if i > 0 else None, use_txt=use_txt, clip_context=clip_context, use_aud=use_aud)
        elif use_wavlm and not use_feature:
            pred_motion = gesture_knn.search_code_knn(clip_test=test_wavlm[i], desired_k=args.desired_k, use_wavlm=True, use_feature=False, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None)
        elif use_wavlm and use_feature:
            if use_phase and use_aud and use_txt:
                pred_motion, pred_phase, vote = gesture_knn.search_code_knn(clip_test=test_wavlm_feat[i], desired_k=args.desired_k, use_wavlm=True, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None, use_wavvq=False, use_phase=use_phase, seed_phase=phase_output[-1][-1] if i > 0 else None, use_txt=use_txt, clip_context=clip_context, use_aud=use_aud)
            else:
                pred_motion = gesture_knn.search_code_knn(clip_test=test_wavlm_feat[i], desired_k=args.desired_k, use_wavlm=True, use_feature=True, use_freq=use_freq, seed_code=motion_output[-1][-1] if i > 0 else None)
        elif not use_wavlm and use_feature:
            pred_motion = gesture_knn.search_code_knn(clip_test=norm_feat_test[i], desired_k=args.desired_k, use_wavlm=False, use_feature=True, use_freq=use_freq)
        elif not use_wavlm and not use_feature:
            pred_motion = gesture_knn.search_code_knn(clip_test=norm_mfcc_test[i], desired_k=args.desired_k, use_wavlm=False, use_feature=False, use_freq=use_freq)
        print(pred_motion)
        motion_output.append(pred_motion)
        phase_output.append(pred_phase)
        # vote_output.append(vote)
        # print(np.array(pred_phase).shape)
    # np.savez_compressed('vote.npz', vote=np.array(vote_output))
    return np.array(motion_output)


def main_codebook(maxFrames=0):
    from data_processing import load_db_codebook
    # (num_seq, NUM_MFCC_FEAT, num_frames=240), (num_seq, num_frames_code=30), (num_seq, NUM_MFCC_FEAT, num_frames=3600)
    train_mfcc, train_code, test_mfcc, train_feat, test_feat, train_wavlm, test_wavlm, train_wavlm_feat, \
    test_wavlm_feat, speech_features, test_speech_features, train_speech_features_feat, test_speech_features_feat, \
    train_wavvq_feat, test_wavvq_feat, train_phase, test_phase, train_context, test_context\
        = load_db_codebook(
        args.train_database, args.train_codebook, args.test_data, args.train_wavlm, args.test_wavlm, args.train_wavvq, args.test_wavvq)
    mfcc_train_mean, mfcc_train_std, _, _ = calc_data_stats(train_mfcc.transpose((0, 2, 1)), test_mfcc.transpose((0, 2, 1)))
    feat_train_mean, feat_train_std, _, _ = calc_data_stats(train_feat.transpose((0, 2, 1)), test_feat.transpose((0, 2, 1)))
    speech_features_train_mean, speech_features_train_std, _, _ = calc_data_stats(speech_features.transpose((0, 2, 1)), test_speech_features.transpose((0, 2, 1)))
    speech_features_feat_train_mean, speech_features_feat_train_std, _, _ = calc_data_stats(train_speech_features_feat.transpose((0, 2, 1)),test_speech_features_feat.transpose((0, 2, 1)))
    data_stats = {
        'mfcc_train_mean': mfcc_train_mean,
        'mfcc_train_std': mfcc_train_std,
        'feat_train_mean': feat_train_mean,
        'feat_train_std': feat_train_std,
        'speech_features_train_mean': speech_features_train_mean,
        'speech_features_train_std': speech_features_train_std,
        'speech_features_feat_train_mean': speech_features_feat_train_mean,
        'speech_features_feat_train_std': speech_features_feat_train_std
    }
    pred_seqs = predict_code_from_audio(train_mfcc, train_code, test_mfcc, data_stats, train_feat, test_feat, train_wavlm, test_wavlm,
                                        train_wavlm_feat, test_wavlm_feat, speech_features, test_speech_features,
                                        train_speech_features_feat, test_speech_features_feat, train_wavvq_feat, test_wavvq_feat,
                                        train_phase, test_phase, train_context, test_context,
                                        use_feature=True, use_wavlm=True, use_freq=False, use_speechfeat=False,
                                        use_wavvq=False, use_phase=True, use_txt=True, use_aud=True, frames=maxFrames)        # if use wavlm, frames should be 15, and test_data should be 240
    print(pred_seqs.shape)
    np.savez_compressed(args.out_knn_filename, knn_pred=pred_seqs)


if __name__ == "__main__":
    '''

240: 
python GestureKNN.py --train_database=../../data/BEAT0909/speaker_10_state_0/speaker_10_state_0_train_240.npz --test_data=../../data/BEAT0909/speaker_10_state_0/speaker_10_state_0_test_240.npz --out_knn_filename=./output/knn_pred.npz --out_video_path=./output/output_video_folder/
3600: 
python GestureKNN.py --train_database=../../data/BEAT0909/speaker_10_state_0/speaker_10_state_0_train_30.npz --test_data=../../data/BEAT0909/speaker_10_state_0/speaker_10_state_0_train_30.npz -f True --out_fake_knn_filename=./output/fake_knn_pred.npz --out_video_path=./output/output_video_folder/    
    '''
    # main()        # original KNN
    main_codebook(maxFrames=args.max_frames)
    # pred_seqs = np.load(args.out_knn_filename)['knn_pred']
    # prefix = '22'
    # generate_seq_videos(pred_seqs, prefix, gaussian_smooth=True, Savitzky_Golay_smooth=True, vis=False)
