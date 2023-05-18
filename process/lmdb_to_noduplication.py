import pdb
import os
import lmdb
import pyarrow
import numpy as np
import librosa


def lmdb_to_noduplication(clip_lmdb_dir, target_dir):
    src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
    src_txn = src_lmdb_env.begin(write=False)
    # sampling and normalization
    cursor = src_txn.cursor()
    for key, value in cursor:
        video = pyarrow.deserialize(value)
        vid = video['vid']
        clips = video['clips']
        for clip_idx, clip in enumerate(clips):
            print(vid)
            clip_skeleton = clip['poses']
            # clip_audio_raw = clip['audio_raw']
            np.savez_compressed(os.path.join(target_dir, vid + '.npz'), poses=clip_skeleton)
            break       # without mirror poses


def make_noduplication_data(root, target_path, train_poses_path, valid_poses_path, test_poses_path, all_mfcc_path, all_audio_path, n_frames=240, speaker_id=None):

    def process_data(path, n_frames):
        print('process data...')
        poses_data = []
        mfcc_data = []
        audio_data = []
        aux_data = []
        for item in os.listdir(path):
            if speaker_id is not None:
                if item.split('_')[0] != speaker_id:
                    continue
            name = item[:-4]
            print(name)
            poses_path = os.path.join(path, item)
            poses = np.load(poses_path)['poses']
            mfcc_path = os.path.join(all_mfcc_path, item)
            mfcc = np.load(mfcc_path)['mfcc']
            MINLEN = min(len(poses), len(mfcc))
            clip_len = n_frames * (MINLEN // n_frames)
            poses = poses[:clip_len].reshape(-1, n_frames,
                                             poses.shape[-1])  # (total_frames, 15*9) -> (len, n_frames, 15*9)
            mfcc = mfcc[:clip_len].reshape(-1, n_frames, mfcc.shape[-1])  # (total_frames, 14) -> (len, n_frames, 14)
            wav_path = os.path.join(all_audio_path, name + '.wav')
            wav, _ = librosa.load(wav_path, sr=16000)
            wav = wav[:int(16000 * clip_len / 60)].reshape(-1, 16000 * n_frames // 60)

            [poses_data.append(i) for i in poses]
            [mfcc_data.append(i) for i in mfcc]
            [audio_data.append(i) for i in wav]
            [aux_data.append(name) for _ in poses]
        return np.array(poses_data), np.array(mfcc_data), np.array(audio_data), np.array(aux_data)

    train_poses_data, train_mfcc_data, train_audio_data, train_aux_data = process_data(train_poses_path, n_frames)
    valid_poses_data, valid_mfcc_data, valid_audio_data, valid_aux_data = process_data(valid_poses_path, n_frames)
    test_poses_data, test_mfcc_data, test_audio_data, test_aux_data = process_data(test_poses_path, n_frames=60 * 60)

    np.savez_compressed(os.path.join(root, target_path, target_path + '_train' + '.npz'), body=train_poses_data, mfcc=train_mfcc_data, wav=train_audio_data, aux=train_aux_data)
    np.savez_compressed(os.path.join(root, target_path, target_path + '_valid' + '.npz'), body=valid_poses_data, mfcc=valid_mfcc_data, wav=valid_audio_data, aux=valid_aux_data)
    np.savez_compressed(os.path.join(root, target_path, target_path + '_test_60' + '.npz'), body=test_poses_data, mfcc=test_mfcc_data, wav=test_audio_data, aux=test_aux_data)


if __name__ == '__main__':
    '''
    cd process/
    python lmdb_to_noduplication.py
    '''
    root = "/mnt/nfs7/y50021900/My/data/BEAT0909"
    name = 'train'
    clip_lmdb_dir = "/mnt/nfs7/y50021900/My/data/BEAT0909/lmdb0919/lmdb_" + name
    target_path = "rotation_15_" + name
    if not os.path.exists(os.path.join(root, target_path)):
        os.mkdir(os.path.join(root, target_path))
    # lmdb_to_noduplication(clip_lmdb_dir, os.path.join(root, target_path))
    train_poses_path = os.path.join(root, target_path)

    name = 'valid'
    clip_lmdb_dir = "/mnt/nfs7/y50021900/My/data/BEAT0909/lmdb0919/lmdb_" + name
    target_path = "rotation_15_" + name
    if not os.path.exists(os.path.join(root, target_path)):
        os.mkdir(os.path.join(root, target_path))
    # lmdb_to_noduplication(clip_lmdb_dir, os.path.join(root, target_path))
    valid_poses_path = os.path.join(root, target_path)

    name = 'test'
    clip_lmdb_dir = "/mnt/nfs7/y50021900/My/data/BEAT0909/lmdb0919/lmdb_" + name
    target_path = "rotation_15_" + name
    if not os.path.exists(os.path.join(root, target_path)):
        os.mkdir(os.path.join(root, target_path))
    # lmdb_to_noduplication(clip_lmdb_dir, os.path.join(root, target_path))
    test_poses_path = os.path.join(root, target_path)

    mfcc_path = os.path.join(root, 'MFCC_60')
    audio_path = os.path.join(root, 'Audio_normalized')
    n_frames = 240
    name = '1'
    target_path = "rotation_15_" + name
    if not os.path.exists(os.path.join(root, target_path)):
        os.mkdir(os.path.join(root, target_path))
    make_noduplication_data(root, target_path, train_poses_path, valid_poses_path, test_poses_path, mfcc_path, audio_path, n_frames, speaker_id=name)
