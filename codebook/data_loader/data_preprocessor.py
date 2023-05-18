""" create data samples """
import lmdb
import math
import numpy as np
import pyarrow
import pdb


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, file='ag',
                 select='specific', n_codes=30):
        self.n_codes = n_codes
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 1024 * 9  # in TB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0
        self.file = file
        self.select = select

    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)

        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                if self.select == 'all_speaker':
                    self._sample_from_clip_allspeakers(vid, clip)
                else:
                    self._sample_from_clip(vid, clip)

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip_allspeakers(self, vid, clip):
        clip_skeleton = clip['poses']
        clip_audio_raw = clip['audio_raw']

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_audio_list = []

        MINLEN = min(len(clip_skeleton), int(len(clip_audio_raw) * 60 / 16000))

        num_subdivision = math.floor(
            (MINLEN - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[audio_start:audio_end]

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            sample_audio_list.append(sample_audio)
            aux_info.append(motion_info)

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for poses, audio, aux in zip(sample_skeletons_list,
                                                    sample_audio_list, aux_info):
                    poses = np.asarray(poses)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    if self.file == 'ag':
                        v = [poses, audio, aux]
                    elif self.file == 'g':
                        v = [poses, np.array([0]), np.array([0])]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1

    def _sample_from_clip(self, vid, clip):
        clip_skeleton = clip['poses']
        clip_audio_raw = clip['audio_raw']
        clip_codes_raw = clip['code_raw']

        clip_codes_raw = clip_codes_raw.flatten()

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_audio_list = []
        sample_codes_list = []

        MINLEN = min(len(clip_skeleton), int(len(clip_audio_raw) * 60 / 16000), len(clip_codes_raw) * 8)

        num_subdivision = math.floor(
            (MINLEN - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            sample_codes = clip_codes_raw[start_idx//(self.n_poses // self.n_codes):fin_idx//(self.n_poses // self.n_codes)]
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[audio_start:audio_end]

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            sample_audio_list.append(sample_audio)
            sample_codes_list.append(sample_codes)
            aux_info.append(motion_info)

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for poses, audio, codes, aux in zip(sample_skeletons_list,
                                                    sample_audio_list, sample_codes_list, aux_info):
                    poses = np.asarray(poses)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    if self.file == 'ag':
                        v = [poses, audio, codes, aux]
                    elif self.file == 'g':
                        v = [poses, np.array([0]), np.array([0]), np.array([0])]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1


