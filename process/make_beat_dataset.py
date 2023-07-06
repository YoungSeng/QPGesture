# -*- coding:utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd. 2022
# @Time         : 10/31/2022
# @File         : make_beat_dataset.py
# Description: process BEAT dataset, handle raw action files, speech, text, MFCC, rotation matrix, rhyme features, etc.

import os
import pdb
import subprocess
import numpy as np
import glob
import math
import copy
import argparse


def make_beat_gesture_audio_dataset(root, save_dir):
    '''
    Merge all speaker data from the original dataset into the target file.
    :param root: original dataset location
    :param save_dir: target dataset location
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    gesture_path = os.path.join(save_dir, 'Motion')
    audio_path = os.path.join(save_dir, 'Audio')
    # TODO: add text process
    # text_path = os.path.join(base_path, 'Transcripts')

    if not os.path.exists(gesture_path):
        os.mkdir(gesture_path)

    if not os.path.exists(audio_path):
        os.mkdir(audio_path)

    source_gesture_path = []
    source_audio_path = []

    for speaker in os.listdir(root):
        for item in os.listdir(os.path.join(root, speaker)):
            if item[-4:] == '.wav':
                source_audio_path.append(os.path.join(root, speaker, item))
            elif item[-4:] == '.bvh':
                source_gesture_path.append(os.path.join(root, speaker, item))
    # check
    for i in source_audio_path:
        if i[:-4] + '.bvh' not in source_gesture_path:
            print(i)
            source_audio_path.remove(i)
    for i in source_gesture_path:
        if i[:-4] + '.wav' not in source_audio_path:
            print(i)
            source_gesture_path.remove(i)
    assert len(source_audio_path) == len(source_gesture_path)

    print('number of training set: {}'.format(len(source_audio_path)))  # 682
    print('processing audio...')
    index = 0
    for item in source_audio_path:
        print(str(index) + ' / ' + str(len(source_audio_path)) + '\r', end='')
        subprocess.call(['cp', '-r', item, audio_path])
        index += 1

    print('Processing gesture...')
    index = 0
    for item in source_gesture_path:
        print(str(index) + ' / ' + str(len(source_gesture_path)) + '\r', end='')
        subprocess.call(['cp', '-r', item, gesture_path])
        index += 1


def remake_beat_bvh(save_dir):
    '''
    Handle the problem of inconsistency between real frames and actual frames.
    :param save_dir: target dataset location
    '''
    from pymo.parsers import BVHParser
    p = BVHParser()
    gesture_path = os.path.join(save_dir, 'Motion')
    for item in os.listdir(gesture_path):
        print('check:', item)
        file_path = os.path.join(gesture_path, item)
        try:
            p.parse(file_path)
        except:
            print('process: ' + item)
            bvh_file = open(file_path, 'r')
            content = bvh_file.readlines()
            length = len(content)  # 15_carlos_0_8_8 -> 5899
            correct_frames = length - 431
            content[429] = 'Frames: ' + str(correct_frames) + '\n'
            file = open(file_path, 'w')
            file.writelines(content)
            bvh_file.close()
            file.close()


def remake_subdataset(save_dir, prefix):
    '''
    Process data for a specific speaker for a specific speaking situation. Generate rotation matrix, audio waveform, text, MFCC, rotation matrix, sox processed audio, speech rhyming features.
    :param save_dir: target dataset location
    :param prefix: a specific speaker for a specific speaking situation
    '''
    from process_beat_txt import get_content, align_words
    from beat_data_to_lmdb import process_bvh
    import librosa
    from speech_feat import AudioProcesser
    import torch.nn.functional as F
    import torch

    _, target_speaker_id, _, target_state = map(str, prefix.split('_'))
    if not os.path.exists(os.path.join(save_dir, prefix)):
        os.mkdir(os.path.join(save_dir, prefix))
    if not os.path.exists(os.path.join(save_dir, prefix, 'Motion')):
        os.mkdir(os.path.join(save_dir, prefix, 'Motion'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'Audio')):
        os.mkdir(os.path.join(save_dir, prefix, 'Audio'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'Transcripts')):
        os.mkdir(os.path.join(save_dir, prefix, 'Transcripts'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'MFCC')):
        os.mkdir(os.path.join(save_dir, prefix, 'MFCC'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'MFCC_20fps')):
        os.mkdir(os.path.join(save_dir, prefix, 'MFCC_20fps'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'Rotation')):
        os.mkdir(os.path.join(save_dir, prefix, 'Rotation'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'Rotation_20fps')):
        os.mkdir(os.path.join(save_dir, prefix, 'Rotation_20fps'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'Wav')):
        os.mkdir(os.path.join(save_dir, prefix, 'Wav'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'Audio_sox')):
        os.mkdir(os.path.join(save_dir, prefix, 'Audio_sox'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'Speech_feat')):
        os.mkdir(os.path.join(save_dir, prefix, 'Speech_feat'))
    if not os.path.exists(os.path.join(save_dir, prefix, 'WavLM_feat')):
        os.mkdir(os.path.join(save_dir, prefix, 'WavLM_feat'))

    BEAT_txt_path = "../data/BEAT/mocap_answer.txt"
    content = get_content(BEAT_txt_path)

    # load the pre-trained checkpoints
    # wavlm_model_path = "../pretrained_model/WavLM-Large.pt"
    # import sys
    # [sys.path.append(i) for i in ['.', '..', './WavLM']]
    # from WavLM import WavLM, WavLMConfig
    # checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))
    # cfg = WavLMConfig(checkpoint['cfg'])
    # model = WavLM(cfg)
    # device = torch.device('cuda:0')
    # model = model.to(device)
    # model.load_state_dict(checkpoint['model'])
    # model.eval()

    for bvh_file in os.listdir(os.path.join(save_dir, 'Motion')):
        print(bvh_file)
        speaker_id, name, state, begin, end = map(str, bvh_file.strip('.bvh').split('_'))
        if speaker_id == target_speaker_id and state == target_state:
            subprocess.call(['cp', '-r', os.path.join(save_dir, 'Motion', bvh_file), os.path.join(save_dir, prefix, 'Motion')])
            subprocess.call(['cp', '-r', os.path.join(save_dir, 'Audio', bvh_file[:-4] + '.wav'), os.path.join(save_dir, prefix, 'Audio')])
            align_words(os.path.join(save_dir, 'Audio', bvh_file[:-4] + '.wav'), content, os.path.join(save_dir, prefix, 'Transcripts'))
            subprocess.call(['cp', '-r', os.path.join(save_dir, 'MFCC_60', bvh_file[:-4] + '.npz'), os.path.join(save_dir, prefix, 'MFCC')])
            poses, _ = process_bvh(os.path.join(save_dir, 'Motion', bvh_file), modetype='rotation', fps=60, output_data_pipe='data_pipe_60_rotation.sav')
            np.savez_compressed(os.path.join(save_dir, prefix, 'Rotation', bvh_file[:-4] + '.npz'), upper=poses)

            wav, _ = librosa.load(os.path.join(save_dir, 'Audio', bvh_file[:-4] + '.wav'), sr=16000)
            np.savez_compressed(os.path.join(save_dir, prefix, 'Wav', bvh_file[:-4] + '.npz'), wav=wav)
            subprocess.call(['sox', os.path.join(save_dir, 'Audio', bvh_file[:-4] + '.wav'), '-b', '16', '-e', 'signed-integer', os.path.join(save_dir, prefix, 'Audio_sox', bvh_file[:-4] + '.wav')])
            ap = AudioProcesser(os.path.join(save_dir, prefix, 'Audio_sox', bvh_file[:-4] + '.wav'), hop_size=256)  # 320 = 20ms, 16000/hop_size = 50
            energy = ap.get_energy()
            pitch = ap.get_pitch(log=True, norm=False)
            volume = ap.calVolume()
            energy = F.interpolate(torch.from_numpy(energy).unsqueeze(0).unsqueeze(0), size=math.ceil(len(wav)/16000*60),
                                   align_corners=True, mode='linear').squeeze(0).squeeze(0).numpy()
            pitch = F.interpolate(torch.from_numpy(pitch).unsqueeze(0).unsqueeze(0), size=math.ceil(len(wav) / 16000 * 60),
                                   align_corners=True, mode='linear').squeeze(0).squeeze(0).numpy()
            volume = F.interpolate(torch.from_numpy(volume).squeeze().unsqueeze(0).unsqueeze(0), size=math.ceil(len(wav) / 16000 * 60),
                                   align_corners=True, mode='linear').squeeze(0).squeeze(0).numpy()
            np.savez_compressed(os.path.join(save_dir, prefix, 'Speech_feat', bvh_file[:-4] + '.npz'), energy=energy,
                                pitch=pitch, volume=volume)

            # 20fps
            poses, _ = process_bvh(os.path.join(save_dir, 'Motion', bvh_file), modetype='rotation', fps=20, output_data_pipe='data_pipe_20_rotation.sav')
            np.savez_compressed(os.path.join(save_dir, prefix, 'Rotation_20fps', bvh_file[:-4] + '.npz'), upper=poses)
            # wav = np.load(os.path.join(save_dir, prefix, 'Wav', bvh_file[:-4] + '.npz'))['wav']
            # wavlm_rep = wav2wavlm(wav, model, device)       # WavLM, out of memory
            # np.savez_compressed(os.path.join(save_dir, prefix, 'WavLM_feat', bvh_file[:-4] + '.npz'), wavlm=wavlm_rep)

            subprocess.call(['cp', '-r', os.path.join(save_dir, 'MFCC_20', bvh_file[:-4] + '.npz'), os.path.join(save_dir, prefix, 'MFCC_20fps')])


def make_dataset(save_dir, prefix, n_frames=240, fps=60, motiontyepe='rotation', mode='duplication', subdivision_stride=30):
    '''
    Convert all the data into an npz file with optional mode/subdivision stride(steps).
    :param save_dir: target dataset location
    :param prefix: a specific speaker for a specific speaking situation
    :param n_frames: The number of frames contained in a step.
    :param fps: FPS(Frames Per Second)
    :param motiontyepe: 'rotation'/'position', will be used in the future.
    :param mode: 'duplication'/'noduplication', Construct a dataset with/without overlapping frames.
    :param subdivision_stride: if 'duplication', it is number of overlapping frames; elif 'noduplication', overlapping frames is n_frames
    '''

    bvh_files = sorted(glob.glob(os.path.join(save_dir, prefix, 'Rotation') + "/*.npz"))
    subname = {'train': [],  # train
             'validation': [],  # validation
             'test': []}  # test
    for bvh_file in bvh_files:
        if '81_86' in bvh_file:continue
        elif '103' in bvh_file: subname['test'].append(bvh_file)      # pick data
        elif '111' in bvh_file:
            subname['validation'].append(bvh_file)
        else:
            subname['train'].append(bvh_file)
    print(len(subname['train']), len(subname['validation']), len(subname['test']))

    def make_duplication_dataset(split_name, stride=30):
        poses_data = []
        mfcc_data = []
        audio_data = []
        for bvh_file in subname[split_name]:
            name = os.path.split(bvh_file)[1][:-4]
            print(name)
            poses = np.load(os.path.join(save_dir, prefix, 'Rotation', name + '.npz'))['upper']
            mfcc = np.load(os.path.join(save_dir, prefix, 'MFCC', name + '.npz'))['mfcc']
            wav = np.load(os.path.join(save_dir, prefix, 'Wav', name + '.npz'))['wav']
            MINLEN = min(len(poses), len(mfcc))     # debug MINLEN=25514
            poses = poses[:MINLEN]
            mfcc = mfcc[:MINLEN]
            wav = wav[:math.floor(MINLEN / fps * 16000)]
            num_subdivision = math.floor((MINLEN - n_frames) / stride) + 1  # floor((K - (N+M)) / S) + 1
            for i in range(num_subdivision):
                start_idx = i * stride
                fin_idx = start_idx + n_frames
                sample_skeletons = poses[start_idx:fin_idx]
                # subdivision_start_time = start_idx / fps
                # subdivision_end_time = fin_idx / fps

                # raw mfcc
                sample_mfcc = mfcc[start_idx:fin_idx]

                # raw audio
                audio_start = math.floor(start_idx / fps * 16000)
                audio_end = audio_start + int(n_frames / fps * 16000)
                sample_audio = wav[audio_start:audio_end]

                poses_data.append(sample_skeletons)
                mfcc_data.append(sample_mfcc)
                audio_data.append(sample_audio)

        np.savez_compressed(os.path.join(save_dir, prefix, prefix + '_' + split_name + '_' + str(stride) + '.npz'),
                                    body=np.array(poses_data), mfcc=np.array(mfcc_data), wav=np.array(audio_data))

    if mode == 'noduplication':
        for key in subname.keys():
            make_duplication_dataset(key, stride=n_frames)
    elif mode == 'duplication':
        for key in subname.keys():
            make_duplication_dataset(key, stride=subdivision_stride)


def dataset_to_code(save_dir, prefix, n_frames=240, model_path=None):
    """
    Converts motion data in a dataset into discrete codes.
    :param save_dir: target dataset location
    :param prefix: a specific speaker for a specific speaking situation
    :param n_frames: The number of frames contained in a step.
    :param model_path: The location of the trained VQVAE model.
    """
    import tqdm
    import torch.nn as nn
    import sys
    [sys.path.append(i) for i in ['.', '..', '../codebook']]
    from codebook.configs.parse_args import parse_args
    args = parse_args()
    from codebook.models.vqvae import VQVAE
    import torch
    mydevice = torch.device('cuda:' + args.gpu)

    from easydict import EasyDict
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v


    config = EasyDict(config)

    def subdataset_to_code(split_name, stride=240, normalize=True):
        dataset_path = os.path.join(save_dir, prefix, prefix + '_' + split_name + '_' + str(stride) + '.npz')
        poses = np.load(dataset_path)['body']
        print(poses.shape)      # N, 240, 135
        poses = poses.reshape(-1, poses.shape[-1])
        # normalize
        if normalize:
            data_mean = np.array(config.data_mean).squeeze()
            data_std = np.array(config.data_std).squeeze()
            std = np.clip(data_std, a_min=0.01, a_max=None)
            poses = (poses - data_mean) / std
        poses = poses.reshape(-1, n_frames, poses.shape[-1])

        with torch.no_grad():
            model = VQVAE(config.VQVAE, 15 * 9)  # n_joints * n_chanels
            model = nn.DataParallel(model, device_ids=[int(config.gpu)])
            model = model.to(mydevice)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_dict'])
            model = model.eval()

            code = []
            i = 0
            for seq in poses:
                in_pose = torch.from_numpy(seq).unsqueeze(0).to(mydevice)
                zs = model.module.encode(in_pose.float())
                # pose_sample = model.module.decode(zs).squeeze(0).data.cpu().numpy()

                code.append(zs[0].squeeze(0).data.cpu().numpy())
                i += 1
        np.savez_compressed(os.path.join(save_dir, prefix, prefix + '_' + split_name + '_' + str(stride) + '_code' + '.npz'),
                                code=np.array(code))

    for key in ['train', 'validation', 'test']:
        subdataset_to_code(key)


def wav2wavlm(wav, model, device):
    import torch
    print(wav.shape)
    with torch.no_grad():
        wav_input_16khz = torch.from_numpy(wav).unsqueeze(0).to(device)
        rep = model.extract_features(wav_input_16khz)[0]
    return rep.detach().cpu().numpy()


def wav_to_wavlm(save_dir, prefix, wavlm_model_path):
    '''
    Converts audio data in a dataset into wavlm features.
    :param save_dir: target dataset location
    :param prefix: a specific speaker for a specific speaking situation
    :param wavlm_model_path: Pre-trained WavLM model locations.
    :return:
    '''
    import torch
    import sys
    [sys.path.append(i) for i in ['.', '..', './WavLM']]
    from WavLM import WavLM, WavLMConfig
    # import torch.nn as nn

    # load the pre-trained checkpoints
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    mydevice = torch.device('cuda:' + args.gpu)
    model = model.to(mydevice)
    model.load_state_dict(checkpoint['model'])
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.eval()

    def make_wavlm_dataset(split_name, stride=240):
        source_wav = np.load(os.path.join(save_dir, prefix, prefix + '_' + split_name + '_' + str(stride) + '.npz'))['wav']
        print(source_wav.shape)
        result = []
        # extract the representation of last layer
        batch_size = 32
        with torch.no_grad():
            i = 0
            for j in range(0, len(source_wav), batch_size):
                print(str(i) + '\r', end='')
                wav_input_16khz = torch.from_numpy(source_wav[j:j+batch_size]).to(mydevice)
                # rep = model.module.extract_features(wav_input_16khz)[0]
                rep = model.extract_features(wav_input_16khz)[0]        # (batch, 64000) -> (batch, 199, 1024)
                # extract the representation of each layer
                # rep, layer_results = model.extract_features(torch.from_numpy(wav_input_16khz).unsqueeze(0), output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
                result.append(rep.cpu())
                i += 1
            result = np.vstack(result)
            print(result.shape)
            np.savez_compressed(
                os.path.join(save_dir, prefix, prefix + '_' + split_name + '_' + str(stride) + '_WavLM' + '.npz'),
                wavlm=np.array(result))

    for key in ['train', 'validation', 'test']:
        make_wavlm_dataset(key)


def wav_to_vq(save_dir, prefix, wavvq_model_path):
    '''
    Converts audio data in a dataset into vq-wav2vec features.
    :param save_dir: target dataset location
    :param prefix: a specific speaker for a specific speaking situation
    :param wavlm_model_path: Pre-trained WavLM model locations.
    :return:
    '''
    import torch
    import fairseq

    # load the pre-trained checkpoints
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wavvq_model_path])
    model = model[0]
    mydevice = torch.device('cuda:' + args.gpu)
    model = model.to(mydevice)
    model.eval()

    def make_wav2vec_dataset(split_name, stride=240):
        source_wav = np.load(os.path.join(save_dir, prefix, prefix + '_' + split_name + '_' + str(stride) + '.npz'))['wav']
        print(source_wav.shape)
        result = []
        # extract the representation of last layer
        batch_size = 32
        with torch.no_grad():
            i = 0
            for j in range(0, len(source_wav), batch_size):
                print(str(i) + '\r', end='')
                wav_input_16khz = torch.from_numpy(source_wav[j:j+batch_size]).to(mydevice)
                z = model.feature_extractor(wav_input_16khz)
                _, idxs = model.vector_quantizer.forward_idx(z)

                result.append(idxs.cpu())
                i += 1
            result = np.vstack(result)
            print(result.shape)
            np.savez_compressed(
                os.path.join(save_dir, prefix, prefix + '_' + split_name + '_' + str(stride) + '_WavVQ' + '.npz'),
                wavvq=np.array(result))

    for key in ['train', 'validation', 'test']:
        make_wav2vec_dataset(key)


def make_txt_dataset(save_dir, prefix, n_frames=240, fps=60, motiontyepe='rotation', mode='duplication',
                     subdivision_stride=30, num_frames_code=30):
    '''
    Convert all the data into an npz file with optional mode/subdivision stride(steps).
    :param save_dir: target dataset location
    :param prefix: a specific speaker for a specific speaking situation
    :param n_frames: The number of frames contained in a step.
    :param fps: fps: FPS(Frames Per Second)
    :param motiontyepe: 'rotation'/'position', will be used in the future.
    :param mode: 'duplication'/'noduplication', Construct a dataset with/without overlapping frames.
    :param subdivision_stride: if 'duplication', it is number of overlapping frames; elif 'noduplication', overlapping frames is n_frames
    :return:
    '''

    from sentence_transformers import SentenceTransformer
    model_sentencetransformer = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda:' + args.gpu)

    bvh_files = sorted(glob.glob(os.path.join(save_dir, prefix, 'Rotation') + "/*.npz"))
    subname = {'train': [],  # train
             'validation': [],  # validation
             'test': []}  # test
    for bvh_file in bvh_files:
        if '81_86' in bvh_file:continue
        elif '103' in bvh_file: subname['test'].append(bvh_file)      # pick data
        elif '111' in bvh_file:
            subname['validation'].append(bvh_file)
        else:
            subname['train'].append(bvh_file)
    print(len(subname['train']), len(subname['validation']), len(subname['test']))

    def make_duplication_dataset(split_name, stride=30):
        poses_data = []
        mfcc_data = []
        audio_data = []
        txt_data = []
        aux_data = []
        energy_data = []
        pitch_data = []
        volume_data = []

        context_data = []
        phase_data = []

        step_sz = int(stride / num_frames_code)
        stride_time = stride // fps

        for bvh_file in subname[split_name]:
            name = os.path.split(bvh_file)[1][:-4]
            print(name)

            poses = np.load(os.path.join(save_dir, prefix, 'Rotation', name + '.npz'))['upper']
            mfcc = np.load(os.path.join(save_dir, prefix, 'MFCC', name + '.npz'))['mfcc']
            wav = np.load(os.path.join(save_dir, prefix, 'Wav', name + '.npz'))['wav']
            speech_feat = np.load(os.path.join(save_dir, prefix, 'Speech_feat', name + '.npz'))
            energy = speech_feat['energy']
            pitch = speech_feat['pitch']
            volume = speech_feat['volume']

            words_with_timestamps = []
            with open(os.path.join(save_dir, prefix, 'Transcripts', name + '.txt'), 'r', encoding='utf-8') as input_file:
                for line in input_file.readlines():
                    content = line.strip().split('\t')
                    content[0] = eval(content[0])
                    content[1] = eval(content[1])
                    words_with_timestamps.append(content)
            # pdb.set_trace()
            MINLEN = min(len(poses), len(mfcc))     # debug MINLEN=25514
            poses = poses[:MINLEN]
            mfcc = mfcc[:MINLEN]
            wav = wav[:math.floor(MINLEN / fps * 16000)]
            num_subdivision = math.floor((MINLEN - n_frames) / stride) + 1  # floor((K - (N+M)) / S) + 1

            phase = np.load(os.path.join(save_dir, prefix, 'Phase', name + '.npz'), allow_pickle=True)['phase']
            print(poses.shape, phase.shape)

            for i in range(num_subdivision):

                start_idx = i * stride
                fin_idx = start_idx + n_frames
                sample_skeletons = poses[start_idx:fin_idx]
                sample_phase = phase[start_idx:fin_idx]
                # subdivision_start_time = start_idx / fps
                # subdivision_end_time = fin_idx / fps

                # raw mfcc
                sample_mfcc = mfcc[start_idx:fin_idx]

                # raw speech feature
                sample_energy = energy[start_idx:fin_idx]
                sample_pitch = pitch[start_idx:fin_idx]
                sample_volume = volume[start_idx:fin_idx]

                # raw audio
                audio_start = math.floor(start_idx / fps * 16000)
                audio_end = audio_start + int(n_frames / fps * 16000)
                sample_audio = wav[audio_start:audio_end]

                poses_data.append(sample_skeletons)
                phase_data.append(sample_phase)
                mfcc_data.append(sample_mfcc)
                audio_data.append(sample_audio)

                energy_data.append(sample_energy)
                pitch_data.append(sample_pitch)
                volume_data.append(sample_volume)

                start_time = start_idx/fps
                end_time = fin_idx/fps

                sample_txt = []
                while words_with_timestamps != [] and (words_with_timestamps[0][0] + words_with_timestamps[0][1])/2 < end_time:
                    sample_txt.append(words_with_timestamps.pop(0))
                txt_data.append(sample_txt)

                code_txt = []

                sample_txt_ = copy.deepcopy(sample_txt)
                tmp_code_txt = [[] for _ in range(num_frames_code)]

                while sample_txt_ != []:
                    tmp = sample_txt_.pop(0)
                    tmp_code_txt[int((tmp[0] % stride_time + (tmp[1] % stride_time if tmp[1] % stride_time != 0 else stride_time)) * 60 / 2 / step_sz)].append(tmp)  # Prevent n*stride_time from being treated as 0
                code_txt.append(tmp_code_txt)

                sample_code_text = [[] for _ in range(num_frames_code)]

                for j in range(num_frames_code):        # for every code
                    for tmp_code_txt in code_txt[0][(j-3 if j-3 > 0 else 0):(j+4 if j+4 < num_frames_code else num_frames_code)]:
                        for tmp in tmp_code_txt:
                            sample_code_text[j].append(tmp[2])

                for j in range(num_frames_code):  # for every code
                    sample_code_text[j] = model_sentencetransformer.encode([' '.join(sample_code_text[j])])
                context_data.append(sample_code_text)

                aux_data.append([name, start_time, end_time])

        np.savez_compressed(os.path.join(save_dir, prefix, prefix + '_' + split_name + '_' + str(stride) + '_' + 'txt' + '.npz'),
                            body=np.array(poses_data), mfcc=np.array(mfcc_data), wav=np.array(audio_data),
                            txt=np.array(txt_data), aux=np.array(aux_data), energy=np.array(energy_data),
                            pitch=np.array(pitch_data), volume=np.array(volume_data), context=np.array(context_data),
                            phase=np.array(phase_data))

    if mode == 'noduplication':
        for key in subname.keys():
            make_duplication_dataset(key, stride=n_frames)
    elif mode == 'duplication':
        for key in subname.keys():
            make_duplication_dataset(key, stride=subdivision_stride)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make_test_data')
    parser.add_argument('--config', default='./configs/codebook.yml')
    parser.add_argument('--BEAT_path', type=str, default="../dataset/orig_BEAT/speakers/")
    parser.add_argument('--save_dir', type=str, default="../dataset/BEAT")
    parser.add_argument('--prefix', type=str, default="speaker_10_state_0")     # speaker_ID:1, self_talk:0
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--step', type=str, default="1")
    args = parser.parse_args()

    root = args.BEAT_path
    save_dir = args.save_dir
    prefix = args.prefix

    if args.step == "1":
        make_beat_gesture_audio_dataset(root, save_dir)         # make dataset
        remake_beat_bvh(save_dir)       # check bvh
    elif args.step == "2":
        remake_subdataset(save_dir, prefix)
        make_dataset(save_dir, prefix, mode='noduplication')
        # make_dataset(save_dir, prefix, mode='duplication')
    elif args.step == "3":
        dataset_to_code(save_dir, prefix, n_frames=240, model_path="../pretrained_model/codebook_checkpoint_best.bin")
        wavlm_path = "../pretrained_model/WavLM-Large.pt"
        wav_to_wavlm(save_dir, prefix, wavlm_model_path=wavlm_path)
    elif args.step == "4":        # sometimes not enough memory to run together...
        wavvq_path = '../process/vq-wav2vec.pt'
        wav_to_vq(save_dir, prefix, wavvq_model_path=wavvq_path)
        make_txt_dataset(save_dir, prefix, mode='noduplication')
