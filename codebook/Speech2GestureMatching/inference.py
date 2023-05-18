import subprocess
import sys
import os
[sys.path.append(i) for i in ['.', '..', '../VisualizeCodebook', '../../process']]
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from process.speech_feat import AudioProcesser

from mfcc import MFCC
import librosa
import soundfile as sf      # write
import numpy as np
import torch.nn.functional as F
import torch

from VisualizeCodebook import visualizeCodeAndWrite


def main(wav_path):
    # normalize audio
    name = wav_path.split('/')[-1][:-4]
    output_fold = wav_path[:-4]
    if not os.path.exists(output_fold):
        os.mkdir(output_fold)
    wav_norm_path = output_fold + '/' + name + '_norm.wav'
    wav_norm_mfcc_path = wav_norm_path[:-4] + '_mfcc.npz'
    wav_sox_path = output_fold + '/' + name + '_sox.wav'

    # resample
    wav_, fs_ = librosa.load(wav_path, sr=16000)
    sf.write(wav_path, wav_, 16000)

    cmd = ['ffmpeg-normalize', wav_path, '-o', wav_norm_path, '-ar', '16000']
    subprocess.call(cmd)
    # extract mfcc
    obj = MFCC(frate=60)
    wav, fs = librosa.load(wav_norm_path, sr=16000)
    mfcc = obj.sig2s2mfc_energy(wav, None)  # (13237, 15) with -1:1.275、2.275、3.375...
    subprocess.call(['sox', wav_path, '-b', '16', '-e', 'signed-integer', wav_sox_path])
    ap = AudioProcesser(wav_sox_path, hop_size=256)  # 320 = 20ms, 16000/hop_size = 50
    energy = ap.get_energy()
    pitch = ap.get_pitch(log=True, norm=False)
    volume = ap.calVolume()
    energy = F.interpolate(torch.from_numpy(energy).unsqueeze(0).unsqueeze(0), size=len(mfcc),
                           align_corners=True, mode='linear').squeeze(0).squeeze(0).numpy()
    pitch = F.interpolate(torch.from_numpy(pitch).unsqueeze(0).unsqueeze(0), size=len(mfcc),
                          align_corners=True, mode='linear').squeeze(0).squeeze(0).numpy()
    volume = F.interpolate(torch.from_numpy(volume).squeeze().unsqueeze(0).unsqueeze(0), size=len(mfcc),
                           align_corners=True, mode='linear').squeeze(0).squeeze(0).numpy()

    print(mfcc[:, :-1].shape, energy.shape, pitch.shape, volume.shape)
    np.savez_compressed(wav_norm_mfcc_path, mfcc=np.expand_dims(mfcc[:, :-1], axis=0),
                        energy=np.expand_dims(energy, axis=0), pitch=np.expand_dims(pitch, axis=0),
                        volume=np.expand_dims(volume, axis=0))

    cmd = ['python', 'GestureKNN.py',
           '--train_database=../../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240_txt.npz',
           '--test_data=' + wav_norm_mfcc_path,
           '--out_knn_filename=' + output_fold + '/knn_pred.npz',
           '--out_video_path=./output/output_video_folder/',
           '--train_codebook=../../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240_code.npz',
           '--codebook_signature=../BEAT_output_60fps_rotation/code.npz',
           '--train_wavlm=../../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_train_240_WavLM.npz',
           '--test_wavlm=../../data/BEAT0909/speaker_1_state_0/speaker_1_state_0_test_240_WavLM.npz']
    subprocess.call(cmd)

    visualizeCodeAndWrite(code_path=output_fold + '/knn_pred.npz', save_path=output_fold, prefix='result_' + name,
                          pipeline_path='../../process/resource/data_pipe_60_rotation.sav', generateGT=False)


if __name__ == '__main__':
    '''
    cd Speech2GestureMatching/
    python inference.py --config=../configs/codebook.yml --train --no_cuda 2 --gpu 2
    '''
    demo_wav = ['GENEA2022_the_Talking_With_Hands_16.2M_val_2022_v1_000.wav',
                'GENEA2022_the_Talking_With_Hands_16.2M_val_2022_v1_002.wav',
                'GENEA2020_Trinity_TestSeq001.wav',
                'My_model_assets_test_wav_16000_source_source.wav',
                'My_model_assets_test_wav_16000_target_target.wav']
    wav_path = './demo_wav/' + demo_wav[0]
    main(wav_path)
