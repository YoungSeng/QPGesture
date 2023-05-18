import librosa
import os
import numpy as np
import torch
import fairseq
import math
import argparse


def process_audio(wavvq_model_path, audio_path, save_path, gpu_num):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    wav, _ = librosa.load(audio_path, sr=16000)
    print(wav.shape)        # 400840, 25.0525s
    np.savez_compressed(os.path.join(save_path, audio_path.split('/')[-1][:-4] + '_wav.npz'), wav=wav)

    MINLEN = wav.shape[0] / 16000 * fps
    num_subdivision = math.floor((MINLEN - n_frames) / n_frames) + 1  # floor((K - (N+M)) / S) + 1
    audio_data = []
    for i in range(num_subdivision):
        start_idx = i * n_frames
        # fin_idx = start_idx + n_frames
        # subdivision_start_time = start_idx / fps
        # subdivision_end_time = fin_idx / fps
        # raw mfcc
        # sample_mfcc = mfcc[start_idx:fin_idx]

        # raw audio
        audio_start = math.floor(start_idx / fps * 16000)
        audio_end = audio_start + int(n_frames / fps * 16000)
        sample_audio = wav[audio_start:audio_end]
        audio_data.append(sample_audio)
    source_wav = np.array(audio_data)
    print(source_wav.shape)
    np.savez_compressed(os.path.join(save_path,  'wav_' + str(n_frames) + '.npz'), wav=source_wav)

    # wavvq
    # load the pre-trained checkpoints
    device = torch.device('cuda:' + gpu_num if torch.cuda.is_available() else 'cpu')
    print(device)
    # wavvq_model_path = torch.load(wavvq_model_path, map_location='cpu')
    # wavvq_model_path = {"vq-wav2vec.pt":wavvq_model_path}
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wavvq_model_path])
    model = model[0]
    model = model.to(device)
    model.eval()

    # extract the representation of last layer
    batch_size = 32
    result = []
    with torch.no_grad():
        i = 0
        for j in range(0, len(source_wav), batch_size):
            print(str(i) + '\r', end='')
            wav_input_16khz = torch.from_numpy(source_wav[j:j + batch_size]).to(device)
            z = model.feature_extractor(wav_input_16khz)
            _, idxs = model.vector_quantizer.forward_idx(z)

            result.append(idxs.cpu())
            i += 1
        result = np.vstack(result)
        print(result.shape)
        wavvq_result = np.array(result)
        np.savez_compressed(os.path.join(save_path, 'wavvq_' + str(n_frames) + '.npz'), wavvq=wavvq_result)     # (6, 398, 2)
        random_tmp = np.random.rand(2, 2, 2)
        np.savez_compressed(os.path.join(save_path, 'testing_data.npz'),
                            body=random_tmp, mfcc=random_tmp, wav=source_wav,
                            txt=random_tmp, aux=random_tmp, energy=random_tmp,
                            pitch=random_tmp, volume=random_tmp, context=random_tmp,
                            phase=random_tmp)


if __name__ == '__main__':
    wavvq_model_path = './vq-wav2vec.pt'
    n_frames = 240
    fps = 60
    parser = argparse.ArgumentParser(description='make_test_data')
    parser.add_argument('--audio_path', type=str, default="../data/Example3/4.wav")
    parser.add_argument('--save_path', type=str, default="../data/Example3/4")
    parser.add_argument('--gpu', type=str, default="0")
    args = parser.parse_args()
    process_audio(wavvq_model_path, args.audio_path, args.save_path, args.gpu)
