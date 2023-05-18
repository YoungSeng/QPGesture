import numpy as np
import os
import scipy.io.wavfile as wavfile

from constant import SR, WAV_TEST_SIZE


def normalize_data(data, mean, std):
    return (data - mean) / (std + 1E-8)


def inv_normalize_data(data, mean, std):
    return data * std + mean


def generate_wavfile(audio, audio_path):
    def pad_audio(audio, num_audio_samples):
        if len(audio) > num_audio_samples:
            audio = audio[:num_audio_samples]
        elif len(audio) < num_audio_samples:
            audio = np.pad(audio, [0, num_audio_samples - len(audio)], mode='constant', constant_values=0)
        return audio

    if not (os.path.exists(audio_path)):
        os.makedirs(audio_path)

    for i in range(len(audio)):
        wavfile.write(audio_path + str(i) + '.wav', SR, pad_audio(audio[i], WAV_TEST_SIZE))


def normalize_feat(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std = np.clip(std, a_min=1E-8, a_max=None)
    norm_data = (data - mean) / std
    return norm_data, mean, std
