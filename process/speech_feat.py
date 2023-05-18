import os
import glob
from pathlib import PurePosixPath
import numpy as np
import pyworld as pw
import soundfile as sf
import librosa
import python_speech_features as psf
import math
import wave


class AudioProcesser:
    def __init__(self, wav_path, hop_size):
        self.hop_size = hop_size
        self.wav_data, self.sr = sf.read(wav_path)
        # make sure input 16kHz audio
        assert self.sr == 16000
        fw = wave.open(wav_path, 'r')
        params = fw.getparams()
        nchannels, sampwidth, self.framerate, self.nframes = params[:4]
        strData = fw.readframes(self.nframes)
        self.waveData = np.fromstring(strData, dtype=np.int16)

    def get_pitch(self, eps=1e-5, log=True, norm=True):
        _f0, t = pw.dio(self.wav_data, self.sr, frame_period=self.hop_size / self.sr * 1000)  # raw pitch extractor
        f0 = pw.stonemask(self.wav_data, _f0, t, self.sr)  # pitch refinement
        if log:
            f0 = np.log(np.maximum(eps, f0))
        if norm:
            f0 = (f0 - f0.mean()) / f0.std()

        return f0

    def wav2mel(self, fft_size=1024, hop_size=256,
                win_length=1024, win_mode="hann",
                num_mels=80, fmin=80,
                fmax=7600, eps=1e-10):
        # get amplitude spectrogram
        x_stft = librosa.stft(self.wav_data, n_fft=fft_size, hop_length=hop_size,
                              win_length=win_length, window=win_mode, pad_mode="constant")
        spc = np.abs(x_stft)  # (n_bins, T)

        # get mel basis
        fmin = 0 if fmin == -1 else fmin
        fmax = self.sr / 2 if fmax == -1 else fmax
        mel_basis = librosa.filters.mel(self.sr, fft_size, num_mels, fmin, fmax)
        mel = mel_basis @ spc
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)

        return mel.T

    def get_energy(self):
        """Extract energy same with FastSpeech2
        """
        mel = self.wav2mel(hop_size=self.hop_size)
        energy = np.sqrt((np.exp(mel) ** 2).sum(-1))

        return energy

    def get_energy_psf(self):
        """Directly use python package python_speech_features to extract pitch.
        Can be compared with self.get_energy(), which one is better.
        """
        # python_speech_features use scipy.io.wavfile to read audio
        # so the returned audio data ranges between (-2^15, 2^15)
        wav_data = self.wav_data * (2 ** 15)
        fbank, energy = psf.fbank(wav_data, samplerate=self.sr, winstep=self.hop_size / self.sr)
        energy = np.log(energy)

        return fbank, energy

    # def get_ppg(self):
    #     return self.ppg_extractor.extract_ppg_from_sentence(self.wav_data, self.sr)

    # method 1: absSum
    def calVolume(self, frameSize=256, overLap=128):
        waveData = self.waveData * 1.0 / max(abs(self.waveData))
        wlen = len(waveData)
        step = frameSize - overLap
        frameNum = int(math.ceil(wlen * 1.0 / step))
        volume = np.zeros((frameNum, 1))
        for i in range(frameNum):
            curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
            curFrame = curFrame - np.median(curFrame)  # zero-justified
            volume[i] = np.sum(np.abs(curFrame))
        return volume

    # method 2: 10 times log10 of square sum
    def calVolumeDB(self, frameSize=256, overLap=128):
        # waveData = librosa.util.normalize(self.waveData)
        waveData = self.waveData * 1.0 / max(abs(self.waveData))  # normalization
        wlen = len(waveData)
        step = frameSize - overLap
        frameNum = int(math.ceil(wlen * 1.0 / step))
        volume = np.zeros((frameNum, 1))
        for i in range(frameNum):
            curFrame = waveData[np.arange(i * step, min(i * step + frameSize, wlen))]
            curFrame = curFrame - np.mean(curFrame)  # zero-justified
            volume[i] = 10 * np.log10(np.sum(curFrame * curFrame))
        return volume


# import torchaudio
# def speech_file_to_array_fn(path, sampling_rate=16000):
#     speech_array, _sampling_rate = torchaudio.load(path)
#     resampler = torchaudio.transforms.Resample(sampling_rate)
#     speech = resampler(speech_array).squeeze().numpy()
#     return speech

import csv

if __name__ == "__main__":
    '''
    wav_path = "<..your path/GENEA/genea_challenge_2022/dataset/TEST_2/trn/wav_/val_2022_v1_001>"
    import soundfile as sf

    src_sig, sr = sf.read(wav_path + ".wav")        # (2778300,) -> 16000 = 173.64375s
    print(src_sig.shape)
    if sr != 16000:
        dst_sig = librosa.resample(src_sig, sr, 16000)
        wav_path = wav_path + '_resample'
        sf.write(wav_path + '.wav', dst_sig, 16000)

    ap = AudioProcesser(wav_path + ".wav", hop_size=128)  # 320 = 20ms, 16000/hop_size = 50
    energy = ap.get_energy()
    pitch = ap.get_pitch(log=True, norm=False)

    # fbank, energy_psf = ap.get_energy_psf()

    # volume = ap.calVolumeDB()
    volume = ap.calVolume()

    # print(fbank.shape, energy_psf.shape, energy.shape, pitch.shape, volume.shape)     # (7873, 26) (7873,) (7876,) (7876,) (7875, 1)
    # print(energy.max(), energy_psf.max(), pitch.max(), volume.max())      # 1.3716236523001186 16.577135714439926 6.512935064927014

    volume_ = np.array([i[0] for i in volume])
    print(energy.shape, pitch.shape, volume_, volume_.shape)     # (7873, 26) (7873,) (7876,) (7876,) (7875, 1)
    print(energy.max(), energy.min(), pitch.max(), pitch.min(), volume.max(), volume.min())      # 1.3716236523001186 16.577135714439926 6.512935064927014

    import matplotlib.pyplot as pl
    import matplotlib
    matplotlib.use('Agg')
    time = np.arange(0, ap.nframes) * (1.0 / ap.framerate)
    time2 = np.arange(0, len(volume)) * (256 - 128) * 1.0 / ap.framerate
    pl.subplot(211)
    pl.plot(time, ap.waveData)
    pl.ylabel("Amplitude")
    pl.subplot(212)
    pl.plot(time2, volume)
    pl.ylabel("volume")
    pl.savefig('2.jpg')

    # 2778300/7876=352.75520;2910600/8251=352.75724

    '''

    # root = "<..your path/GENEA/genea_challenge_2022/dataset/v1_18/tst/wav_48000/>"
    # out_path = "<..your path/GENEA/genea_challenge_2022/dataset/v1_18/tst/wav/>"
    # for item in os.listdir(root):
    #     src_sig, sr = sf.read(root + item)
    #     if sr != 16000:
    #         print(item)
    #         dst_sig = librosa.resample(src_sig, sr, 16000)
    #         sf.write(out_path + item, dst_sig, 16000)

    # wav_path = "<..your path/GENEA/genea_challenge_2022/dataset/TEST_2/trn/wav_/val_2022_v1_001.wav>"
    # y = speech_file_to_array_fn(wav_path)
    # print(y, y.shape)

    path = "<..your path/GENEA/genea_challenge_2022/dataset/v1_18_1/trn/trn_2022_v1_metadata.csv>"

    dictionary = {}
    cnt = 0
    # 读取csv文件
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            cnt += 1
            dictionary[row[0]] = row[-1]

    print(cnt, dictionary)
