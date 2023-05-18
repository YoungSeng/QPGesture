import os
import pdb
import re
import subprocess
import sys
import librosa
import soundfile as sf
import logging

# sys.path.insert(0, '../../tri/gentle/')

import gentle

gentle_resources = gentle.Resources()

def get_content(txt_path):
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    English_txt = []
    tmp = ''

    with open(txt_path, 'r', encoding='utf-8') as beat_txt:
        for line in beat_txt.readlines():
            content = line.strip()
            if content != '':
                if not zhPattern.search(content) and not content[0].isdigit():
                    English_txt.append(content)
                elif content[0].isdigit() and len(content) > len(tmp):
                     tmp = content

    English_txt.insert(0, None)
    English_txt.insert(83, '')
    # print(len(English_txt))
    # print(tmp)
    return English_txt      # 0 - 118(0 is None, 83 is '')


def align(gentle_path, audio_path, content, output_path):
    speaker_id, name, state, begin, end = map(str, audio_path.split('/')[-1].strip('.wav').split('_'))
    tmp_dir = 'tmp_' + begin + '_' + end + '.txt'
    if not os.path.exists(tmp_dir):
        with open(tmp_dir, 'w', encoding='utf-8') as tmp_file:
            for i in range(int(begin), int(end) + 1, 1):
                tmp_file.write(content[i])
                tmp_file.write(' ')
    cmd = ['python3', gentle_path + 'align.py', audio_path, tmp_dir, '-o', output_path + speaker_id + '_' + name + '_' + state + '_' + begin + '_' + end + '.txt']
    subprocess.call(cmd)


def align_words(audio_path, total_text, output_path):
    audio, audio_sr = librosa.load(audio_path, mono=True, sr=16000, res_type='kaiser_fast')
    # resample audio to 8K
    audio_8k = librosa.resample(audio, 16000, 8000)
    wave_file = 'temp.wav'
    sf.write(wave_file, audio_8k, 8000, 'PCM_16')

    speaker_id, name, state, begin, end = map(str, audio_path.split('/')[-1].strip('.wav').split('_'))
    text = ''
    for i in range(int(begin), int(end) + 1, 1):
        text += total_text[i]
        text += ' '

    # run gentle to align words
    aligner = gentle.ForcedAligner(gentle_resources, text, nthreads=2, disfluency=False,
                                   conservative=False)
    gentle_out = aligner.transcribe(wave_file, logging=logging)
    words_with_timestamps = []
    for i, gentle_word in enumerate(gentle_out.words):
        if gentle_word.case == 'success':
            words_with_timestamps.append([gentle_word.word, gentle_word.start, gentle_word.end])
        elif 0 < i < len(gentle_out.words) - 1:
            words_with_timestamps.append([gentle_word.word, gentle_out.words[i-1].end, gentle_out.words[i+1].start])

    output_path = os.path.join(output_path, speaker_id + '_' + name + '_' + state + '_' + begin + '_' + end + '.txt')
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for item in words_with_timestamps:
            if None in item:
                continue
            else:
                output_file.write(str(item[1]) + '\t' + str(item[2]) + '\t' + item[0] + '\n')

    return words_with_timestamps


if __name__ == '__main__':
    '''
    193 Trimodal
    cd process/
    python process_beat_txt.py
    # cd gentle/
    # python /nfs7/y50021900/My/process/process_beat_txt.py
    '''
    BEAT_txt_path = '../beat-main/datasets/BEAT0909/mocap_answer.txt'
    content = get_content(BEAT_txt_path)
    gentle_path = '../../tri/gentle/'
    audio_path = '../data/BEAT0909/Audio/1_wayne_0_103_110.wav'
    output_path = '../tmp/TEST/Transcripts/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # align(gentle_path, audio_path, content, output_path)
    words_with_timestamps = align_words(audio_path, content, output_path)
