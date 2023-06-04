import numpy as np
import pdb


if __name__ == '__main__':
    '''
    cd ./process/
    python fix_device_bug.py
    '''
    source_path = "../data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_txt.npz"
    # source_path = "../data/BEAT/speaker_10_state_0/speaker_10_state_0_test_240_txt.npz"
    save_path = "../data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_txt_2.npz"
    source_dataset = np.load(source_path, allow_pickle=True)
    phases = source_dataset['phase']
    for phase_1 in range(phases.shape[0]):
        for phase_2 in range(phases.shape[1]):
            for phase_3 in range(phases.shape[2]):
                    phases[phase_1][phase_2][phase_3] = phases[phase_1][phase_2][phase_3].detach().cpu()
    np.savez_compressed(save_path, body=source_dataset['body'], phase=phases, mfcc=source_dataset['mfcc'],      # source_path
                        wav=source_dataset['wav'], txt=source_dataset['txt'], aux=source_dataset['aux'],
                        energy=source_dataset['energy'], pitch=source_dataset['pitch'], volume=source_dataset['volume'],
                        context=source_dataset['context'])


