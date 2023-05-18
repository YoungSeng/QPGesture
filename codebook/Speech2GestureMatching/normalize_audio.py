import subprocess
import os


if __name__ == '__main__':
    root = '../../dataset/BEAT/'
    target_path = 'Audio_normalized'
    if not os.path.exists(os.path.join(root, target_path)):
        os.mkdir(os.path.join(root, target_path))
    for item in os.listdir(os.path.join(root, 'Audio')):
        print(item)
        cmd = ['ffmpeg-normalize', os.path.join(root, 'Audio', item), '-o', os.path.join(root, target_path, item), '-ar', '16000']
        subprocess.call(cmd)
