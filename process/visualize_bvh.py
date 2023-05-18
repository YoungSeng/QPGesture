import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np
import os
import subprocess


joint_name_to_idx = {'Spine': 0, 'Spine1': 1, 'Spine2': 2, 'Spine3': 3, 'Neck': 4, 'Neck1': 5, 'Head': 6,
                     'RightShoulder': 7, 'RightArm': 8, 'RightForeArm': 9, 'RightHand': 10,
                     'LeftShoulder': 11, 'LeftArm': 12, 'LeftForeArm': 13, 'LeftHand': 14}
# joint_idx_to_name = {v: k for k, v in joint_name_to_idx.items()}

joint_links = [(0, 1), (1, 2), (2, 3), (3, 4),
               (3, 7), (7, 8), (8, 9), (9, 10),
               (3, 11), (11, 12), (12, 13), (13, 14), (4, 5), (5, 6)]

colors = ['#15B01A', '#929591', '#380282', '#FFFFCB',
          '#AAA662', '#C79FEF', '#7BC8F6', '#76FF7B',
          '#AAFF32', '#C20078', '#650021', '#01153E', '#6E750E', '#F97306']


# BEAT 27
# joint_name_to_idx = {'Hips': 0, 'Spine': 1, 'Spine1': 2, 'Spine2': 3, 'Spine3': 4, 'Neck': 5, 'Neck1': 6, 'Head': 7,
#                      'HeadEnd': 8, 'RightShoulder': 9, 'RightArm': 10, 'RightForeArm': 11, 'LeftShoulder': 12,
#                      'LeftArm': 13,
#                      'LeftForeArm': 14, 'RightUpLeg': 15, 'RightLeg': 16, 'RightFoot': 17, 'RightForeFoot': 18,
#                      'RightToeBase': 19, 'RightToeBaseEnd': 20, 'LeftUpLeg': 21, 'LeftLeg': 22, 'LeftFoot': 23,
#                      'LeftForeFoot': 24, 'LeftToeBase': 25, 'LeftToeBaseEnd': 26}
#
# joint_links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
#                (4, 9), (9, 10), (10, 11),
#                (4, 12), (12, 13), (13, 14),
#                (0, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),
#                (0, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26)]


# BEAT ALL


def visualize(x, save_path=None, code=None, mode='upper'):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.figure(facecolor='white', edgecolor='white')
    def animate(i):
        print(str(i) + '\r', end='')
        pose = x[i]
        ax.clear()

        for j, pair in enumerate(joint_links):
            ax.plot([pose[pair[0], 0], pose[pair[1], 0]],
                    [pose[pair[0], 2], pose[pair[1], 2]],
                    [pose[pair[0], 1], pose[pair[1], 1]],
                    zdir='z', linewidth=3,
                    color=colors[j]
                    )
        lim = 40
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(lim, -lim)
        # ax.set_ylim3d(-lim, lim)  # Mirror
        if mode == 'upper':
            ax.set_zlim3d(0, lim * 2)  # upper body
        elif mode == 'full':
            ax.set_zlim3d(-lim, lim)  # full body
        else:
            raise ValueError("mode must be upper or full")
        ax.set_xlabel('dim 0')
        ax.set_ylabel('dim 2')
        ax.set_zlabel('dim 1')
        ax.spines['top'].set_visible(False)  # 去掉上边框
        ax.spines['bottom'].set_visible(False)  # 去掉下边框
        ax.spines['left'].set_visible(False)  # 去掉左边框
        ax.spines['right'].set_visible(False)  # 去掉右边框
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(False)
        ax.margins(x=0)
        if code is not None:
            ax.text(lim, lim / 2, lim, '{}'.format(code[i // 8]), 'x', fontsize=14)

    num_frames = len(x)

    if save_path:
        ani = animation.FuncAnimation(fig, animate, interval=1, frames=num_frames, repeat=False)
        ani.save(save_path, fps=60, dpi=300)
        del ani
        plt.close(fig)
    else:
        ani = animation.FuncAnimation(fig, animate, interval=5, frames=num_frames, repeat=False)
        plt.show()


def merge(mp4_path, wav_path):
    out_path = mp4_path.replace('.mp4', '_with_audio.mp4')
    cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', mp4_path, '-i', wav_path, '-strict', '-2', out_path, '-shortest']
    subprocess.call(cmd)


def visualize_main(npy_dir, save_dir, wav_path=None, code=None, mode='upper'):
    files = sorted([f for f in glob.iglob(npy_dir + '*.npy')])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, npy_path in enumerate(files):
        print(npy_path)

        mp4_path = os.path.join(save_dir, npy_path.split('/')[-1].replace('.npy', '.mp4'))

        x = np.load(npy_path)[:60]

        x = x.reshape((x.shape[0], -1, 3))

        # visualize(x)  # show animation
        visualize(x, mp4_path, code, mode)  # save to mp4
        if wav_path:
            merge(mp4_path, wav_path)
        break


if __name__ == '__main__':
    '''
    cd process/
    python visualize_bvh.py
    '''

    code = np.load('/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/code11.npy').flatten()
    npy_dir = "/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/11/"
    save_dir = "/mnt/nfs7/y50021900/My/tmp/TEST/video/"
    wav_path = "/mnt/nfs7/y50021900/My/tmp/TEST/1_wayne_0_103_110.wav"
    # visualize_main(npy_dir, save_dir, wav_path=None, code=code, mode='upper')
    x = np.load(npy_dir + 'generate11_generated.npy')[:60]
    visualize(x.reshape((x.shape[0], -1, 3)), npy_dir + 'generate11_generated.mp4', code, 'upper')  # save to mp4
