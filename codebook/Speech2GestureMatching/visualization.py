import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
import subprocess

from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from constant import UPPERBODY_PARENT, FILTER_SMOOTH_STD
from utils import generate_wavfile


def plot_animation(joints, parents, filename=None, fps=15, axis_scale=.50):    
    joints[:, :, 1] -= .1

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-axis_scale, axis_scale)
    ax.set_zlim3d( 0, axis_scale)
    ax.set_ylim3d(-axis_scale, axis_scale)
    ax.set_axis_off()

    ax.view_init(elev=0, azim=-90)
    
    lines = []
    lines.append([plt.plot([0,0], [0,0], [0,0], color='red', 
        lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(joints.shape[1])])

    def animate(i):
        changed = []
        
        for j in range(len(parents)):
            lines[0][j].set_data(np.array([[joints[i,j,0], joints[i,parents[j],0]],[-joints[i,j,2],-joints[i,parents[j],2]]]))
            lines[0][j].set_3d_properties(np.array([ joints[i,j,1],joints[i,parents[j],1]]))

        changed += lines
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(joints.shape[0]), interval=1000/fps)

    if filename != None:
        ani.save(filename, fps=fps, codec='mpeg4', bitrate=13934)
        ani.event_source.stop()
        del ani
        plt.close()


def generate_seq_videos(seqs, prefix, gaussian_smooth=False, Savitzky_Golay_smooth=False, vis=True):
    import sys
    [sys.path.append(i) for i in ['../../process']]
    sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

    from process.beat_data_to_lmdb import process_bvh
    from process.process_bvh import make_bvh_GENEA2020_BT
    from process.bvh_to_position import bvh_to_npy
    from process.visualize_bvh import visualize

    num_seqs, _, num_frames = seqs.shape
    result = []

    for i in range(num_seqs):
        pred_motion = seqs[i]
        pred_motion = pred_motion.transpose((1, 0))

        pred_motion = pred_motion.reshape((num_frames, -1, 9))

        T, J, D = pred_motion.shape
        pred_motion = pred_motion.reshape((T, -1))
        if gaussian_smooth:
            pred_motion = gaussian_filter1d(pred_motion, FILTER_SMOOTH_STD, axis=0)
        if Savitzky_Golay_smooth:
            for j in range(pred_motion.shape[1]):
                pred_motion[:, j] = savgol_filter(pred_motion[:, j], 15, 2)
        result.append(pred_motion)
    out_poses = np.vstack(result)

    save_path = "/mnt/nfs7/y50021900/My/tmp/TEST/npy_position/"
    # pipeline_path = '../../process/resource/data_pipe_60.sav'        # Trinity
    pipeline_path = '../../process/resource/data_pipe_60_rotation.sav'     # BEAT
    # prefix = '18'
    save_path = os.path.join(save_path, prefix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    bvh_path = '/mnt/nfs7/y50021900/My/data/BEAT0909/Motion/1_wayne_0_103_110.bvh'
    print('process bvh...')
    if not os.path.exists(os.path.join(save_path, 'rotation' + 'GT' + '.npy')):
        poses, _ = process_bvh(bvh_path, modetype='rotation')
        np.save(os.path.join(save_path, 'rotation' + 'GT' + '.npy'), poses)
        make_bvh_GENEA2020_BT(save_path, filename_prefix='GT', poses=poses[:3600], smoothing=False, pipeline_path=pipeline_path)
    else:
        print('npy already exists!')
    print('rotation npy to bvh...')
    make_bvh_GENEA2020_BT(save_path, prefix, out_poses, smoothing=False, pipeline_path=pipeline_path)
    if vis:
        print('bvh to position npy...')
        bvh_path_generated = os.path.join(save_path, prefix + '_generated.bvh')
        bvh_to_npy(bvh_path_generated, save_path)
        print('visualize code...')
        npy_generated = np.load(os.path.join(save_path, prefix + '_generated.npy'))
        out_video = os.path.join(save_path, prefix + '_generated.mp4')
        visualize(npy_generated.reshape((npy_generated.shape[0], -1, 3)), out_video, None, 'upper')

# def generate_seq_videos(seqs, audio, parent, out_vid_path, prefix_fn="knn"):
#     tmp_path = os.path.join(out_vid_path, "tmp/")
#     num_seqs, _, num_frames = seqs.shape
#
#     if not os.path.exists(tmp_path):
#         os.makedirs(tmp_path)
#
#     generate_wavfile(audio, tmp_path)
#
#     for i in range(num_seqs):
#         pred_motion =  seqs[i]
#         pred_motion = pred_motion.transpose((1, 0))
#
#         pred_motion = pred_motion.reshape((num_frames, -1, 3))
#         pred_motion_body, pred_motion_rhand, pred_motion_lhand = \
#             pred_motion[:, :13, :], pred_motion[:, 13:34, :], pred_motion[:, 34:, :]
#
#         # adjust the arm length to prevent unnatural arm crossing due to the monocular tracking error of the data
#         pred_motion_body[:, 4:5, :] = \
#             pred_motion_body[:, 3:4, :] + (pred_motion_body[:, 4:5, :] - pred_motion_body[:, 3:4, :]) * 0.85
#         pred_motion_body[:, 7:8, :] = \
#             pred_motion_body[:, 6:7, :] + (pred_motion_body[:, 7:8, :] - pred_motion_body[:, 6:7, :]) * 0.85
#
#         pred_motion_rhand += pred_motion_body[:, 4:5, :]
#         pred_motion_lhand += pred_motion_body[:, 7:8, :]
#         pred_motion = np.concatenate((pred_motion_body, pred_motion_rhand, pred_motion_lhand), axis=1)
#
#         T, J, D = pred_motion.shape
#         pred_motion = pred_motion.reshape((T, -1))
#         pred_motion = gaussian_filter1d(pred_motion, FILTER_SMOOTH_STD, axis=0)
#         pred_motion = pred_motion.reshape((T, J, D))
#         pred_motion[:, :, 1] *= -1
#
#         plot_animation(pred_motion, parent, f"{tmp_path}/{prefix_fn}_pred_{i}.mp4")
#
#         out = subprocess.call(['ffmpeg',
#                                 '-y',
#                                 '-i',
#                                 f"{tmp_path}/{prefix_fn}_pred_{i}.mp4",
#                                 '-i',
#                                 f"{tmp_path}/{i}.wav",
#                                 '-c:v',
#                                 'copy',
#                                 '-c:a',
#                                 'aac',
#                                 '-strict',
#                                 'experimental',
#                                 f"{out_vid_path}/{prefix_fn}_pred_{i}.mp4"])
#
#     out = subprocess.call(['rm', '-rf', f"{tmp_path}"])
