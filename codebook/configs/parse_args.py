import configargparse
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Codebook')
    parser.add_argument('--config', default='./configs/codebook.yml')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--no_cuda', type=list, default=['2'])
    parser.add_argument('--prefix', type=str, required=False, default='knn_pred_wavvq')
    parser.add_argument('--save_path', type=str, required=False, default="./Speech2GestureMatching/output/")
    parser.add_argument('--code_path', type=str, required=False)
    parser.add_argument('--VQVAE_model_path', type=str, required=False)
    parser.add_argument('--BEAT_path', type=str, default="../dataset/orig_BEAT/speakers/")
    parser.add_argument('--save_dir', type=str, default="../dataset/BEAT")
    parser.add_argument('--step', type=str, default="1")
    parser.add_argument('--stage', type=str, default="train")
    args = parser.parse_args()
    return args
