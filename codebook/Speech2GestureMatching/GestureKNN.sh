
test_wavvq=${1:-'../../data/Example1/ZeroEGGS_cut/wavvq_240.npz'}
max_frames=${2:-0}
out_knn_filename=${3:-'./output/result.npz'}
echo $test_wavvq $max_frames $out_knn_filename

python GestureKNN.py \
    --train_database=../../data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_txt_2.npz \
    --test_data=../../data/BEAT/speaker_10_state_0/speaker_10_state_0_test_240_txt_2.npz \
    --out_knn_filename=$out_knn_filename \
    --out_video_path=./output/output_video_folder/ \
    --train_codebook=../../data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_code.npz \
    --codebook_signature=../../data/BEAT/BEAT_output_60fps_rotation/code.npz \
    --train_wavlm=../../data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_WavLM.npz \
    --test_wavlm=../../data/BEAT/speaker_10_state_0/speaker_10_state_0_test_240_WavLM.npz  \
    --train_wavvq=../../data/BEAT/speaker_10_state_0/speaker_10_state_0_train_240_WavVQ.npz \
    --test_wavvq=$test_wavvq\
    --max_frames=$max_frames
