# Process

This directory contains the code to process the data.

## Details

* beat_data_to_lmdb: Process the BEAT dataset into lmdb format to generate a cache and reduce data read time.
* bvh_to_position: Generate the joint tree corresponding to the bvh file and the 3D position coordinates of the particular joint.
* lmdb_to_noduplication: Generate matrix without overlapping frames from lmdb file
* make_beat_dataset: Process BEAT dataset, handle raw action files, speech, text, MFCC, rotation matrix, rhyme features, etc.
* merge_figs: Use opencv to put together multiple images into a video at a specified fps.
* process_beat_txt: Manually process BEAT's text to force alignment based on speech.
* process_bvh: Convert bvh files to euler, rotation matrix.
* speech_feat: Various methods to extract energy, pitch and volume of speech based on pyworld and python_speech_features.
* trinity_data_to_lmdb: Process the Trinity dataset into lmdb format to generate a cache and reduce data read time.
* visualize_bvh: Based on the specified specific joints and connection relationships, the bvh file is converted to 3D position coordinates and visualized using matplotlib.animation.

## 说明

* beat_data_to_lmdb: 将BEAT数据集处理成lmdb格式，生成缓存，减少数据读取时间。
* bvh_to_position: 生成bvh文件对应的关节树和特定关节的3D position坐标。
* lmdb_to_noduplication: 由lmdb文件生成没有重叠帧的矩阵
* make_beat_dataset: 处理BEAT数据集，处理原始动作文件、语音、文本、MFCC、旋转矩阵、韵律特征等。
* merge_figs: 利用opencv将多张图片拼成指定fps的视频。
* process_beat_txt: 手动处理BEAT的文本，根据语音进行强制对齐。
* process_bvh: 将bvh文件转为euler、rotation matrix。
* speech_feat: 基于pyworld和python_speech_features的多种提取语音的energy、pitch和volume的方法。
* trinity_data_to_lmdb: 将Trinity数据集处理成lmdb格式，生成缓存，减少数据读取时间。
* visualize_bvh: 根据指定的特定关节和连接关系，将bvh文件转为3D position坐标，利用matplotlib.animation进行可视化。


[//]: # (# Example:)
[//]: # (```python)
[//]: # ()
[//]: # (```        )
