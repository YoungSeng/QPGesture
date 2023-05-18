import numpy as np
import cv2


def merge_figs(input_path, output_path, cnt=20, fps=20):
    print('merge figs...')
    size = (640,480)#这个是图片的尺寸，一定要和要用的图片size一致
    #完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videowrite = cv2.VideoWriter(output_path,fourcc,fps,size)#20是帧数，size是图片尺寸

    for filename in [input_path.format(i) for i in range(cnt)]:#这个循环是为了读取所有要用的图片文件
        print(filename + '\r', end='')
        img = cv2.imread(filename)
        videowrite.write(img)


if __name__ == '__main__':
    '''
    cd process/
    python merge_figs.py
    '''
    input_path = "/mnt/nfs7/y50021900/My/codebook/BEAT_output_60fps_position/train_PAE_121_1/figs/{0}.jpg"
    output_path = "/mnt/nfs7/y50021900/My/codebook/BEAT_output_60fps_position/train_PAE_121_1/merged_figs.mp4"
    cnt = 5260
    merge_figs(input_path, output_path, cnt=cnt, fps=30)
