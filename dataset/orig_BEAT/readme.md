In response to this [issue](https://github.com/YoungSeng/QPGesture/issues/7)

### 1. Get the original BEAT dataset

First download 

* `1_wayne_0_1_8.TextGrid`, `1_wayne_0_1_8.wav`, `1_wayne_0_1_8.bvh` (as trn)
* `1_wayne_0_103_110.TextGrid`, `1_wayne_0_103_110.wav`, `1_wayne_0_103_110.bvh` (as val)
* `1_wayne_0_111_118.TextGrid`, `1_wayne_0_111_118.wav`, `1_wayne_0_111_118.bvh` (as tst)

from [BEAT](https://pantomatrix.github.io/BEAT/). Let's use these nine files of speaker `1` as an example.
And put them in `./dataset/orig_BEAT/speakers/1/`.

Then run:
```
pip install -U numpy
cd ./process/
python make_beat_dataset.py --BEAT_path "../dataset/orig_BEAT/speakers/" --save_dir "../dataset/BEAT" --prefix "speaker_1_state_0" --step 1
```

You will get:
```
number of training set: 3
processing audio...
Processing gesture...
check: 1_wayne_0_111_118.bvh
check: 1_wayne_0_103_110.bvh
check: 1_wayne_0_1_8.bvh
```

Run:
```
cd ../codebook/Speech2GestureMatching/
python normalize_audio.py
python mfcc.py
cd ../../process/
python make_beat_dataset.py --BEAT_path "../dataset/orig_BEAT/speakers/" --save_dir "../dataset/BEAT" --prefix "speaker_1_state_0" --step 2
```

Then modify these `speaker_10_state_0` to `speaker_1_state_0`:

https://github.com/YoungSeng/QPGesture/blob/b9d0b7fc9b0ff6d45a817cc3ed627378427fa785/codebook/PAE.py#L554-L555

And run:
```
cd ../codebook/
python PAE.py --config=./configs/codebook.yml --gpu 0 --stage inference
```

You will get phase files such as `"./dataset/BEAT/speaker_1_state_0/Phase/1_wayne_0_1_8.npz"`, the shape is `(33778, 4)` instead of `(33778, 4, 1, 8, 1)`.

Then run:
```
cd ../process/
python make_beat_dataset.py --config "../codebook/configs/codebook.yml" --BEAT_path "../dataset/orig_BEAT/speakers/" --save_dir "../dataset/BEAT" --prefix "speaker_1_state_0" --gpu 0 --step 3
python make_beat_dataset.py --config "../codebook/configs/codebook.yml" --BEAT_path "../dataset/orig_BEAT/speakers/" --save_dir "../dataset/BEAT" --prefix "speaker_1_state_0" --gpu 0 --step 4
```

Then you will get:
```
(140, 64000)
(140, 398, 2)
(109, 64000)
(109, 398, 2)
(106, 64000)
(106, 398, 2)
1 1 1
1_wayne_0_1_8
(33778, 135) (33778, 4)
1_wayne_0_111_118
(26364, 135) (26364, 4)
1_wayne_0_103_110
(25514, 135) (25514, 4)
```

To quickly start using these databases with speaker `1`:

```
cd ../codebook/Speech2GestureMatching/
bash GestureKNN_speaker1_issue.sh
```

