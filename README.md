# CRNN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)
.

## Table of contents

- [CRNN-PyTorch](#crnn-pytorch)
  - [Overview](#overview)
  - [Table of contents](#table-of-contents)
  - [Download weights](#download-weights)
  - [Download datasets](#download-datasets)
  - [How Test and Train](#how-test-and-train)
    - [Test](#test)
    - [Train CRNN model](#train-crnn-model)
    - [Resume train CRNN model](#resume-train-crnn-model)
  - [Result](#result)
  - [Contributing](#contributing)
  - [Credit](#credit)
    - [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](#an-end-to-end-trainable-neural-network-for-image-based-sequence-recognition-and-its-application-to-scene-text-recognition)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains ICDAR2013~2019, MJSynth, SynthText, SynthAdd, Verisimilar Synthesis, UnrealText and more, etc.

- [Google Driver](https://drive.google.com/drive/folders/1CwkA0gKd4bnj66W0l6CB14sx-aAe3WOE?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1v31aBT5phe5Ci6N0Wsn3xQ?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 40: `mode` change to `test`.
- line 79: `model_path` change to `results/pretrained_models/CRNN-MJSynth-e9341ede.pth.tar`.

### Train CRNN model

- line 40: `mode` change to `train`.
- line 42: `exp_name` change to `CRNN_MJSynth`.

### Resume train CRNN model

- line 40: `mode` change to `train`.
- line 42: `exp_name` change to `CRNN_MJSynth`.
- line 56: `resume` change to `samples/CRNN_MJSynth/epoch_xxx.pth.tar`.

## Result

Source of original paper results: [https://arxiv.org/pdf/1507.05717.pdf](https://arxiv.org/pdf/1507.05717.pdf)

In the following table, `-` indicates show no test.

|    Model    | IIIT5K(None) | SVT(None) | IC03(None) | IC13(None) |
|:-----------:|:------------:|:---------:|:----------:|:----------:|
| CRNN(paper) |     78.2     |   80.8    |    89.4    |    86.7    |
| CRNN(repo)  |   **81.5**   | **80.1**  |   **-**    |   **-**    |

```bash
# Download `CRNN-Synth90k-e9341ede.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python predict.py --image_path ./figures/Available.png --weights_path ./results/pretrained_models/CRNN-MJSynth-e9341ede.pth.tar
```

Input: <span align="center"><img src="figures/Available.png"/></span>

Output:

```text
Build CRNN model successfully.
Load CRNN model weights `./results/pretrained_models/CRNN-MJSynth-e9341ede.pth.tar` successfully.
``./figures/Available.png` -> `available`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

_Baoguang Shi, Xiang Bai, Cong Yao_ <br>

**Abstract** <br>
Image-based sequence recognition has been a long-standing research topic in computer vision. In this paper, we
investigate the problem of scene text recognition, which is among the most important and challenging tasks in
image-based sequence recognition. A novel neural network architecture, which integrates feature extraction, sequence
modeling and transcription into a unified framework, is proposed. Compared with previous systems for scene text
recognition, the proposed architecture possesses four distinctive properties: (1) It is end-to-end trainable, in
contrast to most of the existing algorithms whose components are separately trained and tuned. (2) It naturally handles
sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization. (3) It is not
confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene
text recognition tasks. (4) It generates an effective yet much smaller model, which is more practical for real-world
application scenarios. The experiments on standard benchmarks, including the IIIT-5K, Street View Text and ICDAR
datasets, demonstrate the superiority of the proposed algorithm over the prior arts. Moreover, the proposed algorithm
performs well in the task of image-based music score recognition, which evidently verifies the generality of it.

[[Paper]](https://arxiv.org/pdf/1507.05717) [[Code(Lua)]](https://github.com/bgshih/crnn)

```bibtex
@article{ShiBY17,
  author    = {Baoguang Shi and
               Xiang Bai and
               Cong Yao},
  title     = {An End-to-End Trainable Neural Network for Image-Based Sequence Recognition
               and Its Application to Scene Text Recognition},
  journal   = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume    = {39},
  number    = {11},
  pages     = {2298--2304},
  year      = {2017}
}
```

