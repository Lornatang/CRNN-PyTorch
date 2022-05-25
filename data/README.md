# Usage

## Step1: Download datasets

Contains ICDAR2013~2019, Synth90k, Synth800k, SynthText, Verisimilar Synthesis, UnrealText and more, etc.

- [Google Driver](https://drive.google.com/drive/folders/1dxrLQ48UodaLavqFHMimiYkuqtfufyrI?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1v4urOutexChkzhLYiOD0QA?pwd=llot)

## Step2: Prepare the dataset in the following format

```text
# Dataset struct
- 90kDICT32px
    - 1
    - 2
    - 3
    ...
    - annotation.txt
    - annotation_train.txt
    - annotation_valid.txt
    - annotation_test.txt
    - images_list.txt
    - lexicon.txt
```

## Special attention

Mean and std in the dataset.
*The following numerical calculations are derived from the `scripts/run.py` file*

### Synth90k

```text
# all dataset
mean = 0.4639
std = 0.1422

# train dataset
mean = 0.4639
std = 0.4620

# valid dataset
mean = 0.4638
std  = 0.1422

# test dataset
mean = 0.4638
std  = 0.1423
```
