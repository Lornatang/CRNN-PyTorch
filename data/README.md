# Usage

## Step1: Download datasets

Contains ICDAR2013~2019, Synth90k, Synth800k, SynthText, Verisimilar Synthesis, UnrealText and more, etc.

- [Google Driver](https://drive.google.com/drive/folders/1dxrLQ48UodaLavqFHMimiYkuqtfufyrI?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1v4urOutexChkzhLYiOD0QA?pwd=llot)

## (Optional) Step2: Custom generated character image data

First, Go to the `<CRNN-PyTorch-main>/scirpts/TextDataGenerator` folder.

Then, Place the data in the following directory structure. run `auto_generator.py`.

```text
- TextDataGenerator
    - background_images  # Folder for background images
    - dicts  # Folder for label files
    - fonts  # Folder for all fonts
    - auto_generator.py  # Program files for automatic generation of character images and annotations
```

Finally, you will get a directory structure like this under this directory.

```text
- output
    - 00000  # Folder containing character images
    - 00001
    - ...
    - annotation.txt  # Text file with image path and annotation information
    - dict.txt  # Arrangement and combination file of custom characters
```

*Don't forget to modify `<CRNN-PyTorch-main>/config.py` file~~~*

## Step3: Prepare the dataset in the following format (e.g Synth90K)

```text
# Dataset struct
- Synth90k
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

## Step4: Check that the final dataset directory schema is completely correct

```text
# Train dataset
- Synth90k
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

# Test dataset
- IIIT5K
    - train
    - test
    - annotation.txt
    - annotation_train.txt
    - annotation_test.txt
    - images_list.txt
    - lexicon.txt
```
