# Usage

## Step1: Download datasets

Contains ICDAR2013~2019, MJSynth, SynthText, SynthAdd, Verisimilar Synthesis, UnrealText and more, etc.

- [Google Driver](https://drive.google.com/drive/folders/1dxrLQ48UodaLavqFHMimiYkuqtfufyrI?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1v4urOutexChkzhLYiOD0QA?pwd=llot)

## (Optional) Step2: Custom generated character image data

First, Go to the `<CRNN-PyTorch-main>/scirpts/TextDataGenerator` folder.

Then, Place the data in the following directory structure. run `auto_generator.py`.

```text
- TextDataGenerator
    - background_images  # Folder for background images
    - lexicons  # Folder for lexicon files
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
```

*Don't forget to modify `<CRNN-PyTorch-main>/config.py` file~~~*

## Step3: Prepare the dataset in the following format (e.g MJSynth)

```text
# Dataset struct
- MJSynth
    - 1
    - 2
    - 3
    ...
    - annotation_train.txt
    - annotation_valid.txt
```

## Step4: Check that the final dataset directory schema is completely correct

```text
# Train dataset
- MJSynth
    - 1
    - 2
    - 3
    ...
    - annotation_train.txt
    - annotation_valid.txt

# Test dataset
- SVT
    - images
    - annotation.txt
    - annotation_test.txt
```
