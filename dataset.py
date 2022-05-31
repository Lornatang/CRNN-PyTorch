# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import imgproc

__all__ = [
    "train_valid_collate_fn", "test_collate_fn",
    "TrainValidImageDataset", "TestImageDataset"
]


def train_valid_collate_fn(batch: [torch.Tensor, torch.Tensor, torch.Tensor]) -> [torch.Tensor,
                                                                                  torch.Tensor,
                                                                                  torch.Tensor]:
    image_path, image_tensor, label_tensor, label_length_tensor = zip(*batch)
    image_tensor = torch.stack(image_tensor, 0)
    label_tensor = torch.cat(label_tensor, 0)
    label_length_tensor = torch.cat(label_length_tensor, 0)

    return image_tensor, label_tensor, label_length_tensor


def test_collate_fn(batch: [str, torch.Tensor, str]) -> [str, torch.Tensor, str]:
    image_path, image_tensor, image_label = zip(*batch)
    image_tensor = torch.stack(image_tensor, 0)

    return image_path, image_tensor, image_label


class TrainValidImageDataset(Dataset):
    def __init__(self,
                 dataroot: str,
                 annotation_file_name: str,
                 label_file_name: str,
                 labels_dict: dict,
                 image_width: int,
                 image_height: int,
                 mean: list,
                 std: list):
        self.dataroot = dataroot
        self.annotation_file_name = annotation_file_name
        self.label_file_name = label_file_name
        self.labels_dict = labels_dict
        self.image_width = image_width
        self.image_height = image_height
        self.mean = mean
        self.std = std

        self.image_paths, self.image_labels = self.load_image_label_from_file()

    def load_image_label_from_file(self):
        # Initialize the definition of image path, image text information, etc.
        lexicon_maps = {}
        image_paths = []
        image_labels = []

        # Read text labels and compose a label map
        with open(os.path.join(self.dataroot, self.label_file_name), "r") as f:
            for i, line in enumerate(f.readlines()):
                lexicon_maps[i] = line.strip()

        # Read image path and corresponding text information
        with open(os.path.join(self.dataroot, self.annotation_file_name), "r", encoding="UTF-8") as f:
            for line in f.readlines():
                image_path, lexicon_index = line.strip().split(" ")
                image_label = lexicon_maps[int(lexicon_index)]
                image_paths.append(os.path.join(self.dataroot, image_path))
                image_labels.append(image_label)

        return image_paths, image_labels

    def __getitem__(self, index: int) -> [str, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]

        # Read the image and convert it to grayscale
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Scale to the size of the image that the model can accept
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.image_height, self.image_width, 1))

        # Normalize and convert to Tensor format
        image_tensor = imgproc.image2tensor(image, mean=self.mean, std=self.std)

        # component file encoding
        label = self.image_labels[index]
        label = [self.labels_dict[character] for character in label]

        label_tensor = torch.LongTensor(label)
        label_length_tensor = torch.LongTensor([len(label)])

        return image_tensor, label_tensor, label_length_tensor

    def __len__(self):
        return len(self.image_paths)


class TestImageDataset(Dataset):
    def __init__(self,
                 dataroot: str,
                 annotation_file_name: str,
                 image_width: int,
                 image_height: int,
                 mean: list,
                 std: list):
        self.dataroot = dataroot
        self.annotation_file_name = annotation_file_name
        self.image_width = image_width
        self.image_height = image_height
        self.mean = mean
        self.std = std

        self.image_paths, self.image_labels = self.load_image_label_from_file()

    def load_image_label_from_file(self):
        # Initialize the definition of image path, image text information, etc.
        image_paths = []
        image_labels = []

        # Read image path and corresponding text information
        with open(os.path.join(self.dataroot, self.annotation_file_name), "r", encoding="UTF-8") as f:
            for line in f.readlines():
                image_path, image_label = line.strip().split(" ")
                image_paths.append(os.path.join(self.dataroot, image_path))
                image_labels.append(image_label)

        return image_paths, image_labels

    def __getitem__(self, index: int) -> [str, torch.Tensor, str]:
        image_path = self.image_paths[index]
        image_label = self.image_labels[index]

        # Read the image and convert it to grayscale
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Scale to the size of the image that the model can accept
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.image_height, self.image_width, 1))

        # Normalize and convert to Tensor format
        image_tensor = imgproc.image2tensor(image, mean=self.mean, std=self.std)

        return image_path, image_tensor, image_label

    def __len__(self):
        return len(self.image_paths)
