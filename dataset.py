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
    "collate_fn",
    "TrainValidImageDataset", "TestImageDataset"
]


def collate_fn(batch: [str, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    image_path, gray_tensor, labels_tensor, labels_length_tensor = zip(*batch)
    gray_tensor = torch.stack(gray_tensor, 0)
    labels_tensor = torch.cat(labels_tensor, 0)
    labels_length_tensor = torch.cat(labels_length_tensor, 0)

    return image_path, gray_tensor, labels_tensor, labels_length_tensor


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

        self.image_paths, self.image_texts = self.load_image_label_from_file()

    def load_image_label_from_file(self):
        # Initialize the definition of image path, image text information, etc.
        labels_map = {}
        image_paths = []
        image_texts = []

        # Read text labels and compose a label map
        with open(os.path.join(self.dataroot, self.label_file_name), "r") as f:
            for i, line in enumerate(f.readlines()):
                labels_map[i] = line.strip()

        # Read image path and corresponding text information
        with open(os.path.join(self.dataroot, self.annotation_file_name), "r", encoding="UTF-8") as f:
            for line in f.readlines():
                path, index = line.strip().split(" ")
                text = labels_map[int(index)]
                image_paths.append(os.path.join(self.dataroot, path))
                image_texts.append(text)

        return image_paths, image_texts

    def __getitem__(self, index: int) -> [str, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]

        # Read the image and convert it to grayscale
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Scale to the size of the image that the model can accept
        gray_image = cv2.resize(gray_image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        gray_image = np.reshape(gray_image, (self.image_height, self.image_width, 1))

        # Normalize and convert to Tensor format
        gray_tensor = imgproc.image2tensor(gray_image, mean=self.mean, std=self.std)

        # component file encoding
        text = self.image_texts[index]
        labels = [self.labels_dict[c] for c in text]
        labels_length = [len(labels)]
        labels_tensor = torch.LongTensor(labels)
        labels_length_tensor = torch.LongTensor(labels_length)

        return image_path, gray_tensor, labels_tensor, labels_length_tensor

    def __len__(self):
        return len(self.image_paths)


class TestImageDataset(Dataset):
    def __init__(self,
                 dataroot: str,
                 annotation_file_name: str,
                 labels_dict: dict,
                 image_width: int,
                 image_height: int,
                 mean: list,
                 std: list):
        self.dataroot = dataroot
        self.annotation_file_name = annotation_file_name
        self.labels_dict = labels_dict
        self.image_width = image_width
        self.image_height = image_height
        self.mean = mean
        self.std = std

        self.image_paths, self.image_texts = self.load_image_label_from_file()

    def load_image_label_from_file(self):
        # Initialize the definition of image path, image text information, etc.
        image_paths = []
        image_texts = []

        # Read image path and corresponding text information
        with open(os.path.join(self.dataroot, self.annotation_file_name), "r", encoding="UTF-8") as f:
            for line in f.readlines():
                path, text = line.strip().split(" ")
                image_paths.append(os.path.join(self.dataroot, path))
                image_texts.append(text.lower())

        return image_paths, image_texts

    def __getitem__(self, index: int) -> [str, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]

        # Read the image and convert it to grayscale
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Scale to the size of the image that the model can accept
        gray_image = cv2.resize(gray_image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        gray_image = np.reshape(gray_image, (self.image_height, self.image_width, 1))

        # Normalize and convert to Tensor format
        gray_tensor = imgproc.image2tensor(gray_image, mean=self.mean, std=self.std)

        # component file encoding
        text = self.image_texts[index]
        labels = [self.labels_dict[c] for c in text]
        labels_length = [len(labels)]
        labels_tensor = torch.LongTensor(labels)
        labels_length_tensor = torch.LongTensor(labels_length)

        return image_path, gray_tensor, labels_tensor, labels_length_tensor

    def __len__(self):
        return len(self.image_paths)
