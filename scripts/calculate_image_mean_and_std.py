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
import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def main(args):
    train_datasets = Synth90kDataset(dataroot=args.dataroot,
                                     annotation_file_name=args.annotation_file_name,
                                     image_width=args.image_width,
                                     image_height=args.image_height)

    train_dataloader = DataLoader(train_datasets, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    mean_value = torch.zeros(1)
    std_value = torch.zeros(1)

    image_number = len(train_dataloader)

    for image_index, image in enumerate(train_dataloader):
        print(f"Process [{image_index + 1:08d}/{image_number:08d}] images.")
        mean_value += image[:, :, :].mean()
        std_value += image[:, :, :].std()

    mean_value.div_(len(train_dataloader))
    std_value.div_(len(train_dataloader))

    mean_value = list(mean_value.numpy())
    std_value = list(std_value.numpy())

    print(args.annotation_file_name)
    print(f"Mean = {mean_value[0]:.4f}")
    print(f"Std  = {std_value[0]:.4f}")


def load_image_label_from_file(dataroot: str, image_file_name: str):
    # Initialize the definition of image path etc.
    image_paths = []

    # Read image path and corresponding text information
    with open(os.path.join(dataroot, image_file_name), "r") as f:
        for line in f.readlines():
            path, _ = line.strip().split(" ")
            image_paths.append(os.path.join(dataroot, path))

    return image_paths


class Synth90kDataset(Dataset):
    def __init__(self,
                 dataroot: str,
                 annotation_file_name: str,
                 image_width: int,
                 image_height: int):
        self.image_width = image_width
        self.image_height = image_height

        self.image_paths = load_image_label_from_file(dataroot, annotation_file_name)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]

        # Read the image and convert it to grayscale
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Scale to the size of the image that the model can accept
        gray_image = cv2.resize(gray_image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        gray_image = np.reshape(gray_image, (self.image_height, self.image_width, 1))

        # Normalize and convert to Tensor format
        gray_image = gray_image.astype(np.float32)
        gray_image /= 255.
        gray_image = gray_image.transpose([2, 0, 1])
        gray_tensor = torch.FloatTensor(gray_image)

        return gray_tensor

    def __len__(self):
        return len(self.image_paths)


def synth90k_collate_fn(batch: [torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)

    return images, targets, target_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the mean and variance of a dataset.")
    parser.add_argument("--dataroot", type=str, help="Dataset root directory path.")
    parser.add_argument("--annotation_file_name", type=str, help="Txt file containing the path to the dataset file.")
    parser.add_argument("--image_width", type=int, help="The width of each image in the dataset.")
    parser.add_argument("--image_height", type=int, help="The height of each image in the dataset.")
    args = parser.parse_args()

    main(args)
