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

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import config
from dataset import ImageDataset, valid_test_collate_fn
from decoder import ctc_decode
from model import CRNN


def load_dataloader() -> DataLoader:
    # Load datasets
    datasets = ImageDataset(dataroot=config.dataroot,
                            annotation_file_name=config.annotation_file_name,
                            image_width=config.model_image_width,
                            image_height=config.model_image_height,
                            mean=config.mean,
                            std=config.std,
                            mode="test")

    dataloader = DataLoader(dataset=datasets,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=valid_test_collate_fn,
                            pin_memory=True,
                            drop_last=False,
                            persistent_workers=True)

    return dataloader


def build_model() -> nn.Module:
    # Initialize the model
    model = CRNN(config.model_num_classes)
    model = model.to(device=config.device)
    print("Build CRNN model successfully.")

    # Load the CRNN model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load CRNN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    if config.fp16:
        # Turn on half-precision inference.
        model.half()

    return model


def main() -> None:
    # Initialize correct predictions image number
    total_correct = 0

    # Initialize model
    model = build_model()

    # Load test dataLoader
    dataloader = load_dataloader()

    # Create a experiment folder results
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Get the number of test image files
    total_files = len(dataloader)

    with open(os.path.join(config.result_dir, config.result_file_name), "w") as f:
        with torch.no_grad():
            for batch_index, (image_path, images, labels) in enumerate(dataloader):
                # Transfer in-memory data to CUDA devices to speed up training
                images = images.to(device=config.device, non_blocking=True)

                if config.fp16:
                    # Convert to FP16
                    images = images.half()

                # Inference
                output = model(images)

                # record accuracy
                output_log_probs = F.log_softmax(output, 2)
                _, prediction_chars = ctc_decode(output_log_probs, config.chars_dict)

                if "".join(prediction_chars[0]) == labels[0].lower():
                    total_correct += 1

                if batch_index < total_files - 1:
                    information = f"`{os.path.basename(image_path[0])}` -> `{''.join(prediction_chars[0])}`"
                    print(information)
                else:
                    information = f"Acc: {total_correct / total_files * 100:.2f}%"
                    print(information)

                # Text information to be written to the file
                f.write(information + "\n")


if __name__ == "__main__":
    main()
