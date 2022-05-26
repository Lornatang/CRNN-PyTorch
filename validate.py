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
from dataset import Synth90kDataset, synth90k_collate_fn
from decoder import ctc_decode
from model import CRNN


def load_dataloader() -> DataLoader:
    # Load datasets
    datasets = Synth90kDataset(dataroot=config.dataroot,
                               annotation_file_name=config.annotation_file_name,
                               label_file_name=config.label_file_name,
                               labels_dict=config.labels_dict,
                               image_width=config.model_image_width,
                               image_height=config.model_image_height,
                               mean=config.all_mean,
                               std=config.all_std)

    dataloader = DataLoader(dataset=datasets,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            collate_fn=synth90k_collate_fn,
                            pin_memory=True,
                            drop_last=False,
                            persistent_workers=True)

    return dataloader


def build_model() -> nn.Module:
    # Initialize the model
    model = CRNN(config.model_num_classes)
    model = model.to(device=config.device)
    print("Build CRNN model successfully.")

    # Load the super-resolution model weights
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
    if not os.path.exists(config.validate_result_dir):
        os.makedirs(config.validate_result_dir)

    # Get the number of test image files
    total_files = len(dataloader)

    with open(os.path.join(config.validate_result_dir, config.validate_result_file_name), "w") as f:
        with torch.no_grad():
            for batch_index, (image_path, images, labels, labels_length) in enumerate(dataloader):
                # Transfer in-memory data to CUDA devices to speed up training
                images = images.to(device=config.device, non_blocking=True)
                labels = labels.to(device=config.device, non_blocking=True)
                labels_length = labels_length.to(device=config.device, non_blocking=True)

                if config.fp16:
                    # Convert to FP16
                    images = images.half()

                # Inference
                output = model(images)

                # record accuracy
                output_probs = F.log_softmax(output, 2)
                prediction_labels, prediction_chars = ctc_decode(output_probs, config.chars_dict)
                labels = labels.cpu().numpy().tolist()
                labels_length = labels_length.cpu().numpy().tolist()

                labels_length_counter = 0
                for prediction_label, label_length in zip(prediction_labels, labels_length):
                    label = labels[labels_length_counter:labels_length_counter + label_length]
                    labels_length_counter += label_length
                    if prediction_label == label:
                        total_correct += 1

                if batch_index < total_files - 1:
                    information = f"{image_path[0]} -> {''.join(prediction_chars[0])}"
                    print(information)
                else:
                    information = f"Acc: {total_correct / total_files * 100:.2f}%"
                    print(information)

                # Text information to be written to the file
                f.write(information + "\n")


if __name__ == "__main__":
    main()
