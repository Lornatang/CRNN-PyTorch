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

import numpy as np
import torch
from PIL import Image
from natsort import natsorted
from torch.nn import functional as F

import config
from decoder import ctc_decode
from model import CRNN


def main() -> None:
    # Initialize model
    model = CRNN(len(config.chars) + 1)
    model = model.to(device=config.device)
    print("Build CRNN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load CRNN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a experiment folder results
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize correct predictions image number
    total_correct = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.dataroot))
    # Get the number of test image files.
    total_files = len(file_names)

    with open(os.path.join(config.result_dir, config.result_name), "w") as f:
        for index in range(total_files):
            with torch.no_grad():
                image_path = os.path.join(config.dataroot, file_names[index])

                print(f"Processing `{os.path.abspath(image_path)}`...")
                # Read OCR image
                image = Image.open(image_path).convert("L")

                # Image convert Tensor
                image = image.resize((config.model_image_width, config.model_image_height), resample=Image.BILINEAR)
                image = np.array(image)
                image = image.reshape((1, 1, config.model_image_height, config.model_image_width))
                # Normalize [0, 255] to [-1, 1]
                image = (image / 127.5) - 1.0
                tensor = torch.Tensor(image)

                # Transfer in-memory data to CUDA devices to speed up training
                tensor = tensor.to(device=config.device)
                tensor = tensor.half()

                # predict characters
                output = model(tensor)

                # Compare the similarity between predicted and actual characters
                output_probs = F.log_softmax(output, 2)
                predictions = ctc_decode(output_probs, config.chars_dict)
                # target = target.cpu().numpy().tolist()
                # target_lengths = target_lengths.cpu().numpy().tolist()
                #
                # target_length_counter = 0
                # for prediction, target_length in zip(predictions, target_lengths):
                #     target = target[target_length_counter:target_length_counter + target_length]
                #     target_length_counter += target_length
                #     # If the text in the image is successfully predicted,
                #     # add 1 to the number of accurately predicted images
                #     if prediction == target:
                #         total_correct += 1

                if index < total_files - 1:
                    f.write(f"{image_path} -> {''.join(predictions[0])}\n")
                else:
                    f.write(f"Accuracy: {total_correct / total_files * 100:.2f}%")


if __name__ == "__main__":
    main()
