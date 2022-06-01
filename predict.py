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

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import config
import imgproc
from decoder import ctc_decode
from model import CRNN


def main(args):
    # Initialize the model
    model = CRNN(config.model_num_classes)
    model = model.to(device=config.device)
    print("Build CRNN model successfully.")

    # Load the CRNN model weights
    checkpoint = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load CRNN model weights `{args.weights_path}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    # Read the image and convert it to grayscale
    image = cv2.imread(args.image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Scale to the size of the image that the model can accept
    gray_image = cv2.resize(gray_image,
                            (config.model_image_width, config.model_image_height),
                            interpolation=cv2.INTER_CUBIC)
    gray_image = np.reshape(gray_image, (config.model_image_height, config.model_image_width, 1))

    # Normalize and convert to Tensor format
    gray_tensor = imgproc.image2tensor(gray_image, mean=config.mean, std=config.std).unsqueeze_(0)

    # Transfer in-memory data to CUDA devices to speed up training
    gray_tensor = gray_tensor.to(device=config.device, non_blocking=True)

    # Inference
    with torch.no_grad():
        output = model(gray_tensor)

        # record accuracy
        output_log_probs = F.log_softmax(output, 2)
        _, prediction_chars = ctc_decode(output_log_probs, config.chars_dict)

    print(f"`{args.image_path}` -> `{''.join(prediction_chars[0])}`\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRNN model predicts character content in images.")
    parser.add_argument("--image_path", type=str, help="Character image address to be tested.")
    parser.add_argument("--weights_path", type=str, help="Txt file containing the path to the dataset file.")
    args = parser.parse_args()

    main(args)
