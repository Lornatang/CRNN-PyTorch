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
import cv2
import numpy as np
import torch

__all__ = [
    "image2tensor"
]


def image2tensor(image: np.ndarray,
                 range_norm: bool = True,
                 mean: float = 0.5,
                 std: float = 0.5,
                 half: bool = False) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]. Default: ``True``
        mean (float): Image mean. Default: 0.5
        std (float): Image std. Default: 0.5
        half (bool): Whether to convert torch.float32 similarly to torch.half type. Default: ``True``

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image2tensor(example_image)

    """
    # Scale to [0,1]
    image = image.astype(np.float32)
    image /= 255.

    # HWC convert to CHW
    image = image.transpose([2, 0, 1])

    # Convert image data type to Tensor data type
    tensor = torch.FloatTensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.sub_(mean).div_(std)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor
