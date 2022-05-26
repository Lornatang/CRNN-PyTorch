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
import torch
from torch import nn

__all__ = [
    "CRNN"
]


class _BidirectionalLSTM(nn.Module):

    def __init__(self, inputs_size: int, hidden_size: int, output_size: int):
        super(_BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(inputs_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.lstm(x)
        sequence_length, batch_size, inputs_size = recurrent.size()
        sequence_length2 = recurrent.view(sequence_length * batch_size, inputs_size)

        out = self.linear(sequence_length2)  # [sequence_length * batch_size, output_size]
        out = out.view(sequence_length, batch_size, -1)  # [sequence_length, batch_size, output_size]

        return out


class CRNN(nn.Module):

    def __init__(self, num_classes: int):
        super(CRNN, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # image size: 16 * 64

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # image size: 8 * 32

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # image size: 4 x 16

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # image size: 2 x 16

            nn.Conv2d(512, 512, (2, 2), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # image size: 1 x 16
        )

        self.recurrent_layers = nn.Sequential(
            _BidirectionalLSTM(512, 256, 256),
            _BidirectionalLSTM(256, 256, num_classes),
        )

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature sequence
        features = self.convolutional_layers(x)  # [b, c, h, w]
        features = features.squeeze(2)  # [b, c, w]
        features = features.permute(2, 0, 1)  # [w, b, c]

        # Deep bidirectional LSTM
        out = self.recurrent_layers(features)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
