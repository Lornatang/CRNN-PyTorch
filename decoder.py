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
import numpy as np
import torch

__all__ = [
    "ctc_decode"
]


def _reconstruct(labels: list, blank: int = 0) -> list:
    new_labels = []

    # Merge same labels
    previous = None
    for label in labels:
        if label != previous:
            new_labels.append(label)
            previous = label

    # Delete blank
    new_labels = [label for label in new_labels if label != blank]

    return new_labels


def _greedy_decode(sequence_log_prob: np.ndarray, blank: int = 0):
    # sequence_log_prob: (batch, length, class)
    labels = np.argmax(sequence_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)

    return labels


def ctc_decode(log_probs: torch.Tensor, label2char: dict = None, blank=0):
    sequence_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))

    decoded_list = []
    for sequence_log_prob in sequence_log_probs:
        # Greedy algorithm to predict characters
        decoded = _greedy_decode(sequence_log_prob, blank=blank)

        # Decode the predicted character information into a whole string of information
        if label2char:
            decoded = [label2char[key] for key in decoded]
        decoded_list.append(decoded)
    return decoded_list
