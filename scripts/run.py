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

# Calculate the mean and variance of all dataset scripts
os.system("python ./calculate_image_mean_and_std.py "
          "--dataroot ../data/90kDICT32px/ "
          "--image_file_name annotation.txt "
          "--image_width 100 "
          "--image_height 32")

<<<<<<< Updated upstream
# # Calculate the mean and variance of a train dataset scripts
=======
# Calculate the mean and variance of a train dataset scripts
>>>>>>> Stashed changes
os.system("python ./calculate_image_mean_and_std.py "
          "--dataroot ../data/90kDICT32px/ "
          "--image_file_name annotation_train.txt "
          "--image_width 100 "
          "--image_height 32")

# Calculate the mean and variance of a valid dataset scripts
os.system("python ./calculate_image_mean_and_std.py "
          "--dataroot ../data/90kDICT32px/ "
          "--image_file_name annotation_valid.txt "
          "--image_width 100 "
          "--image_height 32")

# Calculate the mean and variance of a test dataset scripts
os.system("python ./calculate_image_mean_and_std.py "
          "--dataroot ../data/90kDICT32px/ "
          "--image_file_name annotation_test.txt "
          "--image_width 100 "
          "--image_height 32")
