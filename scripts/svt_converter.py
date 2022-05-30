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
"""The script implementation reference `https://github.com/open-mmlab/mmocr/blob/main/tools/data/textrecog/svt_converter.py`"""
import argparse
import os
import xml.etree.ElementTree as ET
import shutil
import cv2


def main(args):
    # Create the destination output folder if it doesn't exist
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    tree = ET.parse(args.inputs_annotation_file_path)
    root = tree.getroot()

    # Number of indexed pictures
    i = 1

    # The number of images to be intercepted by index
    index = 1

    total_files = len(root)

    with open(args.output_annotation_file_path, "w", encoding="UTF-8") as f:
        for image_node in root.findall("image"):
            image_name = os.path.basename(image_node.find("imageName").text)
            print(f"[{i}/{total_files}] Process image: `{image_name}`")
            i += 1

            # Read the original image, ready to intercept the specified area from the annotation file
            image = cv2.imread(os.path.join(args.inputs_dir, image_name))
            for rectangle in image_node.find("taggedRectangles"):
                x = int(rectangle.get("x"))
                y = int(rectangle.get("y"))
                w = int(rectangle.get("width"))
                h = int(rectangle.get("height"))

                start_ordinate = max(0, y)
                end_ordinate = max(0, y + h)
                start_abscissa = max(0, x)
                end_abscissa = max(0, x + w)

                # Crop the image of the area marked in the label from the original image
                crop_image = image[start_ordinate:end_ordinate, start_abscissa:end_abscissa]

                # Scale image to specified size
                if args.resize:
                    crop_image = cv2.resize(crop_image,
                                            (args.image_width, args.image_height),
                                            interpolation=cv2.INTER_CUBIC)

                # Save the cropped image to the specified location
                crop_image_name = f"{index:05d}.jpg"
                cv2.imwrite(os.path.join(args.output_dir, crop_image_name), crop_image)

                # Write the annotation content to the specified file
                text = f"./{os.path.basename(args.output_dir)}/{crop_image_name} {rectangle.find('tag').text.lower()}"
                # No line breaks are required at the end of the data
                if index != total_files - 1:
                    text += "\n"
                f.write(text)

                index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert SVT dataset to text recognition processing dataset format")
    parser.add_argument("--inputs_dir", help="Image folder in the SVT dataset")
    parser.add_argument("--output_dir", help="Folder to save pictures from text boxes in SVT")
    parser.add_argument("--inputs_annotation_file_path", help="Annotation file address in SVT dataset")
    parser.add_argument("--output_annotation_file_path", help="Save the address of the converted file in the SVT dataset")
    parser.add_argument("--resize", action="store_true", help="Ensure that the image scale of the SVT test dataset meets the model requirements")
    parser.add_argument("--image_width", type=int, help="Image width requirements in the dataset")
    parser.add_argument("--image_height", type=int, help="Image height requirements in the dataset")
    args = parser.parse_args()

    main(args)
