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
import shutil
import time
from enum import Enum

import torch
import torch.optim as optim
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import Synth90kDataset, synth90k_collate_fn
from decoder import ctc_decode
from model import CRNN


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc = 0.0

    train_dataloader, valid_dataloader = load_dataset()
    print("Load all datasets successfully.")

    model = build_model()
    print("Build CRNN model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, config.epochs):
        train(model, train_dataloader, criterion, optimizer, epoch, scaler, writer)
        acc = validate(model, valid_dataloader, epoch, writer, "Valid")
        print("\n")

        # Automatically save the model with the highest index
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        torch.save({"epoch": epoch + 1,
                    "best_acc": best_acc,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "last.pth.tar"))


def load_dataset() -> [DataLoader, DataLoader]:
    # Load train, test and valid datasets
    train_datasets = Synth90kDataset(dataroot=config.dataroot,
                                     annotation_file_name=config.annotation_train_file_name,
                                     label_file_name=config.label_file_name,
                                     labels_dict=config.labels_dict,
                                     image_width=config.model_image_width,
                                     image_height=config.model_image_height,
                                     mean=config.all_mean,
                                     std=config.all_std)
    valid_datasets = Synth90kDataset(dataroot=config.dataroot,
                                     annotation_file_name=config.annotation_valid_file_name,
                                     label_file_name=config.label_file_name,
                                     labels_dict=config.labels_dict,
                                     image_width=config.model_image_width,
                                     image_height=config.model_image_height,
                                     mean=config.all_mean,
                                     std=config.all_std)

    # Generator all dataloader
    train_dataloader = DataLoader(dataset=train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  collate_fn=synth90k_collate_fn,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)

    valid_dataloader = DataLoader(dataset=valid_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  collate_fn=synth90k_collate_fn,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    return train_dataloader, valid_dataloader


def build_model() -> nn.Module:
    model = CRNN(config.model_num_classes)
    model = model.to(device=config.device)

    return model


def define_loss() -> nn.CTCLoss:
    criterion = nn.CTCLoss(reduction="sum")
    criterion = criterion.to(device=config.device)

    return criterion


def define_optimizer(model) -> optim.RMSprop:
    optimizer = optim.RMSprop(model.parameters(), config.model_lr)

    return optimizer


def train(model: nn.Module,
          train_dataloader: DataLoader,
          criterion: nn.CTCLoss,
          optimizer: optim.RMSprop,
          epoch: int,
          scaler: amp.GradScaler,
          writer: SummaryWriter) -> None:
    """Training main program

    Args:
        model (nn.Module): CRNN model
        train_dataloader (DataLoader): training dataset iterator
        criterion (nn.CTCLoss): Calculates loss between a continuous (unsegmented) time series and a target sequence
        optimizer (optim.RMSprop): optimizer for optimizing generator models in generative networks
        epoch (int): number of training epochs during training the generative network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(train_dataloader)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Get the initialization training time
    end = time.time()

    for batch_index, (_, images, labels, labels_length) in enumerate(train_dataloader):
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Get the number of data in the current batch
        curren_batch_size = images.size(0)

        # Transfer in-memory data to CUDA devices to speed up training
        images = images.to(device=config.device, non_blocking=True)
        labels = labels.to(device=config.device, non_blocking=True)
        labels_length = labels_length.to(device=config.device, non_blocking=True)

        # Initialize generator gradients
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = model(images)

            output_probs = F.log_softmax(output, 2)
            images_lengths = torch.LongTensor([output.size(0)] * curren_batch_size)
            labels_length = torch.flatten(labels_length)

            # Computational loss
            loss = criterion(output_probs, labels, images_lengths, labels_length) / curren_batch_size

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), curren_batch_size)

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)


def validate(model: nn.Module,
             dataloader: DataLoader,
             epoch: int,
             writer: SummaryWriter,
             mode: str) -> [float, float]:
    """Test main program

    Args:
        model (nn.Module): CRNN model
        dataloader (DataLoader): Test dataset iterator
        epoch (int): Number of test epochs during training of the adversarial network
        writer (SummaryWriter): Log file management function
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize correct predictions image number
    total_correct = 0
    total_files = 0

    with torch.no_grad():
        for batch_index, (_, images, labels, labels_length) in enumerate(dataloader):
            # Get how many data the current batch has and increase the total number of tests
            total_files += images.size(0)

            # Transfer in-memory data to CUDA devices to speed up training
            images = images.to(device=config.device, non_blocking=True)
            labels = labels.to(device=config.device, non_blocking=True)
            labels_length = labels_length.to(device=config.device, non_blocking=True)

            # Mixed precision testing
            with amp.autocast():
                output = model(images)

            # record accuracy
            output_probs = F.log_softmax(output, 2)
            prediction_labels, _ = ctc_decode(output_probs, config.chars_dict)
            labels = labels.cpu().numpy().tolist()
            labels_length = labels_length.cpu().numpy().tolist()

            labels_length_counter = 0
            for prediction_label, label_length in zip(prediction_labels, labels_length):
                label = labels[labels_length_counter:labels_length_counter + label_length]
                labels_length_counter += label_length
                if prediction_label == label:
                    total_correct += 1

    # print metrics
    acc = (total_correct / total_files) * 100
    print(f"* Acc: {acc:.2f}%")

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/Acc", acc, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return acc


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
