"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
import gc
from argparse import ArgumentParser

import numpy as np
import timm
import torch
# import torchvirtual en
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import NUM_PTS, CROP_SIZE
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission
# from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_PATH = os.path.join("runs", "adam-rn50-32-7_best.pth")


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default="C:/data")
    # parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--batch-size", "-b", default=32, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM

    parser.add_argument("--epochs", "-e", default=20, type=int)


    parser.add_argument("--learning-rate", "-lr", default=5e-4, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()
#
# def train(model, loader, loss_fn, optimizer, device, scaler):
#     model.train()
#     train_loss = []
#     for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
#         images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
#         landmarks = batch["landmarks"].to(device)  # B x (2 * NUM_PTS)
#         with autocast():
#             pred_landmarks = model(images)  # B x (2 * NUM_PTS)
#             loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         train_loss.append(loss.item())
#         optimizer.zero_grad()
#
#     return np.mean(train_loss), train_loss


def train(model, loader, loss_fn, optimizer, device):

    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # del intermediate_variable1, intermediate_variable2, ...
        gc.collect()
        torch.cuda.empty_cache()

    return np.mean(train_loss), train_loss


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    os.makedirs("runs", exist_ok=True)
    print(torch.cuda.is_available())
    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        # TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])

    print("Reading data...")
    if args.epochs > 0:
        train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="train")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True,
                                      shuffle=True, drop_last=True)  # True False
        val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="val")
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True,
                                    shuffle=False, drop_last=False)

    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")

    print("Creating model...")
    # model = models.resnet18(pretrained=True)
    # model = models.resnet18(pretrained=True)
    model = timm.create_model('seresnext26d_32x4d', pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.fc.requires_grad_(True)

    model.load_state_dict(torch.load(MODEL_PATH))
    # model.eval()
    # model = torch.load(MODEL_PATH)
    # for param in model.parameters():
    #     param.requires_grad = True
    # model.eval()
    model.requires_grad_(True)
    #
    # model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    # model.fc.requires_grad_(True)

    model.to(device)

    # optim.lr_scheduler.ReduceLROnPlateau()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = fnn.mse_loss

    # 2. train & validate
    print("Ready for training...")
    # scaler = GradScaler()
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss, overall_train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device)
        # val_loss = validate(model, train_dataloader, loss_fn, device=device)
        val_loss = validate(model, val_dataloader, loss_fn, device=device)

        scheduler.step(val_loss)
        print("Epoch #{:2}:\ttrain loss: {:5.3}\tval loss: {:5.3}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), train_transforms, split="test")
    # test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="test")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False, drop_last=False)

    with open(os.path.join("runs", f"{args.name}_best.pth"), "rb") as fp:  # for current model
    # with open(MODEL_PATH, "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join("runs", f"{args.name}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)
    # train_predictions = predict(model, train_dataloader, device)
    # with open(os.path.join("runs", f"{args.name}_train_predictions.pkl"), "wb") as fp:
    #     pickle.dump({"image_names": train_dataset.image_names,
    #                  "landmarks": train_predictions,
    #                  "overall_loss": overall_train_loss},
    #                 fp)
    create_submission(args.data, test_predictions, os.path.join("runs", f"{args.name}_submit.csv"))


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
