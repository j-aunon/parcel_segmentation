import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex, F1Score, Accuracy
from torch.utils.tensorboard import SummaryWriter

# --- CONFIG ---
DATA_DIR   = "data/dataset"
OUT_DIR    = "models"
EPOCHS     = 50
BATCH_SIZE = 8
LR         = 1e-4
IMG_SIZE   = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def load_classes(path):
    classes = {}
    with open(path) as f:
        for line in f:
            cid, code = line.strip().split(": ")
            classes[int(cid)] = code
    return classes


class SegDataset(Dataset):
    def __init__(self, split):
        self.img_dir  = Path(DATA_DIR) / "images" / split
        self.mask_dir = Path(DATA_DIR) / "masks"  / split
        self.files    = sorted(self.img_dir.glob("*.png"))
        self.img_tf   = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i].name
        img  = Image.open(self.files[i]).convert("RGB")
        mask = Image.open(self.mask_dir / name).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        return self.img_tf(img), torch.from_numpy(np.array(mask)).long()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, masks in loader:
        preds = model(imgs.to(DEVICE)).argmax(1).cpu()
        all_preds.append(preds.flatten())
        all_targets.append(masks.flatten())
    return torch.cat(all_preds), torch.cat(all_targets)


def compute_metrics(preds, targets, n_classes, classes):
    kwargs = {"task": "multiclass", "num_classes": n_classes}
    oa           = Accuracy(**kwargs)(preds, targets)
    iou_macro    = JaccardIndex(**kwargs, average="macro")(preds, targets)
    iou_weighted = JaccardIndex(**kwargs, average="weighted")(preds, targets)
    f1_per_class = F1Score(**kwargs, average=None)(preds, targets)

    lines = [
        f"Overall Accuracy (OA) : {oa:.4f}",
        f"mIoU macro            : {iou_macro:.4f}",
        f"mIoU weighted         : {iou_weighted:.4f}",
        "",
        "F1 per class:",
    ]
    for cls_id, f1 in enumerate(f1_per_class):
        name = classes.get(cls_id, str(cls_id))
        lines.append(f"  {cls_id:2d}  {name:<12} {f1:.4f}")
    return "\n".join(lines), {"oa": oa, "iou_macro": iou_macro, "iou_weighted": iou_weighted, "f1": f1_per_class}


if __name__ == "__main__":
    classes   = load_classes(os.path.join(DATA_DIR, "classes.txt"))
    n_classes = len(classes)
    print(f"Classes: {n_classes}  |  Device: {DEVICE}")

    train_loader = DataLoader(SegDataset("train"), batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(SegDataset("val"),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(SegDataset("test"),  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(OUT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "tb_logs"))
    best_iou, best_epoch = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        preds, targets = evaluate(model, val_loader)
        val_iou = JaccardIndex(task="multiclass", num_classes=n_classes, average="macro")(preds, targets)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  val_mIoU={val_iou:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("mIoU/val",   val_iou,    epoch)
        writer.add_scalar("LR",         scheduler.get_last_lr()[0], epoch)

        if val_iou > best_iou:
            best_iou, best_epoch = float(val_iou), epoch
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))

    # Test evaluation with best model
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth")))
    preds, targets = evaluate(model, test_loader)
    metrics_txt, metrics = compute_metrics(preds, targets, n_classes, classes)

    writer.add_scalar("mIoU_macro/test",    metrics["iou_macro"],    0)
    writer.add_scalar("mIoU_weighted/test", metrics["iou_weighted"], 0)
    writer.add_scalar("OA/test",            metrics["oa"],           0)
    for cls_id, f1 in enumerate(metrics["f1"]):
        writer.add_scalar(f"F1/{classes.get(cls_id, cls_id)}", f1, 0)
    writer.close()

    results = f"Best epoch: {best_epoch}  val_mIoU: {best_iou:.4f}\n\n{metrics_txt}"
    print(f"\n{results}")

    with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
        f.write(results)

    print(f"\nModel   → {OUT_DIR}/best_model.pth")
    print(f"Results → {OUT_DIR}/results.txt")
    print(f"TBoard  → tensorboard --logdir {OUT_DIR}/tb_logs")
