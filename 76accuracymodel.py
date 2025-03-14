import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm
import timm  # for state-of-the-art vision models

# -----------------------------
# 1. Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------------
# 2. Configuration
# -----------------------------
BATCH_SIZE = 8
EPOCHS = 15              # 15 epochs total
LR = 1e-4
IMG_SIZE = 224

TRAIN_DIR = "train/train"   # expects folders "0", "1", ..., "99"
TEST_DIR = "test/test"       # expects test images named "0.jpg", "1.jpg", etc.

SUBMISSION_FILE = "submission.csv"
BEST_MODEL_PATH = "best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 3. Data Transforms
# -----------------------------
# Aggressive training transforms with AutoAugment, stronger jitter, and erasing.
train_transforms = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),  # wider crop range
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.2),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.3))
])

# Validation transforms remain deterministic.
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# TTA transforms for test: slight randomness for multiple inference passes.
tta_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 10, IMG_SIZE + 10)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 4. Prepare Training & Validation Sets
# -----------------------------
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)

# **Fix:** Override the default label mapping to use the folder name.
def get_label_from_path(path):
    folder = os.path.basename(os.path.dirname(path))
    return int(folder)

full_dataset.samples = [(p, get_label_from_path(p)) for (p, _) in full_dataset.samples]
full_dataset.class_to_idx = {folder: int(folder) for folder in full_dataset.classes}

# Build mapping for submission (should be identity if folders are named "0", "1", etc.)
idx_to_label = {v: int(v) for v in full_dataset.class_to_idx.values()}

# Split dataset: 80% train, 20% val.
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(42))
# Set validation transform.
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -----------------------------
# 5. Custom Test Dataset
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Collect only .jpg files and sort numerically by filename prefix.
        self.image_files = sorted(
            [f for f in os.listdir(root) if f.endswith('.jpg')],
            key=lambda x: int(x.split('.')[0])
        )
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        filename = self.image_files[idx]  # e.g. "0.jpg"
        img_path = os.path.join(self.root, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, filename

test_dataset = TestDataset(TEST_DIR, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
if len(test_dataset) == 0:
    print(f"[Warning] No .jpg files found in '{TEST_DIR}'. Submission.csv will be empty!")

# -----------------------------
# 6. Build Model (Swin Transformer)
# -----------------------------
def build_model():
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=100)
    # Partial freezing: initially freeze all layers except layers.2, layers.3, and head.
    for name, param in model.named_parameters():
        if ("head" in name) or ("layers.2" in name) or ("layers.3" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model.to(device)

# -----------------------------
# 7. MixUp & CutMix Functions
# -----------------------------
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    # Random bounding box
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -----------------------------
# 8. Train/Val/TTA Functions
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, use_mixup=True, use_cutmix=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # Use MixUp in first half of training and CutMix in second half.
        if use_mixup and not use_cutmix:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.4)
            outputs = model(images)
            loss = mixup_cutmix_criterion(criterion, outputs, y_a, y_b, lam)
        elif use_cutmix and not use_mixup:
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
            outputs = model(images)
            loss = mixup_cutmix_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total

def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, 100.0 * correct / total

def tta_predict(model, dataset, num_tta=5):
    model.eval()
    predictions = []
    for idx in range(len(dataset)):
        filename = dataset.image_files[idx]
        img_path = os.path.join(dataset.root, filename)
        img = Image.open(img_path).convert("RGB")
        avg_output = None
        for _ in range(num_tta):
            aug_img = tta_transforms(img).unsqueeze(0).to(device)
            output = model(aug_img)
            if avg_output is None:
                avg_output = output
            else:
                avg_output += output
        avg_output /= num_tta
        _, pred = torch.max(avg_output, 1)
        predictions.append((filename, pred.item()))
    return predictions

# -----------------------------
# 9. Gradual Unfreezing
# -----------------------------
def unfreeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        for l_name in layer_names:
            if l_name in name:
                param.requires_grad = True
    print(f"Unfroze layers containing: {layer_names}")

# -----------------------------
# 10. Main Training Loop (15 Epochs)
# -----------------------------
if __name__ == "__main__":
    model = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_acc = 0.0

    print("Starting training with advanced techniques (MixUp, CutMix, gradual unfreezing, etc.)...")
    for epoch in range(1, EPOCHS + 1):
        # Use MixUp for epochs 1-7, CutMix for epochs 8-15.
        use_mixup = (epoch <= 7)
        use_cutmix = (epoch > 7)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                                 use_mixup=use_mixup, use_cutmix=use_cutmix)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best val acc! Model saved. (acc={val_acc:.2f}%)")
        # Gradually unfreeze layers:
        if epoch == 5:
            unfreeze_layers(model, ["layers.1"])  # unfreeze stage 1
            params_to_optimize = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params_to_optimize, lr=LR, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch + 1)
        if epoch == 10:
            unfreeze_layers(model, ["layers.0"])  # optionally unfreeze even earlier layers
            params_to_optimize = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params_to_optimize, lr=LR * 0.5, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch + 1)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded best model from '{BEST_MODEL_PATH}' for TTA inference.")

    predictions = tta_predict(model, test_dataset, num_tta=5)

    # Create submission DataFrame in correct format.
    df_preds = pd.DataFrame(predictions, columns=["ID", "Label"])
    df_preds["ID_int"] = df_preds["ID"].apply(lambda x: int(x.split('.')[0]))
    df_preds.sort_values("ID_int", inplace=True)
    df_preds.drop(columns=["ID_int"], inplace=True)
    df_preds.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSubmission saved to {SUBMISSION_FILE} (rows={len(df_preds)})")
    print(f"Final best val accuracy: {best_val_acc:.2f}%")
