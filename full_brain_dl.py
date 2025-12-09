from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torchio as tio
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F

from monai.networks.nets import resnet

################## Creating the dataset with Torch IO

df = pd.read_csv("tinception_master.csv")
df['group'] = df['group'].map({'CO': 0, 'TI': 1})
tio_subjects = []
subjects = []
bet_dir = Path("/home/ubuntu/volume/Tindl/bet")
for fname in bet_dir.iterdir():
    if fname.suffix == ".gz" and "_bet" not in fname.stem:
        subjects.append(fname.stem[:-4])

print(f"number of subjects are: {len(subjects)}")

for sid in sorted(subjects):
    row = df.loc[df['subject ID'] == sid].iloc[0]

    subject = tio.Subject(
                        t1=tio.ScalarImage(bet_dir / f"{sid}.nii.gz"),
                        brain_mask=tio.LabelMap(bet_dir / f"{sid}_bet.nii.gz"),
                        label=int(row['group']),
                        age=float(row['age']),
                        sex=float(row['sex']),
                        PTA=float(row['PTA'])
                        )
    tio_subjects.append(subject)

preprocess = tio.Compose([
                        tio.ToCanonical(),                   
                        tio.Resample(1),                     
                        tio.ZNormalization(masking_method='brain_mask'),  
                        # tio.CropOrPad((160, 192, 160))
                        tio.CropOrPad((128, 128, 128))
                        ])

dataset = tio.SubjectsDataset(tio_subjects, transform=preprocess)


################## Model training

subjects_list = list(range(len(dataset)))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device is {device}")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def save_checkpoint(model, optimizer, epoch, path="model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)

################## The real part !!!

criterion = nn.CrossEntropyLoss()
for fold, (train_idx, val_idx) in enumerate(kf.split(subjects_list)):
    print(f"\n===== FOLD {fold+1} =====")

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set   = torch.utils.data.Subset(dataset, val_idx)
    
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=1)

    model = resnet.ResNet(
        block=resnet.ResNetBottleneck,
        # layers=[2, 2, 2, 2],
        # block_inplanes=[64, 128, 256, 512],
        layers=[1,1,1,1],
        block_inplanes=[32,64,128,256],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    early_stop = EarlyStopping(patience=8)

    # Store metrics per epoch
    train_losses, val_losses = [], []
    val_accuracies, val_f1s, val_rocauc = [], [], []

    for epoch in range(100):
        # ---- Training ----
        print(f"working on epoch number: {epoch}")
        model.train()
        running_loss = 0
        for batch in train_loader:
            images = batch['t1'][tio.DATA].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        y_true, y_pred, y_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['t1'][tio.DATA].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_probs.extend(probs[:,1].cpu().numpy())  # probability of class 1

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred)
        try:
            val_auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            val_auc = float('nan')  # in case only one class present

        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        val_rocauc.append(val_auc)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.3f} | "
                f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f} | "
                f"F1: {val_f1:.2f} | ROC-AUC: {val_auc:.2f}")

        # ---- Early stopping ----
        if early_stop.step(val_loss):
            print("Early stopping triggered.")
            break

        # ---- Save best checkpoint ----
        if early_stop.best_loss == val_loss:
            save_checkpoint(model, optimizer, epoch, f"best_fold{fold+1}.pth")

    # ---- Save metrics for this fold ----
    metrics_df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accuracies,
        'val_f1': val_f1s,
        'val_roc_auc': val_rocauc
    })
    metrics_df.to_csv(f"fold{fold+1}_metrics.csv", index=False)
