from dp_model.model_files.sfcn import SFCN
from dp_model import dp_loss as dpl
from dp_model import dp_utils as dpu

import os
import random
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve, auc


# -----------------------
# Paths and Hyperparameters
# -----------------------
input_root = '../ABIDE_Dataset/data/JustBrain/ABIDEI'
participants_path = './ABIDEI/participants.tsv'
label_column = 'label'
split_column = 'dataset'
pretrained_model_path = './sex_prediction/run_20191008_00_epoch_last.p'
save_model_path = './ABIDEI/finetuned_sfcn_best.pth'

# -----------------------
# Set Random Seed
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -----------------------
# Parser Arguments
# -----------------------
parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('--lr', type=float, default = 1e-4, help='Learning Rate (ex 3e-4 or 0.0003)')
parser.add_argument('--batch', type=int, default = 4, help='Batch Size')
parser.add_argument('--epochs', type=int, default = 15, help='Total Number of Epochs')

## Parse Data
args = parser.parse_args()
learning_rate = args.lr
batch_size = args.batch
num_epochs = args.epochs

# -----------------------
# Model
# -----------------------
model = SFCN(output_dim=2, channel_number=[28, 58, 128, 256, 256, 64])
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(pretrained_model_path, weights_only=True))
model = model.cuda()

# -----------------------
# Custom Dataset
# -----------------------
class ABIDEIDataset(Dataset):
    def __init__(self, input_root, df_subset, label_column):
        self.input_root = input_root
        self.df = df_subset
        self.subjects = list(df_subset.index)
        self.label_column = label_column

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        anat_dir = os.path.join(self.input_root, subject_id, 'anat')
        nii_files = [f for f in os.listdir(anat_dir) if f.endswith('.nii.gz')]
        t1w_file = next((f for f in nii_files if subject_id in f and 'T1w' in f), None)

        full_path = os.path.join(anat_dir, t1w_file)
        data = nib.load(full_path).get_fdata()
        data = data / data.mean()
        data = dpu.crop_center(data, (160, 192, 160))  # custom center crop
        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # shape: (1, D, H, W)

        label = self.df.loc[subject_id, self.label_column]
        return input_tensor, int(label)

# -----------------------
# Load and Split Data
# -----------------------
df_all = pd.read_csv(participants_path, sep='\t')
df_all['participant_id'] = df_all['participant_id'].str.strip()
df_all = df_all.set_index('participant_id')

df_train = df_all[df_all[split_column] == 'train']
df_val = df_all[df_all[split_column] == 'val']

train_loader = DataLoader(ABIDEIDataset(input_root, df_train, label_column), batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(ABIDEIDataset(input_root, df_val, label_column), batch_size=batch_size, shuffle=False, num_workers=8)

# -----------------------
# Training Setup
# -----------------------
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_auc = 0.0

# -----------------------
# Training & Evaluation Loop
# -----------------------
train_losses = []
val_losses = []
val_aucs = []

for epoch in range(num_epochs):
    # ---- Training ----
    model.train()
    train_loss, train_y, train_pred, train_prob = 0, [], [], []
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)[0].view(inputs.size(0), -1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        probs = torch.exp(outputs)
        preds = torch.argmax(probs, dim=1)

        train_y.extend(labels.cpu().numpy())
        train_pred.extend(preds.cpu().numpy())
        train_prob.extend(probs[:, 1].detach().cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_metrics = {
        'acc': accuracy_score(train_y, train_pred),
        'f1': f1_score(train_y, train_pred),
        'auc': roc_auc_score(train_y, train_prob)
    }

    # ---- Validation ----
    model.eval()
    val_loss, val_y, val_pred, val_prob = 0, [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)[0].view(inputs.size(0), -1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            probs = torch.exp(outputs)
            preds = torch.argmax(probs, dim=1)

            val_y.extend(labels.cpu().numpy())
            val_pred.extend(preds.cpu().numpy())
            val_prob.extend(probs[:, 1].cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_metrics = {
        'acc': accuracy_score(val_y, val_pred),
        'f1': f1_score(val_y, val_pred),
        'auc': roc_auc_score(val_y, val_prob)
    }


    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_aucs.append(val_metrics['auc'])

    
    # Print summary
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

    # Save best model
    if val_metrics['auc'] > best_val_auc:
        best_val_auc = val_metrics['auc']
        torch.save(model.state_dict(), save_model_path)
        print(f">>> Saved new best model at epoch {epoch+1} with AUC {best_val_auc:.4f}")

# -----------------------
# Save training log to CSV
# -----------------------
log_df = pd.DataFrame({
    'epoch': list(range(1, num_epochs + 1)),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'val_auc': val_aucs
})
log_csv_path = './ABIDEI/training_log.csv'
log_df.to_csv(log_csv_path, index=False)
print(f"Training log saved to: {log_csv_path}")

# -----------------------
# Plot ROC Curve using best model
# -----------------------
model.load_state_dict(torch.load(save_model_path))
model.eval()

val_y_true, val_y_prob = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)[0].view(inputs.size(0), -1)
        probs = torch.exp(outputs)
        val_y_prob.extend(probs[:, 1].cpu().numpy())
        val_y_true.extend(labels.numpy())

fpr, tpr, _ = roc_curve(val_y_true, val_y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation ROC Curve (Best Model)')
plt.legend(loc='lower right')
plt.grid(True)
roc_curve_path = './ABIDEI/val_roc_curve.png'
plt.tight_layout()
plt.savefig(roc_curve_path)
plt.show()
print(f"ROC curve saved to: {roc_curve_path}")

print("\nTraining complete.")