import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, data_dir, label_file, feature_extractor):
        self.data_dir = data_dir
        self.label_file = label_file
        self.feature_extractor = feature_extractor
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.label_file, 'r') as f:
            for line in f:
                audio_path, label = line.strip().split(',')
                data.append((audio_path, float(label)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        waveform, sample_rate = torchaudio.load(os.path.join(self.data_dir, audio_path))
        features = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_values
        return features.squeeze(), torch.tensor(label)

# 데이터 경로 및 파일 설정
DIR_DATA = "data/audio"
TRAIN_LABEL_FILE = os.path.join(DIR_LIST, "train_labels.csv")
VAL_LABEL_FILE = os.path.join(DIR_LIST, "val_labels.csv")
TEST_LABEL_FILE = os.path.join(DIR_LIST, "test_labels.csv")

# Wav2Vec2 feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")

# 데이터셋 및 데이터로더 설정
train_dataset = AudioDataset(DIR_DATA, TRAIN_LABEL_FILE, feature_extractor)
val_dataset = AudioDataset(DIR_DATA, VAL_LABEL_FILE, feature_extractor)
test_dataset = AudioDataset(DIR_DATA, TEST_LABEL_FILE, feature_extractor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

class AudioModel(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, output_dim):
        super(AudioModel, self).__init__()
        if model_type == "mlp":
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif model_type == "cnn":
            self.model = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(hidden_dim * (input_dim // 2), output_dim)
            )
        elif model_type == "lstm":
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        elif model_type == "transformer":
            self.transformer = nn.Transformer(input_dim, hidden_dim, num_encoder_layers=3)
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if hasattr(self, 'lstm'):
            x, _ = self.lstm(x)
            x = self.fc(x[:, -1, :])
        elif hasattr(self, 'transformer'):
            x = self.transformer(x, x)
            x = self.fc(x[:, -1, :])
        else:
            x = self.model(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    eval_preds, eval_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            eval_preds.append(outputs.cpu().numpy())
            eval_labels.append(targets.cpu().numpy())
    eval_preds = np.concatenate(eval_preds)
    eval_labels = np.concatenate(eval_labels)
    pcc = pearsonr(eval_labels, eval_preds)[0]
    return total_loss / len(val_loader), pcc

def main():
    LANG = "en"
    LABEL_TYPE1 = "pron"
    LABEL_TYPE2 = "articulation"
    DIR_DATA = "data/audio"
    DIR_LIST = "/home/coldbrew/fluent/01.발음평가모델/1.모델소스코드/datasets_small_list"
    MLP_HIDDEN = 64
    EPOCHS = 200
    PATIENCE = 20
    BATCH_SIZE = 256
    DIR_MODEL = "model_test_pron+articulation"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust-ft-libri-960h")
    model_name = "facebook/wav2vec2-large-robust-ft-libri-960h"
    wav2vec2 = Wav2Vec2Model.from_pretrained(model_name).to(device)

    # 데이터셋 및 데이터로더 설정
    train_dataset = AudioDataset(DIR_DATA, os.path.join(DIR_LIST, "train_labels.csv"), feature_extractor)
    val_dataset = AudioDataset(DIR_DATA, os.path.join(DIR_LIST, "val_labels.csv"), feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for model_type in ["mlp", "cnn", "lstm", "transformer"]:
        model = AudioModel(model_type, input_dim=1024, hidden_dim=MLP_HIDDEN, output_dim=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        best_pcc = -1
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_pcc = evaluate(model, val_loader, criterion, device)

            if val_pcc > best_pcc:
                best_pcc = val_pcc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(DIR_MODEL, f"best_model_{model_type}.pt"))
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val PCC: {val_pcc}")

if __name__ == "__main__":
    main()
