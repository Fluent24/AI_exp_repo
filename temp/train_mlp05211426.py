import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import wandb
from scipy.stats import pearsonr
import audiofile
from transformers import Wav2Vec2Model

class CNN_LSTM_RegressionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super(CNN_LSTM_RegressionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, mel_bins = x.size()
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def open_file(filename):
    with open(filename) as f:
        return f.readlines()

def feat_extraction(args, data_type):
    fname_list = os.path.join(args.dir_list, f'lang_{args.lang}', f'{args.label_type1}_{data_type}.list')
    filelist = open_file(fname_list)
    data_len = len(filelist)

    model = Wav2Vec2Model.from_pretrained(args.base_model).to(args.device)

    features = []
    labels = []

    for idx, line in enumerate(filelist):
        try:
            fname, score1, score2, text = line.split('\t')
        except:
            continue

        try:
            if args.dir_data:
                fname = os.path.join(args.dir_data, fname.split('/audio/')[-1])
            x, sr = audiofile.read(fname)
        except:
            continue

        score = float(score1 if args.label_type2 == 'articulation' else score2)

        if x.shape[-1] > args.audio_len_max:
            x = x[:args.audio_len_max]

        x = torch.tensor(x, device=args.device).reshape(1, -1)
        with torch.no_grad():
            outputs = model(x)
        feat_x = outputs.last_hidden_state.squeeze().cpu().numpy()
        features.append(feat_x)
        labels.append(score)

    # 패딩 처리하여 동일한 길이로 맞춤
    features = pad_sequences(features, dtype=np.float32)

    return features, np.array(labels).reshape(-1, 1)

def load_or_extract_features(args, data_type):
    feature_dir = os.path.join(args.dir_model, "datasets_full_list", f"lang_{args.lang}")
    os.makedirs(feature_dir, exist_ok=True)
    feature_file = os.path.join(feature_dir, f"{args.label_type1}_{data_type}.npz")

    if os.path.exists(feature_file):
        print(f"Loading features from {feature_file}")
        data = np.load(feature_file, allow_pickle=True)
        feat_X, feat_Y = data["X"], data["Y"]
    else:
        print(f"Extracting features and saving to {feature_file}")
        feat_X, feat_Y = feat_extraction(args, data_type)
        np.savez(feature_file, X=feat_X, Y=feat_Y)

    print(f"wav2vec2 feature {data_type}, {len(feat_X)}, {feat_Y.shape}")
    return feat_X, feat_Y

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.):
    lengths = [len(seq) for seq in sequences]

    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = np.asarray(sequences[0]).shape[1:]
    x = np.full((len(sequences), maxlen) + sample_shape, value, dtype=dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '{}' not understood".format(truncating))

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError("Shape of sample {} of sequence at position {} is different from expected shape {}"
                             .format(trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '{}' not understood".format(padding))

    return x

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='en', type=str)
    parser.add_argument("--label_type1", default='pron', type=str, help='fluency, pron')
    parser.add_argument("--label_type2", default='prosody', type=str, help='articulation, prosody')
    parser.add_argument("--dir_list", default='data_list', type=str)
    parser.add_argument("--audio_len_max", default=200000, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--num_workers", type=int, default=4, help="")
    parser.add_argument("--base_model", type=str, default=None, help="")
    parser.add_argument("--dir_model", default='model', type=str)
    parser.add_argument("--dir_data", type=str, default=None, help="")
    parser.add_argument("--dir_resume", type=str, default=None, help="")
    parser.add_argument("--base_dim", default=1024, type=int)
    parser.add_argument("--mlp_hidden", default=64, type=int)
    parser.add_argument("--lr", type=float, default=0.01, help="")
    parser.add_argument("--epochs", type=int, default=400, help="")
    parser.add_argument("--batch_size", type=int, default=16, help="")
    parser.add_argument("--patience", type=int, default=20, help="")
    parser.add_argument("--model_type", type=str, choices=['mlp', 'cnn', 'lstm', 'transformer'], default='transformer', help="Type of model to train")
    args = parser.parse_args()

    wandb.init(project="model_test_pron+articulation", config=args)
    config = wandb.config

    dir_save_model = f'{args.dir_model}/lang_{args.lang}'
    os.makedirs(dir_save_model, exist_ok=True)

    if args.lang == 'en':
        args.base_model = 'facebook/wav2vec2-large-robust-ft-libri-960h'

    print(f'base wav2vec2 model: {args.base_model}')

    trn_feat_x, trn_feat_y = load_or_extract_features(args, 'trn')
    val_feat_x, val_feat_y = load_or_extract_features(args, 'val')
    test_feat_x, test_feat_y = load_or_extract_features(args, 'test')

    trn_feat_x = pad_sequences(trn_feat_x, dtype=np.float32)
    val_feat_x = pad_sequences(val_feat_x, maxlen=trn_feat_x.shape[1], dtype=np.float32)
    test_feat_x = pad_sequences(test_feat_x, maxlen=trn_feat_x.shape[1], dtype=np.float32)

    tr_dataset = TensorDataset(torch.tensor(trn_feat_x), torch.tensor(trn_feat_y))
    val_dataset = TensorDataset(torch.tensor(val_feat_x), torch.tensor(val_feat_y))
    test_dataset = TensorDataset(torch.tensor(test_feat_x), torch.tensor(test_feat_y))

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    net = CNN_LSTM_RegressionModel(input_dim=trn_feat_x.shape[2]).to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    eval_best_pcc = -9
    early_stop_counter = 0
    stop_flag = False

    for epoch in range(args.epochs):
        # Training
        net.train()
        train_loss = 0
        for train_data in tr_loader:
            feat_x, feat_y = train_data
            optimizer.zero_grad()

            prediction = net(feat_x.to(args.device))
            loss = criterion(prediction, feat_y.to(args.device))

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(tr_loader)
        wandb.log({"epoch": epoch, "train_loss": train_loss})
        print(f'epoch {epoch}, train loss: {train_loss}')

        # Validation
        net.eval()
        val_loss = 0
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for val_data in val_loader:
                feat_x, feat_y = val_data

                val_labels.extend(feat_y.tolist())
                prediction = net(feat_x.to(args.device))
                val_preds.extend(prediction.cpu().tolist())

                loss = criterion(prediction, feat_y.to(args.device))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_labels = np.array(val_labels).squeeze()
        val_preds = np.clip(np.array(val_preds).squeeze(), 0, 5)

        eval_pcc = pearsonr(val_labels, val_preds)[0]

        wandb.log({"epoch": epoch, "val_loss": val_loss, "eval_pcc": eval_pcc})
        print(f'epoch {epoch}, val loss: {val_loss}, eval_pcc: {eval_pcc}')

        # Testing
        net.eval()
        test_labels = []
        test_preds = []
        with torch.no_grad():
            for test_data in test_loader:
                feat_x, feat_y = test_data

                test_labels.extend(feat_y.tolist())
                prediction = net(feat_x.to(args.device))
                test_preds.extend(prediction.cpu().tolist())

        test_labels = np.array(test_labels).squeeze()
        test_preds = np.clip(np.array(test_preds).squeeze(), 0, 5)

        test_pcc = pearsonr(test_labels, test_preds)[0]

        print(f'epoch {epoch}, eval_pcc: {eval_pcc}, test_pcc: {test_pcc}')
        wandb.log({"epoch": epoch, "eval_pcc": eval_pcc, "test_pcc": test_pcc})

        # Early stopping
        if eval_pcc > eval_best_pcc and not stop_flag:
            eval_best_pcc = eval_pcc
            test_best_pcc = test_pcc
            early_stop_counter = 0
            torch.save(net.state_dict(), os.path.join(dir_save_model, f'{args.label_type1}_{args.label_type2}_{args.model_type}_checkpoint.pt'))
        else:
            early_stop_counter += 1

        if early_stop_counter > args.patience and not stop_flag:
            print("Early stopping triggered.")
            break

    print(f'Final Test PCC: {test_best_pcc}')
    wandb.log({"final_test_pcc": test_best_pcc})
    wandb.finish()

if __name__ == "__main__":
    train()
