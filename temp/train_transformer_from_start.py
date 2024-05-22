import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import wandb
import audiofile

class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout=0.1):
        super(TransformerRegressionModel, self).__init__()
        
        # TransformerEncoderLayer를 여러 개 쌓아 TransformerEncoder 구성
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout)
        
        # 최종 출력을 위한 선형 레이어
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_dim] 형태로 변환
        x = self.transformer_encoder(x)
        
        # 드롭아웃 적용
        x = self.dropout(x)
        
        # 최종 출력 값 계산 (평균 풀링 후 선형 변환)
        x = x.mean(dim=0)  # 각 시퀀스의 평균 값 계산
        x = self.fc(x)
        return x

def open_file(filename):
    with open(filename) as f:
        return f.readlines()

def feat_extraction(args, data_type):
    ''' wav2vec2 feature extraction part '''
    fname_list = os.path.join(args.dir_list, f'lang_{args.lang}', f'{args.label_type1}_{data_type}.list')
    filelist = open_file(fname_list)
    data_len = len(filelist)

    model = Wav2Vec2Model.from_pretrained(args.base_model).to(args.device)  # load wav2vec2 model

    features = []
    labels = []

    for idx, line in enumerate(filelist):
        try:
            fname, score1, score2, text = line.split('\t')  # wavfile path, articulation score, prosody score, script
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
            x = x[:args.audio_len_max]  # if audio file is long, cut it to audio_len_max

        x = torch.tensor(x, device=args.device).reshape(1, -1)
        with torch.no_grad():
            outputs = model(x)
        feat_x = outputs.last_hidden_state.squeeze().cpu().numpy()  # [seq_len, hidden_dim]
        features.append(feat_x)
        labels.append(score)

    return features, np.array(labels).reshape(-1, 1)

def load_or_extract_features(args, data_type):
    """Load features from file if they exist, otherwise extract them and save to file."""
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
    """Pads sequences to the same length."""
    lengths = [len(seq) for seq in sequences]

    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = np.asarray(sequences[0]).shape[1:]
    x = np.full((len(sequences), maxlen) + sample_shape, value, dtype=dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
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

    # Initialize wandb
    wandb.init(project="model_test_pron+articulation", config=args)
    config = wandb.config

    dir_save_model = f'{args.dir_model}/lang_{args.lang}'
    os.makedirs(dir_save_model, exist_ok=True)

    if args.lang == 'en':
        args.base_model = 'facebook/wav2vec2-large-robust-ft-libri-960h'

    print(f'base wav2vec2 model: {args.base_model}')

    trn_feat_x, trn_feat_y = load_or_extract_features(args, 'trn')  # feature extraction or loading for training data
    val_feat_x, val_feat_y = load_or_extract_features(args, 'val')  # feature extraction or loading for validation data
    test_feat_x, test_feat_y = load_or_extract_features(args, 'test')  # feature extraction or loading for test data

    # 패딩 적용
    trn_feat_x = pad_sequences(trn_feat_x)
    val_feat_x = pad_sequences(val_feat_x, maxlen=trn_feat_x.shape[1])
    test_feat_x = pad_sequences(test_feat_x, maxlen=trn_feat_x.shape[1])

    tr_dataset = TensorDataset(torch.tensor(trn_feat_x, dtype=torch.float32), torch.tensor(trn_feat_y, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_feat_x, dtype=torch.float32), torch.tensor(val_feat_y, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_feat_x, dtype=torch.float32), torch.tensor(test_feat_y, dtype=torch.float32))

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TransformerRegressionModel(input_dim=trn_feat_x.shape[2], hidden_dim=args.base_dim, num_layers=4, num_heads=8, output_dim=1).to(args.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(dir_save_model, 'best_model.pt'))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Load the best model for evaluation
    model.load_state_dict(torch.load(os.path.join(dir_save_model, 'best_model.pt')))
    model.eval()

    test_losses = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_losses.append(loss.item())

    test_loss = np.mean(test_losses)
    print(f'Test Loss: {test_loss:.4f}')
    wandb.log({"test_loss": test_loss})

if __name__ == "__main__":
    train()
