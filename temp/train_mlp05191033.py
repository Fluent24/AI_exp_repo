import os
import json
import audiofile
from transformers import Wav2Vec2ForCTC
import torch
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
import wandb
from score_model import MLP, AudioClassifier1DCNN,AudioClassifier1DCNN_Advanced, AudioClassifierLSTM, MLP_Advanced, AudioClassifierTransformer

def open_file(filename):
    with open(filename) as f:
        return f.readlines()

def feat_extraction(args, data_type):
    ''' wav2vec2 feature extraction part '''
    fname_list = os.path.join(args.dir_list, f'lang_{args.lang}', f'{args.label_type1}_{data_type}.list')
    filelist = open_file(fname_list)
    data_len = len(filelist)

    feat_X = np.zeros((data_len, args.base_dim), dtype=np.float32)  # features
    feat_Y = np.zeros((data_len, 1), dtype=np.float32)  # labels

    model = Wav2Vec2ForCTC.from_pretrained(args.base_model).to(args.device)  # load wav2vec2 model

    valid_idx = 0
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
        output = model(x, output_attentions=False, output_hidden_states=True, return_dict=True)  # wav2vec2 model output

        feat_x = output.hidden_states[-1]  # last hidden state of wav2vec2, (1, frame, 1024)
        feat_x = torch.mean(feat_x, axis=1).cpu().detach().numpy()  # pooled output along time axis, (1, 1024)

        feat_X[valid_idx, :] = feat_x
        feat_Y[valid_idx, 0] = score
        valid_idx += 1

    print(f"wav2vec2 feature extraction {data_type}, {feat_X[:valid_idx, :].shape}, {feat_Y[:valid_idx, :].shape}")

    return feat_X[:valid_idx, :], feat_Y[:valid_idx, :]

def load_or_extract_features(args, data_type):
    """Load features from file if they exist, otherwise extract them and save to file."""
    feature_dir = os.path.join(args.dir_model, "datasets_full_list", f"lang_{args.lang}")
    os.makedirs(feature_dir, exist_ok=True)
    feature_file = os.path.join(feature_dir, f"{args.label_type1}_{data_type}.npz")

    if os.path.exists(feature_file):
        print(f"Loading features from {feature_file}")
        data = np.load(feature_file)
        feat_X, feat_Y = data["X"], data["Y"]
    else:
        print(f"Extracting features and saving to {feature_file}")
        feat_X, feat_Y = feat_extraction(args, data_type)
        np.savez(feature_file, X=feat_X, Y=feat_Y)

    print(f"wav2vec2 feature {data_type}, {feat_X.shape}, {feat_Y.shape}")
    return feat_X, feat_Y

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='en', type=str)
    parser.add_argument("--label_type1", default='pron', type=str, help='fluency, pron')
    parser.add_argument("--label_type2", default='prosody', type=str, help='articulation, prosody')
    parser.add_argument("--dir_list", default='/data/project/rw/nia_db/nia_2022_012/data_prepare', type=str)
    parser.add_argument("--audio_len_max", default=200000, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--num_workers", type=int, default=1, help="")
    parser.add_argument("--base_model", type=str, default=None, help="")
    parser.add_argument("--dir_model", default='model', type=str)
    parser.add_argument("--dir_data", type=str, default=None, help="")
    parser.add_argument("--dir_resume", type=str, default=None, help="")
    parser.add_argument("--base_dim", default=1024, type=int)
    parser.add_argument("--mlp_hidden", default=64, type=int)
    parser.add_argument("--lr", type=float, default=0.01, help="")
    parser.add_argument("--epochs", type=int, default=400, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--patience", type=int, default=20, help="")
    parser.add_argument("--model_type", type=str, choices=['mlp', 'cnn', 'lstm', 'transformer'], default='mlp', help="Type of model to train")
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

    tr_dataset = TensorDataset(torch.tensor(trn_feat_x), torch.tensor(trn_feat_y))
    val_dataset = TensorDataset(torch.tensor(val_feat_x), torch.tensor(val_feat_y))
    test_dataset = TensorDataset(torch.tensor(test_feat_x), torch.tensor(test_feat_y))

    train_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialize the chosen model
    if args.model_type == 'mlp':
        net = MLP_Advanced(args).to(args.device)
    elif args.model_type == 'cnn':
        net = AudioClassifier1DCNN_Advanced(input_dim=args.base_dim).to(args.device)
    elif args.model_type == 'lstm':
        net = AudioClassifierLSTM(input_dim=args.base_dim, hidden_dim=128, num_layers=2).to(args.device)
    elif args.model_type == 'transformer':
        net = AudioClassifierTransformer(input_dim=args.base_dim, nhead=8, num_layers=2).to(args.device)

    if args.dir_resume is not None:
        dir_resume_model = os.path.join(args.dir_resume, f'lang_{args.lang}', f'{args.label_type1}_{args.label_type2}_checkpoint.pt')
        net.load_state_dict(torch.load(dir_resume_model, map_location=args.device))
        print(f'Training a model from {dir_resume_model}')
    else:
        print(f'Training a model from scratch')

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  # training optimizer
    loss_func = torch.nn.MSELoss()  # MSE loss for regression task

    eval_best_pcc = -9
    early_stop_counter = 0
    stop_flag = False

    for epoch in range(args.epochs):
        # Training
        net.train()
        train_loss = 0
        for train_data in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            feat_x, feat_y = train_data
            optimizer.zero_grad()

            # if args.model_type == 'cnn':
            #     feat_x = feat_x.unsqueeze(1)  # Add channel dimension for CNN
            # elif args.model_type in ['lstm', 'transformer']:
            #     feat_x = feat_x.unsqueeze(1)  # Add sequence dimension for LSTM and Transformer

            prediction = net(feat_x.to(args.device))
            loss = loss_func(prediction, feat_y.to(args.device))
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        wandb.log({"epoch": epoch, "train_loss": train_loss})
        print(f'epoch {epoch}, train loss: {train_loss}')

        # Validation
        net.eval()
        val_loss = 0
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for val_data in val_dataloader:
                feat_x, feat_y = val_data

                if args.model_type == 'cnn':
                    feat_x = feat_x.unsqueeze(1)  # Add channel dimension for CNN
                elif args.model_type in ['lstm', 'transformer']:
                    feat_x = feat_x.unsqueeze(1)  # Add sequence dimension for LSTM and Transformer

                val_labels.extend(feat_y.tolist())
                prediction = net(feat_x.to(args.device))
                val_preds.extend(prediction.cpu().tolist())

                loss = loss_func(prediction, feat_y.to(args.device))
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
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
            for test_data in test_dataloader:
                feat_x, feat_y = test_data

                if args.model_type == 'cnn':
                    feat_x = feat_x.unsqueeze(1)  # Add channel dimension for CNN
                elif args.model_type in ['lstm', 'transformer']:
                    feat_x = feat_x.unsqueeze(1)  # Add sequence dimension for LSTM and Transformer

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

#python train_mlp05191033.py --lang='en' --label_type1='pron' --label_type2='articulation'  --dir_list='/home/coldbrew/fluent/01.발음평가모델/1.모델소스코드/datasets_small_list' --mlp_hidden=64 --epochs=200 --patience=20 --batch_size=256 --dir_model='model_test1' --model_type='mlp'