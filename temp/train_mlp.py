import glob, os
import json
import audiofile
from transformers import Wav2Vec2ForCTC
import torch
import numpy as np
import argparse
from torch.utils.data import TensorDataset
from scipy.stats import pearsonr
from tqdm import tqdm
import wandb

from score_model import MLP
from score_model import AudioClassifier1DCNN
from score_model import AudioClassifierLSTM
from score_model import AudioClassifierTransformer

def open_file(filename):
    f = open(filename)
    filelist = f.readlines()
    f.close()
    return filelist

def feat_extraction(args, data_type):
    ''' wav2vec2 feature extraction part '''
    fname_list = os.path.join(args.dir_list, f'lang_{args.lang}', f'{args.label_type1}_{data_type}.list')
    filelist = open_file(fname_list)
    data_len = len(filelist)

    feat_X = np.zeros((data_len, args.base_dim), dtype=np.float32)  # features
    feat_Y = np.zeros((data_len, 1), dtype=np.float32)  # labels

    model = Wav2Vec2ForCTC.from_pretrained(args.base_model).to(args.device)  # load wav2vec2 model

    for idx, line in enumerate(filelist):
        try:
            fname, score1, score2, text = line.split('\t')  # wavfile path, articulation score, prosody score, script
        except:
            data_len -= 1  # if list file format is wrong, we exclude it
            continue

        try:
            if args.dir_data is not None:
                fname = fname.split('/audio/')[-1]
                fname = os.path.join(args.dir_data, fname)
            x, sr = audiofile.read(fname)
        except:
            data_len -= 1
            continue

        if args.label_type2 == 'articulation':
            score = score1
        else:
            score = score2 

        if x.shape[-1] > args.audio_len_max:
            x = x[:args.audio_len_max]  # if audio file is long, cut it to audio_len_max

        x = torch.tensor(x, device=args.device).reshape(1, -1)
        output = model(x, output_attentions=True, output_hidden_states=True, return_dict=True)  # wav2vec2 model output

        feat_x = output.hidden_states[-1]  # last hidden state of wav2vec2, (1, frame, 1024)
        feat_x = torch.mean(feat_x, axis=1).cpu().detach().numpy()  # pooled output along time axis, (1, 1024)

        feat_X[idx, :] = feat_x
        feat_Y[idx, 0] = float(score)

    print(f"wav2vec2 feature extraction {data_type}, {feat_X[:data_len, :].shape}, {feat_Y[:data_len, :].shape}")

    return feat_X[:data_len, :], feat_Y[:data_len, :]

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

    trn_feat_x, trn_feat_y = feat_extraction(args, 'trn')  # feature extraction for training data
    val_feat_x, val_feat_y = feat_extraction(args, 'val')  # feature extraction for validation data
    test_feat_x, test_feat_y = feat_extraction(args, 'test')  # feature extraction for test data

    tr_dataset = TensorDataset(torch.tensor(trn_feat_x), torch.tensor(trn_feat_y))
    val_dataset = TensorDataset(torch.tensor(val_feat_x), torch.tensor(val_feat_y))
    test_dataset = TensorDataset(torch.tensor(test_feat_x), torch.tensor(test_feat_y))

    train_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialize the chosen model
    if args.model_type == 'mlp':
        net = MLP(args).to(args.device)
    elif args.model_type == 'cnn':
        net = AudioClassifier1DCNN(input_dim=args.base_dim).to(args.device)
    elif args.model_type == 'lstm':
        net = AudioClassifierLSTM(input_dim=args.base_dim, hidden_dim=128, num_layers=2).to(args.device)
    elif args.model_type == 'transformer':
        net = AudioClassifierTransformer(input_dim=args.base_dim, nhead=8, num_layers=2).to(args.device)

    if args.dir_resume is not None:
        dir_resume_model = os.path.join(args.dir_resume, f'lang_{args.lang}', f'{args.label_type1}_{args.label_type2}_checkpoint.pt')
        net = torch.load(dir_resume_model).to(args.device)
        print(f'Training a model from {dir_resume_model}')
    else:
        print(f'Training a model from scratch')

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  # training optimizer, we use adam optimizer for training
    loss_func = torch.nn.MSELoss()  # MSE loss for regression task

    eval_best_pcc = -9
    early_stop_counter = 0
    stop_flag = False

    for i in range(args.epochs):
        # Training
        net.train()
        for idx, train_data in tqdm(enumerate(train_dataloader)):
            feat_x, feat_y = train_data

            if args.model_type == 'cnn':
                feat_x = feat_x.unsqueeze(1)  # Add channel dimension for CNN
            elif args.model_type == 'lstm' or args.model_type == 'transformer':
                feat_x = feat_x.unsqueeze(1)  # Add sequence dimension for LSTM and Transformer

            prediction = net(feat_x.to(args.device))
            loss = loss_func(prediction, feat_y.to(args.device))
            loss.backward()
            if i % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f'epoch{i}, loss: {loss.item()}')
        wandb.log({"epoch": i, "loss": loss.item()})

        # Calculate PCC of validation data
        net.eval()
        eval_labels, eval_preds = [], []
        for idx, eval_data in enumerate(val_dataloader):
            feat_x, feat_y = eval_data

            if args.model_type == 'cnn':
                feat_x = feat_x.unsqueeze(1)  # Add channel dimension for CNN
            elif args.model_type == 'lstm' or args.model_type == 'transformer':
                feat_x = feat_x.unsqueeze(1)  # Add sequence dimension for LSTM and Transformer

            eval_labels.extend(feat_y.tolist())
            prediction = net(feat_x.to(args.device))
            eval_preds.extend(prediction.cpu().tolist())
        eval_labels = np.array(eval_labels).squeeze()
        eval_preds = np.clip(np.array(eval_preds).squeeze(), 0, 5)

        eval_pcc = pearsonr(eval_labels, eval_preds)[0]

        # Calculate PCC of test data
        test_labels, test_preds = [], []
        for idx, test_data in enumerate(test_dataloader):
            feat_x, feat_y = test_data

            if args.model_type == 'cnn':
                feat_x = feat_x.unsqueeze(1)  # Add channel dimension for CNN
            elif args.model_type == 'lstm' or args.model_type == 'transformer':
                feat_x = feat_x.unsqueeze(1)  # Add sequence dimension for LSTM and Transformer

            test_labels.extend(feat_y.tolist())
            prediction = net(feat_x.to(args.device))
            test_preds.extend(prediction.cpu().tolist())
        test_labels = np.array(test_labels).squeeze()
        test_preds = np.clip(np.array(test_preds).squeeze(), 0, 5)

        test_pcc = pearsonr(test_labels, test_preds)[0]

        print(f'eval_pcc: {eval_pcc}, test_pcc: {test_pcc}')
        wandb.log({"epoch": i, "eval_pcc": eval_pcc, "test_pcc": test_pcc})

        # Early stopping
        if eval_pcc > eval_best_pcc and not stop_flag:
            eval_best_pcc = eval_pcc
            test_best_pcc = test_pcc
            early_stop_counter = 0
            torch.save(net, os.path.join(dir_save_model, f'{args.label_type1}_{args.label_type2}_{args.model_type}_checkpoint.pt'))
        else:
            early_stop_counter += 1

        if early_stop_counter > args.patience and not stop_flag:  # Training will stop if the model doesn't show improvement over args.patience epochs
            break

    print(f'Final Test PCC: {test_best_pcc}')
    wandb.log({"final_test_pcc": test_best_pcc})
    wandb.finish()

if __name__ == "__main__":
    train()

