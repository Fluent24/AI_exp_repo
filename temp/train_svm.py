import os
import audiofile
import numpy as np
import argparse
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from transformers import Wav2Vec2ForCTC
from scipy.stats import pearsonr

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

    for idx, line in enumerate(filelist):

        try:
            fname, score1, score2, text = line.split('\t')  # wavfile path, articulation score, prosody score, script
        except:
            data_len -= 1  # if list file format is wrong, we exclude it
            continue

        try:
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

def load_or_extract_features(args, data_type):
    """Load features from file if they exist, otherwise extract them and save to file."""
    feature_dir = os.path.join("datasets_full_list_feature_extracted", f"lang_{args.lang}")
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

# def train_and_evaluate(args):
#     trn_feat_x, trn_feat_y = load_or_extract_features(args, 'trn')  # feature extraction or loading for training data
#     val_feat_x, val_feat_y = load_or_extract_features(args, 'val')  # feature extraction or loading for validation data
#     test_feat_x, test_feat_y = load_or_extract_features(args, 'test')  # feature extraction or loading for test data

#     scaler = StandardScaler()
#     trn_feat_x = scaler.fit_transform(trn_feat_x)
#     val_feat_x = scaler.transform(val_feat_x)
#     test_feat_x = scaler.transform(test_feat_x)

#     model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

#     model.fit(trn_feat_x, trn_feat_y.ravel())
    
#     val_preds = model.predict(val_feat_x)
#     test_preds = model.predict(test_feat_x)

#     val_mse = mean_squared_error(val_feat_y, val_preds)
#     val_r2 = r2_score(val_feat_y, val_preds)
    
#     test_mse = mean_squared_error(test_feat_y, test_preds)
#     test_r2 = r2_score(test_feat_y, test_preds)
    
#     print(f"Validation MSE: {val_mse}, R^2: {val_r2}")
#     print(f"Test MSE: {test_mse}, R^2: {test_r2}")

#     joblib.dump(model, os.path.join(args.dir_model, 'svr_model.joblib'))
#     joblib.dump(scaler, os.path.join(args.dir_model, 'scaler.joblib'))
from sklearn.ensemble import RandomForestRegressor

def train_and_evaluate(args):
    trn_feat_x, trn_feat_y = load_or_extract_features(args, 'trn')
    val_feat_x, val_feat_y = load_or_extract_features(args, 'val')
    test_feat_x, test_feat_y = load_or_extract_features(args, 'test')

    scaler = StandardScaler()
    trn_feat_x = scaler.fit_transform(trn_feat_x)
    val_feat_x = scaler.transform(val_feat_x)
    test_feat_x = scaler.transform(test_feat_x)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(trn_feat_x, trn_feat_y.ravel())
    
    val_preds = model.predict(val_feat_x)
    test_preds = model.predict(test_feat_x)

    val_mse = mean_squared_error(val_feat_y, val_preds)
    val_r2 = r2_score(val_feat_y, val_preds)
    
    test_mse = mean_squared_error(test_feat_y, test_preds)
    test_r2 = r2_score(test_feat_y, test_preds)
    
    print(f"Validation MSE: {val_mse}, R^2: {val_r2}")
    print(f"Test MSE: {test_mse}, R^2: {test_r2}")

    pearson_corr, _ = pearsonr(test_feat_y.ravel(), test_preds)
    print(f"Test Pearson Correlation: {pearson_corr}")
    if not os.path.exists(args.dir_model):
        os.makedirs(args.dir_model)

    joblib.dump(model, os.path.join(args.dir_model, 'rf_model.joblib'))
    joblib.dump(scaler, os.path.join(args.dir_model, 'scaler.joblib'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='en', type=str)
    parser.add_argument("--label_type1", default='pron', type=str, help='fluency, pron')
    parser.add_argument("--label_type2", default='prosody', type=str, help='articulation, prosody')

    parser.add_argument("--base_dim", default=1024, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--audio_len_max", default=200000, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--dir_list", default='', type=str)
    parser.add_argument("--dir_model", default='model_svr', type=str)

    args = parser.parse_args()

    if args.lang == 'en':
        args.base_model = 'facebook/wav2vec2-large-robust-ft-libri-960h'

    train_and_evaluate(args)


#python train_svm.py --dir_list="datasets_list" --dir_model="model_rf"