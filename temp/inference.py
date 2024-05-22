# Copyright 2022 kakaoenterprise  heize.s@kakaoenterprise.com
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# proninciation score inference & pcc score calculation
# input: wav list, output: pronunciation score, pcc score 

import os, argparse
import torch
import audiofile
from transformers import Wav2Vec2ForCTC
from score_model import MLP
import numpy as np
from scipy.stats import pearsonr


def open_file(filename):
    f = open(filename)
    filelist = f.readlines()
    f.close()
    return filelist


def inference():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='en', type=str)
    parser.add_argument("--label_type1", default='pron', type=str, help='fluency, pron')
    parser.add_argument("--label_type2", default='prosody', type=str, help='articulation, prosody')


    parser.add_argument("--dir_list", default='data_prepare', type=str)
    parser.add_argument("--dir_model", default='model', type=str)
    parser.add_argument("--data_type", default='test', type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--audio_len_max", default=200000, type=int)
    parser.add_argument("--dir_data", type=str, default='/data/project/rw/nia_db/nia_2022_012/pron_data/audio', help="")

    args = parser.parse_args()

    fname_list = os.path.join(args.dir_list, f'lang_{args.lang}', f'{args.label_type1}_{args.data_type}.list')
    filelist = open_file(fname_list)
    data_len = len(filelist)


    dir_model = os.path.join(args.dir_model, f'lang_{args.lang}', f'{args.label_type1}_{args.label_type2}_checkpoint.pt')
    score_model = torch.load(dir_model).to(args.device)
    score_model.eval()

    if args.lang == 'en':
        base_model_name = 'facebook/wav2vec2-large-robust-ft-libri-960h'
    elif args.lang == 'jp':
        base_model_name = 'NTQAI/wav2vec2-large-japanese'
    elif args.lang == 'zh':
        base_model_name = 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn'
    elif args.lang == 'de':
        base_model_name = 'facebook/wav2vec2-large-xlsr-53-german'
    elif args.lang == 'es':
        base_model_name = 'facebook/wav2vec2-large-xlsr-53-spanish'
    elif args.lang == 'fr':
        base_model_name = 'facebook/wav2vec2-large-xlsr-53-french' 
    elif args.lang == 'ru':
        base_model_name = 'bond005/wav2vec2-large-ru-golos'

    f = open(f'result_{args.lang}_{args.label_type1}_{args.label_type2}_{args.data_type}.txt', 'w')
    f.write(f'{args.lang}, {args.label_type1}, {args.label_type2}, base_model: {base_model_name}')

    base_model = Wav2Vec2ForCTC.from_pretrained(base_model_name).to(args.device) # load wav2vec2 model
    test_labels, test_preds = [], []
    for idx, line in enumerate(filelist):

        try:
            if args.label_type2 == 'articulation':
                fname, score, _, text = line.split('\t')
            else:
                fname, _, score, text = line.split('\t')

            fname = fname.split('/audio/')[-1]
            fname = os.path.join(args.dir_data, fname)
            x, sr = audiofile.read(fname)



            x = torch.tensor(x[:min(x.shape[-1], args.audio_len_max)], device =args.device).reshape(1, -1)
            feat_x = base_model(x, output_attentions=True, output_hidden_states=True, return_dict=True).hidden_states[-1]
            feat_x = torch.mean(feat_x, axis = 1)

            pred_score = score_model(feat_x).cpu().detach().numpy() # inference pronunciation score
            pred_score = np.clip(pred_score, 0, 5)
            test_preds.append(pred_score[0][0])
            test_labels.append(float(score))

            f.write(f'{fname} {round(float(pred_score[0][0]), 2)}\n')

        except:
            continue

    test_pcc = pearsonr(test_labels, test_preds)[0]
    f.write(f'lang_{args.lang}/{args.label_type1}_{args.data_type}.list\n')
    print(f'lang_{args.lang}/{args.label_type1}_{args.data_type}.list')
    f.write(f'{args.label_type2} pcc : {round(test_pcc, 2)}')
    print(f'{args.label_type2} pcc : {round(test_pcc, 2)}')
    f.close()
        





if __name__=="__main__":
    inference()