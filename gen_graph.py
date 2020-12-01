import argparse
import pickle
import json
import torch
import os
import numpy as np
import scipy.sparse as sp
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def parse_IEMOCAP():
    f = pickle.load(open('data_raw/IEMOCAP_features_raw.pkl', 'rb'), encoding='latin1')
    text = f[6]
    train_id = f[7]
    test_id = f[8]
    cat = f[2]
    speaker_id = f[1]
    dialogues = list()
    labels = list()
    roles = list()
    speakers = list()
    for tid in train_id:
        dialogue = list()
        label = list()
        speaker = list()
        speaker_dict = dict()
        for uttrance, uttr_cat, uttr_speaker in zip(text[tid], cat[tid], speaker_id[tid]):
            dialogue.append(uttrance)
            label.append(uttr_cat)
            if not uttr_speaker in speaker_dict:
                speaker_dict[uttr_speaker] = len(speaker_dict)
            speaker.append(speaker_dict[uttr_speaker])
        dialogues.append(dialogue)
        labels.append(label)
        roles.append(0)
        speakers.append(speaker)
    for tid in test_id:
        dialogue = list()
        label = list()
        speaker = list()
        speaker_dict = dict()
        for uttrance, uttr_cat, uttr_speaker in zip(text[tid], cat[tid], speaker_id[tid]):
            dialogue.append(uttrance)
            label.append(uttr_cat)
            if not uttr_speaker in speaker_dict:
                speaker_dict[uttr_speaker] = len(speaker_dict)
            speaker.append(speaker_dict[uttr_speaker])
        dialogues.append(dialogue)
        labels.append(label)
        roles.append(2)
        speakers.append(speaker)
    return dialogues, speakers, labels, roles

def parse_MELD():
    f = pickle.load(open('data_raw/MELD_features_raw.pkl', 'rb'))
    text = f[5]
    train_id = f[6]
    test_id = f[7]
    cat = f[2]
    speaker_id = f[1]
    dialogues = list()
    labels = list()
    roles = list()
    speakers = list()
    for tid in train_id:
        dialogue = list()
        label = list()
        speaker = list()
        for uttrance, uttr_cat, uttr_speaker in zip(text[tid], cat[tid], speaker_id[tid]):
            dialogue.append(uttrance)
            label.append(uttr_cat)
            speaker.append(np.argmax(np.array(uttr_speaker)))
        dialogues.append(dialogue)
        labels.append(label)
        roles.append(0)
        speakers.append(speaker)
    for tid in test_id:
        dialogue = list()
        label = list()
        speaker = list()
        for uttrance, uttr_cat, uttr_speaker in zip(text[tid], cat[tid], speaker_id[tid]):
            dialogue.append(uttrance)
            label.append(uttr_cat)
            speaker.append(np.argmax(np.array(uttr_speaker)))
        dialogues.append(dialogue)
        labels.append(label)
        roles.append(2)
        speakers.append(speaker)
    return dialogues, speakers, labels, roles

def to_coo(row, col, data, size):
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return sp.coo_matrix((data, (row, col)), shape=(size, size))

def parse_dailydialogue():
    emo_dict = {'no_emotion': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
    raw = list()
    for line in open('data_raw/dailydialog/train.json', 'r'):
        raw.append(json.loads(line))
    for line in open('data_raw/dailydialog/valid.json', 'r'):
        raw.append(json.loads(line))
    for line in open('data_raw/dailydialog/test.json', 'r'):
        raw.append(json.loads(line))
    dialogues = list()
    labels = list()
    roles = list()
    speakers = list()
    for diag in raw:
        dialogue = list()
        label = list()
        speaker = list()
        uttr_speaker = 0
        for uttr in diag['dialogue']:
            dialogue.append(uttr['text'])
            label.append(emo_dict[uttr['emotion']])
            speaker.append(uttr_speaker)
            uttr_speaker = 1 if uttr_speaker == 0 else 0
        dialogues.append(dialogue)
        labels.append(label)
        roles.append(2 if diag['fold']=='test' else 0)
        speakers.append(speaker)
    return dialogues, speakers, labels, roles

def gen_graph(path, dialogues, speakers, labels, roles):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').cuda()
    features_pooled = list()
    features_mean = list()
    ground_truth = list()
    split = list()
    tid = 0
    adj_full_row = list()
    adj_full_col = list()
    adj_full_data = list()
    adj_self_row = list()
    adj_self_col = list()
    adj_self_data = list()
    adj_past_row = list()
    adj_past_col = list()
    adj_past_data = list()
    adj_futr_row = list()
    adj_futr_col = list()
    adj_futr_data = list()
    with torch.no_grad():
        for dialogue, speaker, label, role in tqdm(zip(dialogues, speakers, labels, roles), total = len(dialogues)):
            # fully connected graph for one dialogue
            for i in range(len(dialogue)):
                tot = 0
                for j in range(len(dialogue)):
                    adj_full_row.append(tid + i)
                    adj_full_col.append(tid + j)
                    adj_full_data.append(1 / len(dialogue))
                    adj_full_row.append(tid + j)
                    adj_full_col.append(tid + i)
                    adj_full_data.append(1 / len(dialogue))
                    if speakers[i] == speakers[j]:
                        adj_self_row.append(tid + i)
                        adj_self_col.append(tid + j)
                        tot += 1
                        adj_self_row.append(tid + j)
                        adj_self_col.append(tid + i)
                    if j < i:
                        adj_past_row.append(tid + i)
                        adj_past_col.append(tid + j)
                        adj_past_data.append(1 / i)
                        adj_past_row.append(tid + j)
                        adj_past_col.append(tid + i)
                        adj_past_data.append(1 / i)
                    elif i < j:
                        adj_futr_row.append(tid + i)
                        adj_futr_col.append(tid + j)
                        adj_futr_data.append(1 / (len(dialogue) - i - 1))
                        adj_futr_row.append(tid + j)
                        adj_futr_col.append(tid + i)
                        adj_futr_data.append(1 / (len(dialogue) - i - 1))
                for _ in range(2 * tot):
                    adj_self_data.append(1 / tot)
            for i in range(len(dialogue)):
                tokens = tokenizer.encode(dialogue[i])
                inputs = torch.tensor(tokens).unsqueeze(0).cuda()
                outputs = model(inputs)
                features_pooled.append(outputs[1][0].cpu().detach().numpy())
                features_mean.append(torch.mean(outputs[0][0],dim=0).cpu().detach().numpy())
                ground_truth.append(label[i])
                split.append(role)
                tid += 1
    adj_full = to_coo(adj_full_row, adj_full_col, adj_full_data, tid)
    adj_self = to_coo(adj_self_row, adj_self_col, adj_self_data, tid)
    adj_past = to_coo(adj_past_row, adj_past_col, adj_past_data, tid)
    adj_futr = to_coo(adj_futr_row, adj_futr_col, adj_futr_data, tid)
    features_pooled = np.stack(features_pooled, axis=0)
    features_mean = np.stack(features_mean, axis=0)
    split = np.array(split, dtype=np.int32)
    ground_truth = np.array(ground_truth, dtype=np.int32)
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('data/' + path):
        os.mkdir('data/' + path)
    np.save('data/' + path + '/features_pooled.npy', features_pooled)
    np.save('data/' + path + '/features_mean.npy', features_mean)
    np.save('data/' + path + '/role.npy', split)
    np.save('data/' + path + '/label.npy', ground_truth)
    sp.save_npz('data/' + path + '/adj_full.npz', adj_full)
    sp.save_npz('data/' + path + '/adj_self.npz', adj_self)
    sp.save_npz('data/' + path + '/adj_past.npz', adj_past)
    sp.save_npz('data/' + path + '/adj_futr.npz', adj_futr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to generate graph.')

    parser.add_argument('-d', type=str, help='which dataset to parse')

    args = parser.parse_args()

    if args.d == 'IEMOCAP':
        dialogues, speakers, labels, roles = parse_IEMOCAP()
    elif args.d == 'MELD':
        dialogues, speakers, labels, roles = parse_MELD()
    elif args.d == 'dailydialogue':
        dialogues, speakers, labels, roles = parse_dailydialogue()
     
    gen_graph(args.d, dialogues, speakers, labels, roles)