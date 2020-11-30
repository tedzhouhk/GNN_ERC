import argparse
import pickle
import json
import numpy as np

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
     
    import pdb; pdb.set_trace()