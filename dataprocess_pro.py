from itertools import combinations
import os
import csv
import json
import logging
import random
from transformers import RobertaTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = RobertaTokenizer.from_pretrained('/home/ubuntu/models/roberta-large')



def merge_data(data_list):
    # Step 1: Initialize a dictionary to group data by 'text'
    grouped_data = defaultdict(list)
    for item in data_list:  # Convert list to tuple to use as dictionary key
        text = tuple(item[0])  # Convert list to tuple to use as dictionary key
        emotion_sentence = item[2]  # Current emotion sentence
        key = (text, emotion_sentence)
        grouped_data[key].append(item)

    # Step 2: Prepare the final merged output
    result = []
    doc_id = 1
    for (text, emotion_sentence), items in grouped_data.items():
        doc_len = len(text)
        clauses = []
        pairs = set()

        # Collect unique clauses and their indices
        clause_map = {}
        for idx, clause_text in enumerate(text):
            clause_map[clause_text] = idx + 1
            clauses.append({
                "clause_id": idx + 1,
                "clause": clause_text,
                "emotion_category": items[0][1][idx],  # Emotion category from the first item
                "emotion_token": "null"
            })

        # Collect pairs

        for item in items:
            try:
                emotion_index = text.index(item[2]) + 1
                reason_index = text.index(item[4]) + 1
                if item[6][0] == 1:
                    pairs.add((emotion_index, reason_index))
            except:
                continue


        # Convert set of pairs to list of lists
        pairs = list(pairs)

        # Create the merged dictionary entry
        merged_entry = {
            "doc_id": doc_id,
            "doc_len": doc_len,
            "clauses": clauses,
            "pairs": pairs
        }
        result.append(merged_entry)
        doc_id += 1

    return result


from collections import defaultdict

# Example data


# Function to find the index of a sentence in the text
def find_index(text, sentence):
    for i, s in enumerate(text):
        if s == sentence:
            return i+1
    return -1  # If the sentence is not found, return -1

def md(data):
# Step 1: Organize data by text and emotion sentence
    merged_data = defaultdict(lambda: {'pairs': [], 'clauses': [], 'candidate_index': [], 'doc_len': 0})
    text_emotion_to_id = defaultdict(int)

    for idx, (text, emotions, emotion_sentence, _, candidate_sentence, _, label,len_dia) in enumerate(data):
        text_key = (tuple(text), emotion_sentence)

        if text_key not in text_emotion_to_id:
            doc_id = len(text_emotion_to_id) + 1
            text_emotion_to_id[text_key] = doc_id
        else:
            doc_id = text_emotion_to_id[text_key]

        text_data = merged_data[doc_id]
        text_data['doc_len'] = len(text)

        # Find indices of sentences
        emotion_index = find_index(text, emotion_sentence)
        reason_index = find_index(text, candidate_sentence)

        if label[0] == 1 and [emotion_index, reason_index] not in  text_data['pairs']:
            text_data['pairs'].append([emotion_index, reason_index])

        if reason_index not in text_data['candidate_index'] or reason_index==-1:
            text_data['candidate_index'].append(reason_index)

        # Create the list of clauses
        text_data['clauses'] = [
            {'clause_id': i + 1, 'clause': clause, 'emotion_category': emotions[i], 'emotion_token': 'null'}
            for i, clause in enumerate(text)
        ]
        text_data['len_dia']=len_dia

    # Convert merged_data to desired format
    result = [
        {
            'doc_id': doc_id,
            'doc_len': data['doc_len'],
            'candidate_index': data['candidate_index']+[-1]*(data['len_dia']-len(data['candidate_index'])),
            'pairs': data['pairs'],
            'clauses': data['clauses']
        }
        for doc_id, data in merged_data.items()
    ]
    return result


emotion_mapping = {
    'RECCON': {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'},
}

data_name='RECCON'
random.seed(1)
dataset = [[], [], []]
emotionsdic = emotion_mapping[data_name]
emotion_indexes = {v: k + 1 for k, v in emotion_mapping[data_name].items()}

input_dir="./"

for idx, data_type in enumerate(["train","valid",'test']):
    input_file = os.path.join(input_dir, '{}.json'.format(data_type))
    neg_examples = []

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data_type == 'train':
        random.shuffle(data)

    for dialog in data:
        speakers = {}
        context = []
        emotions = []
        context_num = 0
        len_dia=len(dialog)
        for ut in dialog:

            speaker_id = speakers.get(ut['speaker'])
            if speaker_id is None:
                speaker_id = "S{}".format(len(speakers) + 1)
                speakers[ut['speaker']] = speaker_id

            info = speaker_id + " " + emotionsdic[ut['emotion']] + " " + ut["text"].lower()
            context.append(info)
            emotions.append(int(ut['emotion']) + 1)
            context_num += 1
            assert context_num == len(context), "{} \n {}".format(context_num, context)

            if 'cause' not in ut.keys():
                continue

            cur_len = 0
            text_a = []
            emotion_a = []
            i = len(context) - 1
            while i >= 0 and len(tokenizer.encode(''.join(str(item) for item in text_a), add_special_tokens=False)) <= 410:
                text_a.insert(0, context[i])
                emotion_a.insert(0, emotions[i])
                i -= 1
                cur_len += len(context[i])
            if len(tokenizer.encode(''.join(str(item) for item in text_a), add_special_tokens=False)) > 490:
                text_a.pop(0)
                emotion_a.pop(0)

            for i in range(len(context)):
                emotion_b = int(ut['emotion']) + 1
                emotion_c =emotion_indexes[context[i].split(' ')[1]]
                if i + 1 in list(ut['cause']):
                    example = [text_a, emotion_a, info, emotion_b, context[i], emotion_c, [1],len_dia]
                else:
                    example = [text_a, emotion_a, info, emotion_b, context[i], emotion_c, [0],len_dia]

                dataset[idx].append(example)


traindata=md(dataset[0])
valdata = md(dataset[1])
testdata = md(dataset[2])
logger.info(
    "Train set:{} \t validation set:{} \t test set:{}.".format(len(dataset[0]), len(dataset[1]),len(dataset[2])))
logger.info(
    "Train set:{} \t validation set:{} \t test set:{}.".format(len(traindata), len(valdata),len(testdata)))

for dicts_list in [traindata,valdata,testdata]: # [traindata,valdata,testdata]:
    total_doc=0
    for d in dicts_list:
        if 'candidate_index' in d:
            total_doc += len(d['candidate_index'])

    print("Total 'doc' value:", total_doc)


for dicts_list in [traindata,valdata,testdata]: # [traindata,valdata,testdata]:
    total_doc = 0
    for d in dicts_list:
        if 'doc_len' in d and isinstance(d['doc_len'], (int, float)):
            if d['doc_len'] != len(d['candidate_index'])-d['candidate_index'].count(-1):
                print(d["doc_id"])




for dicts_list in [traindata,valdata,testdata]: # [traindata,valdata,testdata]:
    total_doc=0
    for d in dicts_list:
        if 'candidate_index' in d:
            total_doc += d['candidate_index'].count(-1)
    print("Total 'doc' value:", total_doc)


for dicts_list in [traindata,valdata,testdata]: # [traindata,valdata,testdata]:
    total_doc=0
    for d in dicts_list:
        if 'doc_len' in d and isinstance(d['doc_len'], (int, float)):
            total_doc += d['doc_len']
    print("Total 'doc' value:", total_doc)

for type in [ "train","val","test"]:
    with open("{}_pro.json".format(type), 'w') as f:
        json.dump(eval("{}data".format(type)), f)