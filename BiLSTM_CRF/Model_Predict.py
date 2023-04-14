import torch
from tqdm import tqdm
from itertools import chain
import json

def predict(model,valid_dataloader,device,train_dataset,data_path):
    text_data=data_get(data_path)
    all_label,all_pred = [],[]
    with torch.no_grad():
        j=-1
        for text, label, seq_len in tqdm(valid_dataloader, desc='eval: '):
            j=j+1
            text = text.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(text, label, seq_len)
            predict=[[train_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)]
            for i in range(len(predict)):
                vector2text(text_data[j*42+i],predict[i])
            all_label.extend([[train_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[train_dataset.label_map_inv[t] for t in l] for l in batch_tag])
    #all_pred = list(chain.from_iterable(all_pred))
    return

def vector2text(string,predict):
    item = {"string": string, "entities": []}
    entity_name = ""
    flag = []
    #visit = False
    for char, tag in zip(string, predict):
        if tag[0] == "B":
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                item["entities"].append({"word": entity_name, "type": y[0]})
                flag.clear()
                entity_name = ""
            entity_name += char
            flag.append(tag[2:])
        elif tag[0] == "I":
            entity_name += char
            flag.append(tag[2:])
        else:
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                item["entities"].append({"word": entity_name, "type": y[0]})
                flag.clear()
            flag.clear()
            entity_name = ""

    if entity_name != "":
        x = dict((a, flag.count(a)) for a in flag)
        y = [k for k, v in x.items() if max(x.values()) == v]
        item["entities"].append({"word": entity_name, "type": y[0]})
    print(item)

def data_get(data_path):
    # 读书数据Json，存入一个列表，元素为输入的每一句话
    with open(data_path, 'r', encoding='utf-8') as fp:
        json_data=[json.loads(line) for line in fp]
    texts = [''.join([t for t in json_data[i]['text']]) for i in range(len(json_data))]
    return texts