import time
import torch
from torch.utils.data import DataLoader
from itertools import chain
import json

from Data_load import *

def txt_read(file_local):
    file = open(file_local, 'r', encoding='utf-8')
    content = file.read()
    file.close()
    return content

def text_pre(content):
    sentence_lists = [sentence for sentence in re.split(r'[？?！!。\n\r]', content) if sentence]
    sentence_list = [re.split('[,，:：；]', sentence_lists[i]) for i in range(len(sentence_lists))]
    paragraph_index = [[i + 1 for _ in sentence_list[i]] for i in range(len(sentence_list))]
    paragraph_index = [paragraph_index[i][j] for i in range(len(paragraph_index)) for j in
                       range(len(paragraph_index[i]))]
    sentence_list = [sentence_list[i][j] for i in range(len(sentence_list)) for j in range(len(sentence_list[i]))]
    return sentence_list, paragraph_index, sentence_lists

def vector2text(string,predict):
    # 标签转录BIO格式
    item = {"string": string, "entities": []}
    entity_name = ""
    flag,items= [],[]
    visit = False
    for char, tag in zip(string, predict):
        if tag[0] == "B":
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                item["entities"].append({"word": entity_name, "type": y[0]})
                items.append([entity_name, y[0]])
                flag.clear()
                entity_name = ""
            visit = True
            entity_name += char
            flag.append(tag[2:])
        elif tag[0] == "I" and visit:
            entity_name += char
            flag.append(tag[2:])
        else:
            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                item["entities"].append({"word": entity_name, "type": y[0]})
                items.append([entity_name, y[0]])
                flag.clear()
            flag.clear()
            visit = False
            entity_name = ""
    if entity_name != "":
        x = dict((a, flag.count(a)) for a in flag)
        y = [k for k, v in x.items() if max(x.values()) == v]
        item["entities"].append({"word": entity_name, "type": y[0]})
        items.append([entity_name,y[0]])
    return items

def predict(vocab_path,label_map_path,data_path,model_path,device,model_state,text_list):
    start=time.time()
    # 建立词表
    vocab = get_vocab('0', vocab_path)
    # 建立字典标签
    label_map = get_label_map('0', label_map_path)
    global label_map_index
    for i in range(len(label_map)):
        label_map_index=label_map[i]
    dataset = Mydataset(data_path, vocab, label_map, text_list,'use')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False, shuffle=False,
                            collate_fn=dataset.Collect_Fn)
    model=torch.load(model_path,map_location=device)
    model.eval()
    model.state=model_state
    result=[]
    with torch.no_grad():
        k = -1
        for text, seq_len in dataloader:
            k=k+1
            text = text.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(text,None, seq_len)
            predict=[[label_map_index[t] for t in l] for l in batch_tag]
            for i in range(len(predict)):
                items=vector2text(text_list[k*len(predict)+i], predict[i])
                result.append([text_list[k*len(predict)+i]]+items)
    for i in range(len(result)):
        print(result[i])
    end = time.time()
    time_s=end-start
    print("******Using Time:"+str(time_s)+"******")

# 调用 load.h5
vocab_path = '/home/ModelTrain/NLP/Data/vocab.pkl'
label_map_path = '/home/ModelTrain/NLP/Data/label_map.json'
data_path = '/home/ModelTrain/NLP/Data/new_test.json'
model_path = '/home/ModelTrain/NLP/Data/BiLSTM+CRF.h5'
device = "cpu"
model_state='eval'

text_path="/home/ModelTrain/Data/test0608.txt"
sentence_list, paragraph_index, sentence_lists=text_pre(txt_read(text_path))
sentence_list=[sentence_list[i] for i in range(len(sentence_list)) if sentence_list[i]!='']
time_s=predict(vocab_path,label_map_path,data_path,model_path,device,model_state,text_list)

#直接下载一个模型和参数在一起的.h5


