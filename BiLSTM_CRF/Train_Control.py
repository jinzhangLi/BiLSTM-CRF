from Train_model_build import BiLSTM_CRF
from Data_load import *

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
from tqdm import tqdm
from itertools import chain
import datetime
from sklearn import metrics

def train(epochs, train_dataloader, valid_dataloader, model, device,optimizer, batch_size, train_dataset, model_save_path):
    total_start = time.time()
    best_score = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        model.state = 'train'
        for step, (text, label, seq_len) in enumerate(train_dataloader, start=1):
            start = time.time()
            text = text.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)

            loss = model(text, label, seq_len)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch: [{epoch + 1}/{epochs}],'
                  f'  cur_epoch_finished: {step * batch_size / len(train_dataset) * 100:2.2f}%,'
                  f'  loss: {loss.item():2.4f},'
                  f'  cur_step_time: {time.time() - start:2.2f}s,'
                  f'  cur_epoch_remaining_time: {datetime.timedelta(seconds=int((len(train_dataloader) - step) / step * (time.time() - epoch_start)))}',
                  f'  total_remaining_time: {datetime.timedelta(seconds=int((len(train_dataloader) * epochs - (len(train_dataloader) * epoch + step)) / (len(train_dataloader) * epoch + step) * (time.time() - total_start)))}')

        # 每周期验证一次，保存最优参数
        score = evaluate(model, valid_dataloader, device, train_dataset)
        if score > best_score:
            print(f'score increase:{best_score} -> {score}')
            best_score = score
            torch.save(model, model_save_path)
        print(f'current best score: {best_score}')

def evaluate(model, valid_dataloader, device, train_dataset):
    # model.load_state_dict(torch.load('./model1.bin'))
    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for text, label, seq_len in tqdm(valid_dataloader, desc='eval: '):
            text = text.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(text, label, seq_len)
            all_label.extend([[train_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[train_dataset.label_map_inv[t] for t in l] for l in batch_tag])

    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in train_dataset.label_map.keys()]
    # 使用sklearn库得到F1分数
    f1 = metrics.f1_score(all_label, all_pred, average='macro', labels=sort_labels[:-3])

    print(metrics.classification_report(all_label, all_pred, labels=sort_labels[:-3], digits=3))
    return f1

def Train_control(train_path,valid_path,vocab_path,label_map_path,model_save_path,embedding_size,hidden_dim,epochs,batch_size,device):
    # 建立词表
    vocab = get_vocab(train_path, vocab_path)
    # 建立字典标签
    label_map = get_label_map(train_path, label_map_path)
    print("词表@标签构建完成")
    text_list=[]
    train_dataset = Mydataset(train_path, vocab, label_map,text_list,'train')
    valid_dataset = Mydataset(valid_path, vocab, label_map,text_list,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True,
                                  collate_fn=train_dataset.collect_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False,
                                  collate_fn=valid_dataset.collect_fn)
    model = BiLSTM_CRF(train_dataset, embedding_size, hidden_dim, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    train(epochs, train_dataloader, valid_dataloader, model, device,
                optimizer, batch_size, train_dataset, model_save_path)

if __name__=='__main__':
    torch.manual_seed(42)
    embedding_size = 128
    hidden_dim = 768
    epochs = 100
    batch_size = 32
    device = "cpu"

    # 训练集和验证集地址导入
    train_path = '/home/ModelTrain/NLP/Data/new_train.json'
    valid_path = '/home/ModelTrain/NLP/Data/new_dev.json'
    # 词表保存路径
    vocab_path = '/home/ModelTrain/NLP/Data/vocab.pkl'
    # 标签字典保存路径
    label_map_path = '/home/ModelTrain/NLP/Data/label_map.json'
    # 模型保存的路径
    model_save_path = '/home/ModelTrain/NLP/Data/BiLSTM+CRF.h5'
    Train_control(train_path,valid_path,vocab_path,label_map_path,model_save_path,embedding_size,hidden_dim,epochs,batch_size,device)
