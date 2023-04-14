import time
import torch
from tqdm import tqdm
from itertools import chain
import datetime
from sklearn import metrics

def train(epochs,train_dataloader,valid_dataloader,model,device,
          optimizer,batch_size,train_dataset,model_save_path):
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
        score = evaluate(model,valid_dataloader,device,train_dataset)
        if score > best_score:
            print(f'score increase:{best_score} -> {score}')
            best_score = score
            #model_save_path='./model.bin'
            torch.save(model.state_dict(), model_save_path)
        print(f'current best score: {best_score}')


def evaluate(model,valid_dataloader,device,train_dataset):
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

    print(metrics.classification_report(
        all_label, all_pred, labels=sort_labels[:-3], digits=3
    ))
    return f1