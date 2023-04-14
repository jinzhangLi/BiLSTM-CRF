from Train_model_build import BiLSTM_CRF
import torch
import Train_data_load as dl
from Train_data_load import Mydataset
from torch.utils.data import DataLoader
import torch.optim as optim
from BiLSTM_CRF import Train

torch.manual_seed(3407)

embedding_size = 128
hidden_dim = 768
epochs = 100
batch_size = 32
device = "cpu"

#训练集和验证集地址导入
train_path='/tmp/pycharm_project_367/Data/train.json'
valid_path='/tmp/pycharm_project_367/Data/dev.json'

# 词表保存路径
vocab_path = '/tmp/pycharm_project_367/Data/vocab.pkl'
# 标签字典保存路径
label_map_path = '/tmp/pycharm_project_367/Data/label_map.json'

#建立词表
vocab=dl.get_vocab(train_path,vocab_path)
#建立字典标签
label_map=dl.get_label_map(train_path,label_map_path)

train_dataset = Mydataset(train_path, vocab, label_map)
valid_dataset = Mydataset(valid_path, vocab, label_map)

print('训练集长度:', len(train_dataset))
print('验证集长度:', len(valid_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True,
                              collate_fn=train_dataset.collect_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False,
                              collate_fn=valid_dataset.collect_fn)

model = BiLSTM_CRF(train_dataset, embedding_size, hidden_dim, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

model_save_path='/tmp/pycharm_project_367/BiLSTM_CRF/BiLSTM+CRF.h5'

Train.train(epochs, train_dataloader, valid_dataloader, model, device,
            optimizer, batch_size, train_dataset, model_save_path)
