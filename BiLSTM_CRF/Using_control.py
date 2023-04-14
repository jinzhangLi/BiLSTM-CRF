import time
from Train_model_build import BiLSTM_CRF
import Train_data_load as dl
from Train_data_load import Mydataset
from torch.utils.data import DataLoader
import torch
import Model_Predict as MP

start=time.time()

#远端服务器上文件的地址路径
# data_path='/tmp/pycharm_project_367/Data/dev.json'
# model_path='/tmp/pycharm_project_367/BiLSTM_CRF/BiLSTM+CRF.bin'
# train_path='/tmp/pycharm_project_367/Data/train.json'
# vocab_path = '/tmp/pycharm_project_367/Data/vocab.pkl'
# label_map_path = '/tmp/pycharm_project_367/Data/label_map.json'
# valid_path='/tmp/pycharm_project_367/Data/dev.json'

#本地地址
data_path='E:/NER_model/Data/train.json'
model_path='E:/NER_model/BiLSTM_CRF/BiLSTM+CRF.bin'
train_path='E:/NER_model/Data/train.json'
vocab_path = 'E:/NER_model/Data/vocab.pkl'
label_map_path ='E:/NER_model/Data/label_map.json'
valid_path='E:/NER_model/Data/dev.json'

device="cpu"
batch_size = 32
embedding_size = 128
hidden_dim = 768

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


#创建模型后，load_state_dict加载模型参数,再进行预测
model = BiLSTM_CRF(train_dataset, embedding_size, hidden_dim, device).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
model.state = 'eval'
MP.predict(model,valid_dataloader,device,train_dataset,data_path)

end=time.time()

print("******Using Time:"+str(end-start)+"******")