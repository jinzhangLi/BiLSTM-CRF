import os
import pickle
import json
import torch

#建立词表，每个词在输入到LSTM之前都需要转换成一个向量，这就是通常所说的词向量。
def get_vocab(data_path,vocab_path):
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if  os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as fp:
            vocab = pickle.load(fp)
    else:
        json_data = []
        # 加载数据集
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                json_data.append(json.loads(line))
        # 建立词表字典，提前加入'PAD'和'UNK'
        # 'PAD'：在一个batch中不同长度的序列用该字符补齐
        # 'UNK'：当验证集或测试集出现词表以外的词时，用该字符代替
        vocab = {'PAD': 0, 'UNK': 1}
        # 遍历数据集，不重复取出所有字符，并记录索引
        for data in json_data:
            for word in data['text']:  # 获取实体标签，如'name'，'company
                if word not in vocab:
                    vocab[word] = len(vocab)
        # vocab：{'PAD': 0, 'UNK': 1, '浙': 2, '商': 3, '银': 4, '行': 5...}
        # 保存成pkl文件
        with open(vocab_path, 'wb') as fp:
            pickle.dump(vocab, fp)

    # 翻转字表，预测时输出的序列为索引，方便转换成中文汉字
    # vocab_inv：{0: 'PAD', 1: 'UNK', 2: '浙', 3: '商', 4: '银', 5: '行'...}
    vocab_inv = {v: k for k, v in vocab.items()}
    return vocab, vocab_inv

def get_label_map(data_path,label_map_path):
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as fp:
            label_map = json.load(fp)
    else:
        # 读取json数据
        json_data = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                json_data.append(json.loads(line))
        # 统计共有多少类别
        n_classes = []
        for data in json_data:
            for label in data['label'].keys():  # 获取实体标签，如'name'，'company'
                if label not in n_classes:  # 将新的标签加入到列表中
                    n_classes.append(label)
        n_classes.sort()
        # n_classes: ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        # 设计label_map字典，对每个标签设计两种，如B-name、I-name，并设置其ID值
        label_map = {}
        for n_class in n_classes:
            label_map['B-' + n_class] = len(label_map)
            label_map['I-' + n_class] = len(label_map)
        label_map['O'] = len(label_map)
        # 对于BiLSTM+CRF网络，需要增加开始和结束标签，以增强其标签约束能力
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        label_map[START_TAG] = len(label_map)
        label_map[STOP_TAG] = len(label_map)
        # 将label_map字典存储成json文件
        with open(label_map_path, 'w', encoding='utf-8') as fp:
            json.dump(label_map, fp, indent=4)
    # {0: 'B-address', 1: 'I-address', 2: 'B-book', 3: 'I-book'...}
    label_map_inv = {v: k for k, v in label_map.items()}
    return label_map, label_map_inv

def data_process(path,is_train,text_lsit):
    # 读取每一条json数据放入列表中
    # 由于该json文件含多个数据，不能直接json.loads读取，需使用for循环逐条读取
    json_data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))
    if is_train=='train':
        data = []
        # 遍历json_data中每组数据
        for i in range(len(json_data)):
            # 将标签全初始化为'O'
            label = ['O'] * len(json_data[i]['text'])
            # 遍历'label'中几组实体，如样例中'name'和'company'
            for n in json_data[i]['label']:
                # 遍历实体中几组文本，如样例中'name'下的'叶老桂'（有多组文本的情况，样例中只有一组）
                for key in json_data[i]['label'][n]:
                    # 遍历文本中几组下标，如样例中[[9, 11]]（有时某个文本在该段中出现两次，则会有两组下标）
                    for n_list in range(len(json_data[i]['label'][n][key])):
                        # 记录实体开始下标和结尾下标
                        start = json_data[i]['label'][n][key][n_list][0]
                        end = json_data[i]['label'][n][key][n_list][1]
                        # 将开始下标标签设为'B-' + n，如'B-' + 'name'即'B-name'
                        # 其余下标标签设为'I-' + n
                        label[start] = 'B-' + n
                        label[start + 1: end + 1] = ['I-' + n] * (end - start)
            # 对字符串进行字符级分割
            # 英文文本如'bag'分割成'b'，'a'，'g'三位字符，数字文本如'125'分割成'1'，'2'，'5'三位字符
            texts = []
            for t in json_data[i]['text']:
                texts.append(t)
            # 将文本和标签编成一个列表添加到返回数据中
            data.append([texts, label])
    elif is_train=='dev':
        label=None
        data = []
        # 遍历json_data中每组数据
        for i in range(len(json_data)):
            texts = []
            for t in json_data[i]['text']:
                texts.append(t)
            # 将文本和标签编成一个列表添加到返回数据中
            data.append([texts,label])
    else:
        label=None
        data = []
        for i in range(len(text_lsit)):
            texts=[]
            for j in range(len(text_lsit[i])):
                texts.append(text_lsit[i][j])
            data.append([texts,label])
    return data

class Mydataset():
    def __init__(self, file_path, vocab, label_map,text_list,is_train):
        self.file_path = file_path
        # 数据预处理
        self.data = data_process(self.file_path,is_train,text_list)
        self.label_map, self.label_map_inv = label_map
        self.vocab, self.vocab_inv = vocab
        # self.data为中文汉字和英文标签，将其转化为索引形式
        self.examples = []
        if is_train=='train':
            for text, label in self.data:
                t = [self.vocab.get(t, self.vocab['UNK']) for t in text]
                l = [self.label_map[l] for l in label]
                self.examples.append([t, l])
        else:
            for text, label in self.data:
                t = [self.vocab.get(t, self.vocab['UNK']) for t in text]
                l=None
                self.examples.append([t, l])

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.data)

    def collect_fn(self, batch):
        # 取出一个batch中的文本和标签，将其单独放到变量中处理
        # 长度为batch_size，每个序列长度为原始长度
        text = [t for t, l in batch]
        label = [l for t, l in batch]
        # 获取一个batch内所有序列的长度，长度为batch_size
        seq_len = [len(i) for i in text]
        # 提取出最大长度用于填充
        max_len = max(seq_len)

        # 填充到最大长度，文本用'PAD'补齐，标签用'O'补齐
        text = [t + [self.vocab['PAD']] * (max_len - len(t)) for t in text]
        label = [l + [self.label_map['O']] * (max_len - len(l)) for l in label]

        # 将其转化成tensor，再输入到模型中，这里的dtype必须是long否则报错
        # text 和 label shape：(batch_size, max_len)
        # seq_len shape：(batch_size,)
        text = torch.tensor(text, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)

        return text, label, seq_len

    def Collect_Fn(self, batch):
        # 取出一个batch中的文本和标签，将其单独放到变量中处理
        # 长度为batch_size，每个序列长度为原始长度
        text = [t for t, l in batch]
        # 获取一个batch内所有序列的长度，长度为batch_size
        seq_len = [len(i) for i in text]
        # 提取出最大长度用于填充
        max_len = max(seq_len)
        # 填充到最大长度，文本用'PAD'补齐，标签用'O'补齐
        text = [t + [self.vocab['PAD']] * (max_len - len(t)) for t in text]
        # 将其转化成tensor，再输入到模型中，这里的dtype必须是long否则报错
        # text 和 label shape：(batch_size, max_len)
        # seq_len shape：(batch_size,)
        text = torch.tensor(text, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)
        return text, seq_len
