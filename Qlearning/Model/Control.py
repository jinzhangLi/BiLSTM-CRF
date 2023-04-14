from data_load import data_process
from model_build import HMM_model
from sklearn import metrics
from itertools import chain
import data_load as dl

#训练集和验证集地址导入
train_path='/tmp/pycharm_project_367/Data/train.json'
valid_path='/tmp/pycharm_project_367/Data/dev.json'

json_data=dl.GetJosnData(train_path)
n_classes=dl.GetKind(json_data)
count,tag2idx=dl.set_tag2idx(n_classes)

train_data = data_process(train_path)
valid_data = data_process(valid_path)

print('训练集长度:', len(train_data))
print('验证集长度:', len(valid_data))

model = HMM_model(tag2idx)
model.train(train_data)
model.load_paramters()

model.predict('浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言')
print()
y_pred = model.valid(valid_data)
y_true = [data[1] for data in valid_data]

# 排好标签顺序输入，否则默认按标签出现顺序进行排列
sort_labels = [k for k in tag2idx.keys()]

y_true = list(chain.from_iterable(y_true))
y_pred = list(chain.from_iterable(y_pred))

# 打印详细分数报告，包括precision(精确率)，recall(召回率)，f1-score(f1分数)，support(个数)，digits=3代表保留3位小数
print(metrics.classification_report(
    y_true, y_pred, labels=sort_labels[1:]))