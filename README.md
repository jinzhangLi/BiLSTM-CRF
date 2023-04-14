# BiLSTM-CRF
NER实体抽取任务，使用BiLSTM+CRF搭建模型

基于Anaconda使用torch+torchvision，训练方式使用CPU

BiLSTM+CRF文件夹为模型

->Train_XXX.py系列文件为训练时的数据加载，模型搭建，训练的文件，里面还有个训练一半的.bin模型，有框架有参数，Mode_Predict.py顾名思义就是给你load训练好的model做predict

Qlearning文件夹为模型

->HMM模型，.txt用来放概率矩阵的
