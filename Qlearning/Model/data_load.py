import json
from collections import defaultdict

# 读取json数据
def GetJosnData(train_path):
    #train_path='./data/cluener_public/train.json'
    json_data = []
    with open(train_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))
    return json_data

'''
json_data[0]数据为该格式：
{'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
 'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
'''

# 统计共有多少类别
def GetKind(json_data):
    n_classes = []
    for data in json_data:
        for i in data['label'].keys():  # 获取实体标签，如'name'，'company'
            if i not in n_classes:  # 将新的标签加入到列表中
               n_classes.append(i)
    return n_classes

# n_classes: ['name', 'company', 'game', 'organization', 'movie',
#             'address', 'position', 'government', 'scene', 'book']

# 设计tag2idx字典，对每个标签设计两种，如B-name、I-name，并设置其ID值
def set_tag2idx(n_classes):
    tag2idx = defaultdict()
    tag2idx['O'] = 0
    count = 1
    for n_class in n_classes:
        tag2idx['B-' + n_class] = count
        count += 1
        tag2idx['I-' + n_class] = count
        count += 1
    return count,tag2idx

def data_process(path):
    # 读取每一条json数据放入列表中
    # 由于该json文件含多个数据，不能直接json.loads读取，需使用for循环逐条读取
    json_data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))

    # json_data中每一条数据的格式为
    '''
    {'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
     'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
     '''

    # 将json文件处理成如下格式
    '''
    [['浙', '商', '银', '行', '企', '业', '信', '贷', '部', '叶', '老', '桂', '博', '士', '则', '从', '另', '一', 
    '个', '角', '度', '对', '五', '道', '门', '槛', '进', '行', '了', '解', '读', '。', '叶', '老', '桂', '认', 
    '为', '，', '对', '目', '前', '国', '内', '商', '业', '银', '行', '而', '言', '，'], 
    ['B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 
    'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
    '''
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
    return data