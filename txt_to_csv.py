import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

train = pd.DataFrame(open('train_1000.txt', 'r+').readlines(), columns = ['question'])
test = pd.DataFrame(open('test_500.txt', 'r+').readlines(), columns = ['question'])

train['qtype'] = train.question.apply(lambda x: x.split(' ', 1)[0])
train['question'] = train.question.apply(lambda x: x.split(' ', 1)[1])
train['qtype_coarse'] = train.qtype.apply(lambda x: x.split(':')[0])
train['qtype_fine'] = train.qtype.apply(lambda x: x.split(':')[1])
test['qtype'] = test.question.apply(lambda x: x.split(' ', 1)[0])
test['question'] = test.question.apply(lambda x: x.split(' ', 1)[1])
test['qtype_coarse'] = test.qtype.apply(lambda x: x.split(':')[0])
test['qtype_fine'] = test.qtype.apply(lambda x: x.split(':')[1])

train.to_csv('train_1000.csv')
test.to_csv('test_500.csv')