import paddle
import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle_model import CRNN
from paddle.io import DataLoader,Dataset
from paddle.vision.transforms import Compose, Normalize

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import csv


charset = [' '] + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + ['+', '-', '*'] + ['=']
chardict = {}
i = 0
for char in charset:
    chardict[char] = i
    i += 1


class CapchaDataset(Dataset):
    def __init__(self, char_dict, data, labels, input_length, label_length):
        super(CapchaDataset, self).__init__()
        self.transform=Compose([Normalize(mean=[127.5], std=[127.5], data_format="CHW")])
        self.data = data
        self.labels = labels
        self.input_length = input_length
        self.label_length = label_length
        self.char_dict = char_dict

    def __getitem__(self, index):
        img = self.data[index]
        img = paddle.vision.transforms.to_tensor(img)
        label = self.labels[index]
        label = list(label)
        for i in range(len(label)):
            label[i] = self.char_dict[label[i]]
        label = paddle.to_tensor(label, dtype='int32')
        input_length = paddle.full(shape=(1,), fill_value=self.input_length, dtype='int64')
        target_length = paddle.full(shape=(1,), fill_value=self.label_length, dtype='int64')
        return img, [label, input_length, target_length]

    def __len__(self):
        return len(self.data)

print('Loading data...🤔')

data = 'datasets_ok/'
csv_path = os.path.join(data, 'captcha_mapping.csv')
img_data = []
img_label = []
with open(csv_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['image_name']
        label = row['label']
        image_path = os.path.join(data, image_name)
        img = Image.open(image_path)
        img_data.append(img)
        if len(label) == 4:
            label = ' ' + label + ' '
        elif len(label) == 5:
            label = label + ' '
        img_label.append(label)

batch_size = 20
width, height = 130, 42

dataset = CapchaDataset(chardict, img_data, img_label, 8, 6)
train_size = int(0.8 * len(dataset))
train_data, test_data = paddle.io.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print('Loading data OK!🤗')

n_classes = len(charset)

net= CRNN(n_classes, (3, height, width))


def decode_target(target):
    return ''.join([charset[i] for i in target[target != -1]]).replace(' ', '')


def decode(sequence):
    a = ''.join([charset[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != charset[0] and x != a[j + 1]])
    if len(s) == 0:
        return ''
    if a[-1] != charset[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


log = open('log.txt', 'w+', encoding='utf-8')


def eval_acc(targets, preds):

    preds_argmax = preds.detach().transpose([1, 2,0]).argmax(axis=1)
    targets = targets.numpy()
    preds_argmax = preds_argmax.numpy()
    a = paddle.to_tensor([(1.0 if decode_target(gt) == decode(pred) else 0.0) for gt,pred in zip(targets, preds_argmax)])
    for gt, pred in zip(targets, preds_argmax):
        log.write(decode_target(gt) + " " + decode(pred) + '\n')
    return a.mean()

def train(model,epochs=10):
    model.train()
    optim = paddle.optimizer.Adam(
        learning_rate=0.0002,
        parameters=model.parameters(),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=5.0)
    )
    """
    第118行的作用？
    """
    # 用Adam作为优化函数
    for epoch in range(epochs):
        acc1=[]
        for batch_id, data in enumerate(train_loader()):
            img = data[0]
            label = data[1][0]
            input_lengths = data[1][1].squeeze()
            label_lengths = data[1][2].squeeze()
            predicts = model(img)
            preds_log_softmax = F.log_softmax(predicts, axis=-1)
            loss = F.ctc_loss(preds_log_softmax, label, input_lengths, label_lengths)
            acc = eval_acc(label,predicts)
            acc1.append(acc)
            loss.backward()
            if batch_id % 1 == 0:
                print(
                    "epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
                        epoch, batch_id, loss.numpy(), acc.numpy()
                    )
                )
            optim.step()
            optim.clear_grad()
        print("epoch: {}, acc is: {}".format(epoch,paddle.to_tensor(acc1).mean().numpy()))

model = CRNN(len(charset))
train(model)

paddle.save(model.state_dict(), 'model2.pdparams')
model.set_state_dict(paddle.load('model2.pdparams'))
# 加载测试数据集
def test(model):
    model.eval()
    batch_size = 64
    acc1 = []
    for batch_id, data in enumerate(test_loader()):
        img = data[0]
        label = data[1][0]
        input_lengths = data[1][1].squeeze()  # 移除单维度 [batch_size,1] => [batch_size]
        label_lengths = data[1][2].squeeze()
        predicts = model(img)
        preds_log_softmax = F.log_softmax(predicts, axis=-1)
        loss = F.ctc_loss(preds_log_softmax, label, input_lengths, label_lengths)
        acc = eval_acc(label, predicts)
        acc1.append(acc)
        if batch_id % 20 == 0:
            print(
                "batch_id: {}, loss is: {}, acc is: {}".format(
                    batch_id, loss.numpy(), acc.numpy()
                )
            )
    print("acc is: {}".format(paddle.to_tensor(acc1).mean().numpy()))
test(model)
log.close()