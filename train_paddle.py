import time

import paddle
import paddle.nn.functional as F
from paddle_model import CRNN
from paddle.io import DataLoader,Dataset
from paddle.vision.transforms import Compose, Normalize

from visualdl import LogWriter
import numpy as np
from PIL import Image
import os
import csv

charset = [' '] + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + ['+', '-', '*'] + ['=']
chardict = {}
i = 0
for char in charset:
    chardict[char] = i
    i += 1

model_name = 'crnn_' + time.strftime("%Y%m%d_%H%M%S")

paddle.device.set_device('gpu:0')
class CapchaDataset(Dataset):
    def __init__(self, char_dict, path, labels, input_length):
        super(CapchaDataset, self).__init__()
        self.transform=Compose([Normalize(mean=[127.5], std=[127.5], data_format="CHW")])
        self.path = path
        self.labels = labels
        self.input_length = input_length
        self.char_dict = char_dict

    def __getitem__(self, index):
        img_p = self.path[index]
        img_o = Image.open(img_p)
        img = paddle.vision.transforms.to_tensor(img_o)
        label = self.labels[index]
        label_length = len(label)
        if len(label) == 4:
            label = label + '  '
        elif len(label) == 5:
            label = label + ' '
        label = list(label)
        for i in range(len(label)):
            label[i] = self.char_dict[label[i]]
        label = paddle.to_tensor(label, dtype='int32')
        input_length = paddle.full(shape=(1,), fill_value=self.input_length, dtype='int64')
        target_length = paddle.full(shape=(1,), fill_value=label_length, dtype='int64')
        img_o.close()
        return img, [label, input_length, target_length]

    def __len__(self):
        return len(self.path)

print('Loading data...🤔')

data = 'datasets_final_ok/'
csv_path = os.path.join(data, 'captcha_mapping.csv')
img_path = []
img_label = []
with open(csv_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['image_name']
        label = row['label']
        image_path = os.path.join(data, image_name)
        img_path.append(image_path)
        img_label.append(label)

batch_size = 32
width, height = 130, 42
input_shape = (3, height, width)

n_classes = len(charset)
net = CRNN(n_classes, input_shape)
seq_len = net.get_seq_len()

dataset = CapchaDataset(chardict, img_path, img_label, seq_len)

total_size = len(dataset)
train_size = int(0.7 * total_size)           # 70% 训练集
val_size = int(0.2 * total_size)             # 20% 验证集
test_size = total_size - train_size - val_size  # 剩余 10% 测试集

train_data = paddle.io.Subset(dataset, indices=list(range(0, train_size)))
val_data = paddle.io.Subset(dataset, indices=list(range(train_size, train_size + val_size)))
test_data = paddle.io.Subset(dataset, indices=list(range(train_size + val_size, total_size)))

# 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print('Loading data OK!🤗')

def decode_target(target):
    return ''.join([charset[i] for i in target[target != -1]]).replace(' ', '')

def decode(sequence):
    decoded = []
    prev_char = None
    for x in sequence:
        char = charset[x]
        if char != prev_char and char != ' ':
            decoded.append(char)
        prev_char = char
    return ''.join(decoded)

# log = open('log.txt', 'w+', encoding='utf-8')
log_writer = LogWriter(logdir="./log")

def eval_acc(targets, preds):
    preds_argmax = preds.detach().transpose([1, 2,0]).argmax(axis=1)
    targets = targets.numpy()
    preds_argmax = preds_argmax.numpy()
    a = paddle.to_tensor([
        (1.0 if decode_target(gt) == decode(pred) else 0.0)
        for gt,pred in zip(targets, preds_argmax)
    ])
    # for gt, pred in zip(targets, preds_argmax):
    #     log.write(decode_target(gt) + " " + decode(pred) + '\n')
    return a.mean()

def test_eval_acc(targets, preds):
    preds_argmax = preds.detach().transpose([1, 2,0]).argmax(axis=1)
    targets = targets.numpy()
    preds_argmax = preds_argmax.numpy()
    a = paddle.to_tensor([
        (1.0 if decode_target(gt) == decode(pred) else 0.0)
        for gt,pred in zip(targets, preds_argmax)
    ])
    # for gt, pred in zip(targets, preds_argmax):
    #     log.write(decode_target(gt) + " " + decode(pred) + '\n')
    return a.mean()

def validate(model):
    model.eval()
    acc1 = []
    for batch_id, data in enumerate(val_loader()):
        img = data[0]
        label = data[1][0]
        predicts = model(img)
        acc = eval_acc(label, predicts)
        acc1.append(acc)
    val_acc = paddle.to_tensor(acc1).mean().numpy()
    model.train()
    return val_acc

def train(model, epochs=40, patience=5, stopping_acc=0.005):
    model.train()
    optim = paddle.optimizer.Adam(
        learning_rate=0.0002,
        parameters=model.parameters(),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=5.0)
    )

    # 早停法
    best_val_acc = 0.0
    waited_epoch = 0
    loss = 0

    for epoch in range(epochs):
        acc1=[]
        for batch_id, data in enumerate(train_loader()):
            img = data[0]
            label = data[1][0]
            input_lengths = data[1][1].squeeze()
            label_lengths = data[1][2].squeeze()
            predicts = model(img)
            preds_log_softmax = F.log_softmax(predicts, axis=-1)
            loss = F.ctc_loss(preds_log_softmax, label, input_lengths, label_lengths, blank=0, reduction='mean', norm_by_times=False)
            loss = paddle.where(paddle.isnan(loss), paddle.zeros_like(loss), loss)  # 处理NaN
            loss = paddle.where(paddle.isinf(loss), paddle.zeros_like(loss), loss)  # 处理Inf

            acc = eval_acc(label,predicts)
            acc1.append(acc)
            loss.backward()
            if batch_id % 40 == 0:
                print(
                    "epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
                        epoch, batch_id, loss.numpy(), acc.numpy()
                    )
                )
            optim.step()
            optim.clear_grad()
        epoch_acc = paddle.to_tensor(acc1).mean().numpy().item()
        val_acc = validate(model).item()

        if val_acc - best_val_acc > stopping_acc:
            waited_epoch = 0
            best_val_acc = val_acc
            paddle.save(model.state_dict(), './model/best.pdparams')
            paddle.save(optim.state_dict(), './model/best.pdopt')
        else:
            waited_epoch += 1

        print("Waited {} epochs".format(waited_epoch))
        print("epoch: {}, acc is: {}, val_acc is: {}, best_val_acc is: {}".format(epoch, epoch_acc, val_acc, best_val_acc))
        log_writer.add_scalar(tag="loss", step=epoch, value=loss.numpy().item())
        log_writer.add_scalar(tag="val_acc", step=epoch, value=val_acc)
        log_writer.add_scalar(tag="epoch_acc", step=epoch, value=epoch_acc)
        if waited_epoch >= patience:
            print("early stopped at epoch {}".format(epoch))
            break

model = CRNN(len(charset))
train(model)

model.set_state_dict(paddle.load('model/best.pdparams'))

# 注意input_shape
paddle.onnx.export(
    model,
    path='./model/' + model_name,
    input_spec= [
        paddle.static.InputSpec(shape=[1, input_shape[0], input_shape[1], input_shape[2]])
    ]
)
print("ONNX 模型已导出为 ./model/" + model_name + ".onnx")

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
        if batch_id % 40 == 0:
            print(
                "batch_id: {}, loss is: {}, acc is: {}".format(
                    batch_id, loss.numpy(), acc.numpy()
                )
            )
    print("acc is: {}".format(paddle.to_tensor(acc1).mean().numpy()))

test(model)
# log.close()