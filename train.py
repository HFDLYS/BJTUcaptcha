import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchkeras import KerasModel, summary

import IPython
import numpy as np
from PIL import Image
from collections import OrderedDict
import os

from net import CRNN


charset = [' '] + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + ['+', '-', '×'] + ['=']

chardict = {}
i = 0
for char in charset:
    chardict[char] = i
    i += 1

class CapchaDataset(Dataset):
    def __init__(self, char_dict, data, labels, input_length, label_length):
        self.data = data
        self.labels = labels
        self.input_length = input_length
        self.label_length = label_length
        self.char_dict = char_dict
    
    def __getitem__(self, index):
        img = self.data[index]
        img = to_tensor(img)
        label = self.labels[index]
        label = list(label)
        for i in range(len(label)):
            label[i] = self.char_dict[label[i]]
        label = torch.tensor(label, dtype=torch.long)
        input_length = torch.full(size=(1, ), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1, ), fill_value=self.label_length, dtype=torch.long)
        return img, label, input_length, target_length
    
    def __len__(self):
        return len(self.data)



print('Loading data...🤔')

data = 'detect_failed/'
img_data = []
img_label = []
for i in os.listdir(data):
    img = Image.open(data + i)
    img_data.append(img)
    label = i[:-4]
    if len(label) == 4:
        label = ' ' + label + ' '
    elif len(label) == 5:
        label = label + ' '
    img_label.append(label)

batch_size = 2
width, height = 130, 42

dataset = CapchaDataset(chardict, img_data, img_label, 8, 6)
train_data, test_data = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print('Loading data OK!🤗')

n_classes = len(charset)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net= CRNN(n_classes, (3, height, width)).to(device)
net.load_state_dict(torch.load('model.pt'))
input = torch.zeros((batch_size, 3, height, width)).to(device)
out = net(input)
#torch.save(net.state_dict(), 'model.pt')
print(out.shape)


def decode_target(target):
    return ''.join([charset[i] for i in target[target != -1]]).replace(' ', '')


def decode(sequence):
    a = ''.join([charset[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != charset[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != charset[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

        
f = open('log.txt', 'w+', encoding='utf-8')

def eval_acc(targets, preds):
    preds_argmax = preds.detach().permute(1, 0, 2).argmax(dim=-1)
    targets = targets.cpu().numpy()
    preds_argmax = preds_argmax.cpu().numpy()
    a = np.array([decode_target(gt) == decode(pred) for gt,
                  pred in zip(targets, preds_argmax)])
    
    for gt, pred in zip(targets, preds_argmax):
        f.write(decode_target(gt)+" "+decode(pred) + '\n')
    
    return a.mean()

class StepRunner:
    def __init__(self, net, loss_fn, accelerator, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        
        images, targets, input_lengths, target_lengths = batch
        images, targets, input_lengths, target_lengths = images.cuda(), targets.cuda(), input_lengths.cuda(), target_lengths.cuda()
        #loss
        print(images.shape)
        preds = self.net(images)
        preds_log_softmax = F.log_softmax(preds, dim=-1)
        loss = F.ctc_loss(preds_log_softmax, targets, input_lengths, target_lengths)
        acc = eval_acc(targets,preds)
        

        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            
        all_loss = self.accelerator.gather(loss).sum()
        
        #losses
        step_losses = {self.stage+"_loss":
                       all_loss.item(),
                       self.stage+'_acc':acc}
        
        #metrics
        step_metrics = {}
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics
    
    
KerasModel.StepRunner = StepRunner 

model = KerasModel(net,
                   loss_fn=None,
                   optimizer = torch.optim.Adam(net.parameters(),lr = 0.0002, amsgrad=True),
                   )

model.load_ckpt('model.pt')

def display(x):
    IPython.display.clear_output(wait=True)
    IPython.display.display(x)


def display_fn(model):
    model.eval()
    right = True
    while right:
        image, target, input_length, label_length = dataset[0]
        output = model(image.unsqueeze(0).cuda())
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        right = (decode_target(target) == decode(output_argmax[0]))
        print('gt:', decode_target(target),' ','pred:', decode(output_argmax[0]))
    display(to_pil_image(image))
    

from torchkeras.kerascallbacks import VisDisplay
import accelerate

model.fit(
    train_data = train_loader,
    val_data= train_loader,
    ckpt_path='model.pt',
    epochs=40,
    patience=10,
    monitor="val_acc", 
    mode="max",
    plot = True,
    wandb = False,
    quiet = lambda epoch: epoch>5
)

f.close()
torch.save(model.net.state_dict(), 'last.pt')