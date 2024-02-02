import requests
import re
import bs4
from PIL import Image
from net import CRNN
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import requests
import re
import bs4
from PIL import Image

charset = [' '] + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + ['+', '-', '×'] + ['=']

chardict = {}
i = 0
for char in charset:
    chardict[char] = i
    i += 1
n_classes = len(charset)

width, height = 130, 42
batch_size = 1
device = torch.device('cpu')
net= CRNN(n_classes, (3, height, width)).to(device)
net.load_state_dict(torch.load('last.pt'))

def decode(sequence):
    a = ''.join([charset[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != charset[0] and x != a[j+1]])
    if len(s) == 0:
        return ''
    if a[-1] != charset[0] and s[-1] != a[-1]:
        s += a[-1]
    return s

def login(id, password, patient=20):
    url = 'https://cas.bjtu.edu.cn/auth/login/'
    flag = 1
    while flag:
        try:
            res = requests.get(url)
            flag = 0
        except:
            print('reconnect')
            patient -= 1
            if patient == 0:
                print('totally failed')
                return -1
    
    cookies = res.cookies
    res.encoding = 'utf-8'
    soup = bs4.BeautifulSoup(res.text, 'html.parser')
    lt = soup.find('input', id='id_captcha_0')['value']
    csrfmiddlewaretoken = soup.find('input', attrs={'name': 'csrfmiddlewaretoken'})['value']
    url2 = 'https://cas.bjtu.edu.cn/captcha/image/' + lt + '/'
    flag = 1
    while flag:
        try:
            res2 = requests.get(url2, cookies=cookies)
            flag = 0
        except:
            print('reconnect')
            patient -= 1
            if patient == 0:
                print('totally failed')
                return -1
    with open('captcha.jpg', 'wb') as f:
        f.write(res2.content)
    img = Image.open('captcha.jpg')
    img = to_tensor(img).to(device)
    img = img.unsqueeze(0)
    preds = net(img)
    preds_argmax = preds.detach().permute(1, 0, 2).argmax(dim=-1).cpu().numpy()
    name = decode(preds_argmax[0])
    order = name
    print(name)
    name = name[:-1].replace('×', '*')
    try:
        name = str(eval(name))
        print(name)
    except:
        print('retry')
        #记录下，然后重来
        with open('detect_failed/' + order +'.jpg', 'wb') as f:
            f.write(res2.content)
        return login(id, password, patient)
    parmas = {'csrfmiddlewaretoken': csrfmiddlewaretoken, 'captcha_0': lt, 'captcha_1': name, 'loginname': id, 'password': password, 'next': ''}
    headers = {
        'Host': 'cas.bjtu.edu.cn', 
        'Content-Length': '194',
        'Cache-Control': 'max-age=0',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Upgrade-Insecure-Requests': '1',
        'Origin': 'https://cas.bjtu.edu.cn',
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.199 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Referer': 'https://cas.bjtu.edu.cn/auth/login/',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Priority': 'u=0, i'
    }
    flag = 1
    while flag:
        try:
            res3 = requests.post(url, data=parmas, headers=headers, cookies=cookies)
            flag = 0
        except:
            print('reconnect')
            patient -= 1
            if patient == 0:
                print('totally failed')
                return -1
    #print(res3.headers)
    #print(res3.cookies)
    if res3.cookies.get('csrftoken') != None:
        soup = bs4.BeautifulSoup(res3.text, 'html.parser')
        lt = soup.find('div', class_ = 'profile-info-value').text
        print('欢迎回来', lt)
        return res3.cookies.get('csrftoken')
    else:

        with open('detect_failed/' + order +'.jpg', 'wb') as f:
            f.write(res2.content)
        return login(id, password, patient)

id = ''         #输入学号
passwd = ''     #输入密码mis的，你自己设置的！！！
print(login(id, passwd))