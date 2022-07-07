import pandas as pd
import glob
import numpy as np
import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

files = glob.glob("data/*")
datas = []

for f in files:
    data = pd.read_excel(f, sheet_name='file')
    data = data.dropna(axis=0, how='any')
    datas.append(data)
print(datas[0])

lst_data = {}
for data in datas:
    name = data['名称'][0]
    tmp_data = []
    # "日期",
    for item in ["开盘价(元)", "最高价(元)", "最低价(元)", "收盘价(元)", "成交额(百万)", "成交量"]:
        tmp_data.append(list(data[item]))
    lst_data[name] = np.transpose(np.asarray(tmp_data), [1, 0])

train_data = []
test_data = []

for d in lst_data:
    d = lst_data[d]
    d = d / np.max(d, axis=0, keepdims=True)
    tmp_data = []
    for i in range(len(d) - 5):
        x = d[i:i+5]
        y = d[i+5:i+6]
        # mean = np.mean(x, axis=0, keepdims=True)
        # var = np.var(x, axis=0, keepdims=True)
        # x = (x - mean)
        # y = (y - mean)
        tmp_data.append((x, y))
    split_id = int(len(tmp_data) * 0.9)
    train_data.extend(tmp_data[:split_id])
    test_data.extend(tmp_data[split_id:])


# 定义模型
LR = 0.01
EPOCH = 8000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(32, 6)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None即隐层状态用0初始化
        r_out = r_out.mean(1, keepdims=True)
        out = self.out(r_out)
        return out


rnn = RNN(6).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

tx = torch.tensor(np.asarray([t[0] for t in train_data], dtype=np.float32)).to(device)
ty = torch.tensor(np.asarray([t[1] for t in train_data], dtype=np.float32)).to(device)

vx = torch.tensor(np.asarray([t[0] for t in test_data], dtype=np.float32)).to(device)
vy = torch.tensor(np.asarray([t[1] for t in test_data], dtype=np.float32)).to(device)

for step in range(EPOCH):
    output = rnn(tx)
    loss = loss_func(output, ty)
    # loss = loss.mean(0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        val_loss = loss_func(rnn(vx), vy)
        print(loss, val_loss)
a = 0


# 表 github
# 相关性
# 12 * 12