import glob
import os
import numpy as np
import torchvision.transforms as transforms
from tool.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 帧数抽取工具，从序列中按一定间隔抽帧
def extract_frame(data, num):
    N, C, T, V, M = data.shape
    sample = np.zeros((N, C, num, V, M))
    for i in range(num):
        sample[:, :, i, :, :] = data[:, :, int(T*i/num), :, :]

    return sample


transform = transforms.Compose([
    transforms.Normalize([0.5], [0.5], [0.5])
])


# #数据读取以及数据处理
files = sorted(glob.glob(os.path.join("data", "test") + '/*/*.*'))
labels = [file.split('\\')[2] for file in files]
dict = {'000': 0, '001': 1, '002': 2, '003': 3, '004': 4}
frames = 48  # 抽取的帧数
# 预分配数据空间
datas = torch.zeros((len(labels),3,frames,34)).to(device)
target = torch.zeros((1, len(labels))).int().to(device)
# 读取并处理数据，获得测试数据集datas，标记集target
for index in range(len(labels)):
    sample = np.load(files[index])
    sample[:, 1, :, :, :] = 0
    sample = extract_frame(sample, frames)
    sample = np.resize(sample, (1, 3, frames, 34))
    sample = torch.Tensor(np.squeeze(sample))
    # get the transformed data and label
    if transform is not None:
        sample = transform(sample)
        datas[index, :, :, :] = sample
    label = dict[labels[index]]
    target[0, index] = label


# 调用模型，并利用测试集数据对模型进行检验
model = SequencesNet().to(device)
model.load_state_dict(torch.load("model/net_trained.pth"))
correct = 0
total = len(labels)

with torch.no_grad():
    outputs = model(datas)
    _, predicted = torch.max(outputs.data, 1)
    print("predict:\n", predicted)
    print("target:\n", target)
    print("is equal:\n", predicted == target)
    correct += (predicted == target).sum().item()
print("correct:", correct, "total:", total)
print('Accuracy of the network on the test_dataset: %d %%' % (
        100 * correct / total))














