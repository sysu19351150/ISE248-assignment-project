from torch.utils.data import DataLoader
from tool.cutout import *
from tool.datasets import *
from tool.model import *


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    acc1 = []
    count = 0  # 计算batch数量
    for ii in range(epoch):
        for i, (input, target) in enumerate(train_loader):

            # 使用cutout进行数据增强，获取更多数据，并将获得的新数据与原数据混合，用以训练
            temp = cutout(input)
            input = torch.cat((input, temp), dim=0)
            target = torch.cat((target, target), dim=0)

            output = model(input)
            loss = criterion(output, target)

            # 每训练50个epoch的数据，计算一次当前的平均准确率
            acc1.append((output.argmax(1) == target).float().mean().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            if count % 50 == 0:
                print("epoch:{}, acc:{}, ".format(count, np.mean(acc1)))
                acc1 = []

        # 每五大轮保存一次模型，最后一轮的模型也要保存
        if (ii + 1) % 5 == 0:
            torch.save(model.state_dict(), "model/net_trained.pth")
        elif ii == epoch-1:
            torch.save(model.state_dict(), "model/net_trained.pth")


# 训练部分
Dataset_npyFiles = NpyDataset()
cutout = Cutout(1, 32, 8)
model = SequencesNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.003)
best_acc = 0.0
BatchSize = 8
train_loader = DataLoader(Dataset_npyFiles, BatchSize, True)
epoch = 30
train(train_loader, model, criterion, optimizer, epoch)

# 训练完成后，对总体训练数据进行测试，得出网络在训练集上的正确率
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(correct,total)
    print('Accuracy of the network on the train_dataset: %d %%' % (
            100 * correct / total))