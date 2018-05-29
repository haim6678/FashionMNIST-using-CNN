import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

xViector = range(1, 11)
lossTestViector = {'regular': [], 'norm': [], 'drop': []}
lossValidViector = {'regular': [], 'norm': [], 'drop': []}
lossTrainViector = {'regular': [], 'norm': [], 'drop': []}
batchSize = 4

transforms = transforms.Compose([
    transforms.ToTensor()])

train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms)
valid_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms)
test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(0.8 * num_train)
train_idx, valid_idx = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchSize, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True)


class FirstNet(nn.Module):
    def __init__(self, image_size, batch_norm=False, drop=False):
        super(FirstNet, self).__init__()
        self.image_size = image_size

        self.batch_norm = batch_norm
        self.drop = drop
        self.bn0 = nn.BatchNorm1d(100)
        self.bn1 = nn.BatchNorm1d(50)
        self.do = nn.Dropout(p=0.25)

        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)

        if self.batch_norm:
            x = F.relu(self.bn0(self.fc0(x)))
            x = F.relu(self.bn1(self.fc1(x)))
        elif self.drop:
            x = F.relu(self.do(self.fc0(x)))
            x = F.relu(self.do(self.fc1(x)))
        else:
            x = F.relu(self.fc0(x))
            x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


def train(model, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(model, name):
    global lossTestViector
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    lossTestViector[name].append(test_loss)


def test_valid(model, name):
    global lossValidViector
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        output = model(data)
        valid_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    valid_loss /= len(valid_loader) * batchSize
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        valid_loss, correct, len(valid_loader) * batchSize,
                             100. * correct / (len(valid_loader) * batchSize)))
    lossValidViector[name].append(valid_loss)


def test_train(model, name):
    global lossTrainViector
    model.eval()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        output = model(data)
        train_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    train_loss /= len(train_loader) * batchSize
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader) * batchSize,
                             100. * correct / (len(train_loader) * batchSize)))
    lossTrainViector[name].append(train_loss)


def run_model(model, optimize, name):
    for epoch in range(1, 10 + 1):
        train(model, optimize)
        print name + ' epoch:', epoch
        test(model, name)
        test_valid(model, name)
        test_train(model, name)
        print ' '
        if epoch == 5:
            if name != 'regular':
                optimize = optim.SGD(model.parameters(), lr=0.008)
        if epoch == 7:
            if name == 'regular':
                optimize = optim.SGD(model.parameters(), lr=0.003)
            else:
                optimize = optim.SGD(model.parameters(), lr=0.005)


def init_model(learning, batch, norm, drop, name):
    global train_loader, valid_loader, test_loader, batchSize
    batchSize = batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=True)
    model = FirstNet(image_size=28 * 28, batch_norm=norm, drop=drop)
    optimizer = optim.SGD(model.parameters(), lr=learning)
    return model, optimizer


def write_result(model):
    model.eval()
    with open('test.pred', 'w') as the_file:
        for data, target in test_loader:
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            pre_index = np.argmax(output.cpu().data.numpy())
            the_file.writelines(str(pre_index) + '\n')


def iterate_all_models():
    global train_loader, valid_loader, test_loader

    print '--------------------------------- model: A ------------------------------' + '\n'
    model, optimizer = init_model(0.009, 1, False, False, 'regular')
    run_model(model, optimizer, 'regular')
    write_result(model)
    print '--------------------------------- model: batch ------------------------------' + '\n'
    model, optimizer = init_model(0.015, 6, True, False, 'norm')
    run_model(model, optimizer, 'norm')

    print '---------------------------------- model: drop ------------------------------' + '\n'
    model, optimizer = init_model(0.015, 6, False, True, 'drop')
    run_model(model, optimizer, 'drop')


# def plot(name):
#     global xViector, lossTrainViector, lossValidViector
#     plt.plot(xViector, lossValidViector[name], "r", label='Model')
#     plt.plot(xViector, lossTrainViector[name], "b--", label='Real')
#     plt.legend(('epoch', 'loss'))
#     plt.show()


def main():
    iterate_all_models()
    # plot('regular')
    # plot('norm')
    # plot('drop')


if __name__ == "__main__":
    main()
