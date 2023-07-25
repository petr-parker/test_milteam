from collections import OrderedDict

import torch
import torch.nn as nn


class Sample(nn.Module):
    '''
    Шаблон класса подмодели суперсети.
    Подмодель имеет вид: in -> layer1 -> [block1] -> layer2 -> [block2] -> global_avg_pool -> linear -> out.
    [block] может состоять из 1, 2 или 3 сверток.
    '''
    def __init__(self): 
        super(Sample, self).__init__()
        self.layer1 = None
        self.block1 = None
        self.layer2 = None
        self.block2 = None
        self.global_avg_pool = None
        self.fc = None

    def forward(self, x):
        out = self.layer1(x)
        for layer in self.block1:
            out = layer(out)

        out = self.layer2(out)
        for layer in self.block2:
            out = layer(out)

        out = self.global_avg_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

channels = 32

class SuperNet():
    '''
    Класс суперсети, в котором хранятся слои всех вариантов архитектуры. 
    '''
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.layer1 = nn.Sequential(nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.block1 = [
            nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        ]
        self.layer2 = nn.Sequential(nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(2 * channels), nn.ReLU())
        self.block2 = [
            nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(2 * channels), nn.ReLU()),
            nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(2 * channels), nn.ReLU()),
            nn.Sequential(nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(2 * channels), nn.ReLU())
        ]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2 * channels, 10)

    def weights_init(self):
        '''
        Метод для случайной инициализации всех слоев.
        '''
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -1, 1)
                if m.bias is not None:
                    torch.nn.init.uniform_(m.bias, -1, 1)
        
        self.layer1.apply(init_weights)
        for layer in self.block1:
            layer.apply(init_weights)
        
        self.layer2.apply(init_weights)
        for layer in self.block2:
            layer.apply(init_weights)
        
        self.global_avg_pool.apply(init_weights)
        torch.nn.init.uniform_(self.fc.weight, -1, 1)
        torch.nn.init.uniform_(self.fc.bias, -1, 1)

    def sample(self, submod):
        '''
        Возвращает одну из подмоделей суперсети.

        :param submod = (x, y): номер подмодели. x, y лежат от 1 до 3. x отвечает за первый изменяемый блок, y за второй.
        '''
        x, y = submod
        assert 1 <= x <= 3 and 1 <= y <= 3, "Некорректное задание модели: 1 <= x, y <= 3"

        sam = Sample()
        sam.layer1 = self.layer1
        sam.block1 = self.block1[0 : x]
        sam.layer2 = self.layer2
        sam.block2 = self.block2[0 : y]
        sam.global_avg_pool = self.global_avg_pool
        sam.fc = self.fc

        return sam

    def train_sample(self, submod, train_loader, num_epochs = 5, learning_rate = 0.001):
        '''
        Обучение конкретного экземпляра.

        :param submod = (x, y): номер подмодели. x, y лежат от 1 до 3. x отвечает за первый изменяемый блок, y за второй.
        :param train_loader: загрузчик обучающих данных.
        '''
        criterion = nn.CrossEntropyLoss()
        model = self.sample(submod)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        total_step = len(train_loader)
        loss_list, acc_list = [], []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

        return loss_list, acc_list
    
    def train(self, train_loader, num_epochs = 5, learning_rate = 0.001):
        '''
        Обучение сети целиком. На каждой итерации сэмплируется подмодель, для которой обновляются веса.

        :param train_loader: загрузчик обучающих данных.
        '''
        criterion = self.criterion
        total_step = len(train_loader)
        loss_lists = [[] for _ in range(9)]
        acc_lists = [[] for _ in range(9)]
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                x, y = torch.randint(3, (2,)) + 1
                model = self.sample((x, y))
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_lists[(x - 1) * 3 + (y - 1)].append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_lists[(x - 1) * 3 + (y - 1)].append(1.0 * correct / total)

                if (i + 1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

        return loss_lists, acc_lists
    
    def validate_sample(self, submod, test_loader):
        '''
        Обучение конкретного экземпляра.

        :param submod = (x, y): номер подмодели. x, y лежат от 1 до 3. x отвечает за первый изменяемый блок, y за второй.
        :param train_loader: загрузчик обучающих данных.
        '''
        model = self.sample(submod)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

