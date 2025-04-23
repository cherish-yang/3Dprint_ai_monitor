import pandas as pd
import logging
from sklearn.utils import shuffle
import numpy as np
from PIL import Image
from torch import nn
import torch as t
from torch.nn import functional as F
import torch as torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PIL import ImageFile
from tqdm import tqdm
from torchvision.transforms import ToPILImage
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ReadPath:
    def __init__(self, path, root):
        self.path = path
        self.root = root

    def read_csv(self):
        data = pd.read_csv(self.path)
        return data

    def read_img(self):
        datas = self.read_csv()
        img_paths = datas['img_path']
        # 取出 datas 的 'flow_rate_class', 'feed_rate_class', 'z_offset_class', 'hotend_class' 列。
        labels = datas[['flow_rate_class', 'feed_rate_class', 'z_offset_class', 'hotend_class']].values
        img_mean = datas[['img_mean']].values
        img_std = datas[['img_std']].values
        nozzle_tip_x=datas[['nozzle_tip_x']].values
        nozzle_tip_y=datas[['nozzle_tip_y']].values
        # 构造完整图片路径，但不加载图片
        path = [self.root + '/' + str(img_path) for img_path in img_paths]
        return path, img_mean, img_std, labels,nozzle_tip_x,nozzle_tip_y


# 创建一个简单的自定义 Dataset，将路径和标签封装
class CustomDataset():
    def __init__(self, img_paths, img_mean, img_std, labels,nozzle_tip_x,nozzle_tip_y):
        self.img_paths = img_paths
        self.labels = labels
        self.img_mean = img_mean
        self.img_std = img_std
        self.nozzle_tip_x=nozzle_tip_x
        self.nozzle_tip_y=nozzle_tip_y

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img_mean = self.img_mean[idx]
        img_std = self.img_std[idx]
        nozzle_tip_x=self.nozzle_tip_x[idx]
        nozzle_tip_y=self.nozzle_tip_y[idx]
        # 暂时只返回路径和标签，不加载图片
        return img_path, img_mean, img_std, label,nozzle_tip_x,nozzle_tip_y


def path_to_tensor(train_data_path, img_means, img_stds,nozzle_tip_xs,nozzle_tip_ys):
    # train_data_path路径指向的图片
    img_array = None
    for i, img_mean, img_std,nozzle_tip_x,nozzle_tip_y in zip(train_data_path, img_means, img_stds,nozzle_tip_xs,nozzle_tip_ys):
        img = Image.open(i)
        # 将图片转换为RGB模式，确保它有三个颜色通道
        img = img.convert('RGB')
        pixels_array = np.array(img)
        pixels_array = pixels_array[nozzle_tip_x-160:nozzle_tip_x+160, nozzle_tip_y-160:nozzle_tip_y+160]
        img = Image.fromarray(pixels_array)
        pixels_array=img.resize((224, 224))
        pixels_array = np.array(pixels_array)
        
        #将图片的长和宽裁成224x224
        # 归一化图像
        img_mean = np.array(img_mean)
        img_std = np.array(img_std)
        img_mean = np.expand_dims(img_mean, axis=0)
        img_std = np.expand_dims(img_std, axis=0)
        normalized_image_array = (pixels_array - img_mean) / img_std

        # 确保归一化后的值在0到1之间
        pixels_array = np.clip(normalized_image_array, 0, 1)
        pixels_array = np.expand_dims(pixels_array, axis=0)
        if img_array is None:
            img_array = pixels_array
        else:
            img_array = np.concatenate((img_array, pixels_array), axis=0)
    return img_array


class ResidualBlock(nn.Module):
    # 实现子module: Residual Block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    # 实现主module:ResNet34
    # ResNet34包含多个layer，每个layer又包含多个residual block
    # 用子module实现residual block，用_make_layer函数实现layer

    def __init__(self, num_classes=3, head_num=4):
        super(ResNet, self).__init__()
        self.head_num = head_num
        self.num_classes = num_classes
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 重复的layer，分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        # 分类用的全连接
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, self.num_classes)

        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 16)
        self.fc6 = nn.Linear(16, self.num_classes)

        self.fc7 = nn.Linear(512, 128)
        self.fc8 = nn.Linear(128, 16)
        self.fc9 = nn.Linear(16, self.num_classes)

        self.fc10 = nn.Linear(512, 128)
        self.fc11 = nn.Linear(128, 16)
        self.fc12 = nn.Linear(16, self.num_classes)

    def fc(self, x, case):
        if case == "flow":
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        elif case == "feed":
            x = self.fc4(x)
            x = self.fc5(x)
            x = self.fc6(x)
        elif case == "z_offset":
            x = self.fc7(x)
            x = self.fc8(x)
            x = self.fc9(x)
        elif case == "hotend":
            x = self.fc10(x)
            x = self.fc11(x)
            x = self.fc12(x)
        x = torch.sigmoid(x)
        x = F.softmax(x)
        return x

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构造layer，包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 应用平均池化层
        x = F.avg_pool2d(x, 7)
        # 将结果重塑为(batch_size, head_num, -1)
        x = x.reshape(x.size(0), -1)
        x_flow = self.fc(x, "flow")
        x_feed = self.fc(x, "feed")
        z_offset = self.fc(x, "z_offset")
        hotend = self.fc(x, "hotend")
        x=torch.stack((x_flow, x_feed, z_offset, hotend), dim=0)
        return x

# 训练循环 前向传播,后向传播，更新
def train(inputs, target):
    running_loss = 0
    optimizer.zero_grad()
    x = model.forward(inputs)
    x=x.transpose(1, 0)
    loss=0
    for i in range(len(x)):
        loss+=criterion(x[i], target[i]) 
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    return running_loss


def test(inputs, label, correct, total, x_flow_correct, x_feed_correct, z_offset_correct, hotend_correct):
    with torch.no_grad():
        x_flow, x_feed, z_offset, hotend = model.forward(inputs)
        _, x_flow_predicted = torch.max(x_flow.data, dim=1)
        _, x_feed_predicted = torch.max(x_feed.data, dim=1)
        _, z_offset_predicted = torch.max(z_offset.data, dim=1)
        _, hotend_predicted = torch.max(hotend.data, dim=1)
        sample = len(label)
        total += sample
        for i in range(sample):
            if (x_flow_predicted[i] == label.T[0][i] and x_feed_predicted[i] == label.T[1][i] and z_offset_predicted[
                i] == label.T[2][i] and hotend_predicted[i] == label.T[3][i]):
                correct += 1
            if x_flow_predicted[i] == label.T[0][i]:
                x_flow_correct += 1
            if x_feed_predicted[i] == label.T[1][i]:
                x_feed_correct += 1
            if z_offset_predicted[i] == label.T[2][i]:
                z_offset_correct += 1
            if hotend_predicted[i] == label.T[3][i]:
                hotend_correct += 1
    return correct, x_flow_correct, x_feed_correct, z_offset_correct, hotend_correct, total


# 找当前根路径
root = r"/home/x1/pythonProject"
read_data = ReadPath(f'{root}/test.csv', root)
img_paths, img_mean, img_std, labels,nozzle_tip_x,nozzle_tip_y = read_data.read_img()

# 将路径列表和标签转化为 PyTorch 张量
# 注意这里不转换 img_paths，保留为字符串列表
labels = torch.from_numpy(labels).long()
# 创建数据集
dataset = CustomDataset(img_paths, img_mean, img_std, labels,nozzle_tip_x,nozzle_tip_y)

# 划分训练集和测试集
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
logging.basicConfig(filename='resnet_100轮.log', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# 打印训练集和测试集的大小
logging.info(f"训练集大小: {train_size}, 测试集大小: {test_size}")

model = ResNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)
USE_GPU = 1
if USE_GPU:
    device = torch.device('cuda')
    model.to(device)

# 设置 batch_size 和创建 DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
correct_rate_max=0
for epoch in range(300):
    epoch_loss = 0
    for train_data_path, img_mean, img_std, label,nozzle_tip_x,nozzle_tip_y in tqdm(train_loader, desc="Epoch progress"):
        Train_Data = path_to_tensor(train_data_path, img_mean, img_std,nozzle_tip_x,nozzle_tip_y)
        Train_Data = torch.from_numpy(Train_Data).float().permute(0, 3, 1, 2)
        Train_Data = Train_Data.to('cuda')
        label = label.to('cuda')
        run_loss = train(Train_Data, label)
        epoch_loss += run_loss
        loss_avg=epoch_loss / train_size
    logging.info(f"***************第{epoch + 1}轮训练完成，其平均损失是{loss_avg}***************")
    # 保存整个模型到.pt文件
    correct = 0
    total = 0
    x_flow_correct, x_feed_correct, z_offset_correct, hotend_correct = (0, 0, 0, 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for test_data_path, img_mean, img_std, label,nozzle_tip_x,nozzle_tip_y in test_loader:
        Test_Data = path_to_tensor(test_data_path, img_mean, img_std,nozzle_tip_x,nozzle_tip_y)
        Test_Data = torch.from_numpy(Test_Data).float().permute(0, 3, 1, 2)
        Test_Data = Test_Data.to('cuda')
        label = label.to('cuda')
        correct, x_flow_correct, x_feed_correct, z_offset_correct, hotend_correct, total = test(Test_Data, label,
                                                                                                correct, total,
                                                                                                x_flow_correct,
                                                                                                x_feed_correct,
                                                                                                z_offset_correct,
                                                                                                hotend_correct)
    correct_rate=correct / total
    if correct_rate>correct_rate_max:
        correct_rate_max=correct_rate
        torch.save(model, f'resnet34_{epoch}_{correct / total * 100}%.pt')
        with torch.no_grad():
            torch.onnx.export(
                model,
                Test_Data,
                f"resnet34_{epoch}_{correct / total * 100}%.onnx",
                export_params=True,
                opset_version=10,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
        logging.info(f"综合正确率是:{correct / total * 100}%")
        logging.info(f"流量控制正确率是：{x_flow_correct / total * 100}%")
        logging.info(f"喂料控制正确率是：{x_feed_correct / total * 100}%")
        logging.info(f"偏移控制正确率是：{z_offset_correct / total * 100}%")
        logging.info(f"热沸控制正确率是：{hotend_correct / total * 100}%")

