import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class BearingFaultDetector(nn.Module):
    def __init__(self, input_size=3, num_classes=4):
        super(BearingFaultDetector, self).__init__()

        # 1D卷积层，用于特征提取
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM层，捕捉时序特征
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=50, num_layers=2, batch_first=True
        )

        # 全连接层，用于分类
        self.fc1 = nn.Linear(50, 20)
        self.fc2 = nn.Linear(20, num_classes)

        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x的输入形状应为[batch_size, input_size, sequence_length]

        # 卷积层处理
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # 调整形状以适应LSTM输入 [batch_size, sequence_length, features]
        x = x.permute(0, 2, 1)

        # LSTM处理
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 只取最后一个时间步的输出

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 数据加载和处理函数
def load_cwru_data(path):
    """加载西储大学轴承数据集"""
    # 实际实现时需要根据数据集格式进行调整
    # 这里只是示例
    pass


# 特征提取函数
def extract_features(data, window_size=1024, overlap=512):
    """从原始振动信号中提取特征"""
    features = []
    labels = []

    # 实际实现时需要根据数据格式进行窗口滑动和特征提取
    # 这里只是示例框架

    return np.array(features), np.array(labels)


# 模型训练函数
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """训练轴承故障检测模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    return model


if __name__ == "__main__":
    # 加载数据
    X, y = load_cwru_data("path/to/dataset")

    # 数据预处理
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).view(
        -1, 3, 1024
    )  # 假设每个样本有3个通道，1024个时间点
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).view(-1, 3, 1024)
    y_test_tensor = torch.LongTensor(y_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    model = BearingFaultDetector()

    # 训练模型
    trained_model = train_model(model, train_loader, test_loader)

    # 保存模型
    torch.save(trained_model.state_dict(), "bearing_fault_model.pth")
