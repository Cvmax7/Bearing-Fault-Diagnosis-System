import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime

# CNN+LSTM混合模型
class BearingFaultDetector(nn.Module):
    def __init__(self, input_size=1, seq_length=1024, num_classes=4):
        super(BearingFaultDetector, self).__init__()
        
        # 1D卷积层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算LSTM输入大小
        conv_output_size = seq_length // 8  # 经过3次池化，长度变为原来的1/8
        
        # LSTM层
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=2, batch_first=True)
        
        # 全连接层
        self.fc1 = nn.Linear(50, 20)
        self.fc2 = nn.Linear(20, num_classes)
        
        # Dropout和激活函数
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x形状: [batch_size, input_size, seq_length]
        
        # 卷积层处理
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # 调整形状以适应LSTM输入 [batch_size, seq_length, features]
        x = x.permute(0, 2, 1)
        
        # LSTM处理
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 纯CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_size=1, seq_length=1024, num_classes=4):
        super(CNNModel, self).__init__()
        
        # 1D卷积层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc_input_size = 128 * (seq_length // 16)  # 经过4次池化
        self.fc1 = nn.Linear(self.fc_input_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_classes)
        
        # Dropout和激活函数
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 卷积层处理
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# 基于RNN的模型
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=4):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN层
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向RNN，所以是hidden_size*2
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # 调整输入形状 [batch_size, seq_length, input_size]
        x = x.permute(0, 2, 1)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # RNN前向传播
        out, _ = self.rnn(x, h0)
        
        # 取最后一个时间步的输出
        out = self.dropout(out[:, -1, :])
        
        # 全连接层
        out = self.fc(out)
        
        return out

# 1D ResNet模型
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, input_size=1, num_classes=4):
        super(ResNet1D, self).__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet18_1d(input_size=1, num_classes=4):
    return ResNet1D(ResidualBlock, [2, 2, 2, 2], input_size=input_size, num_classes=num_classes)

# 模型训练器类
class ModelTrainer:
    def __init__(self, model_type='cnn_lstm', input_size=1, seq_length=1024, num_classes=4, learning_rate=0.001, batch_size=32, num_epochs=50):
        self.model_type = model_type
        self.input_size = input_size
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # 选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = self._initialize_model()
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 创建保存模型的目录
        os.makedirs("models", exist_ok=True)
        
    def _initialize_model(self):
        """初始化模型"""
        if self.model_type == 'cnn_lstm':
            return BearingFaultDetector(input_size=self.input_size, seq_length=self.seq_length, num_classes=self.num_classes)
        elif self.model_type == 'cnn':
            return CNNModel(input_size=self.input_size, seq_length=self.seq_length, num_classes=self.num_classes)
        elif self.model_type == 'rnn':
            return RNNModel(input_size=self.input_size, num_classes=self.num_classes)
        elif self.model_type == 'resnet':
            return resnet18_1d(input_size=self.input_size, num_classes=self.num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def prepare_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """
        准备训练、验证和测试数据
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            X_test: 测试数据
            y_test: 测试标签
            
        Returns:
            train_loader, val_loader, test_loader
        """
        # 确保数据是三维的 [样本数, 通道数, 序列长度]
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        
        if X_val is not None and len(X_val.shape) == 2:
            X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            
        if X_test is not None and len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        # 如果提供了验证集
        val_dataset = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 如果提供了测试集
        test_loader = None
        if X_test is not None and y_test is not None:
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            history: 训练历史记录
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        best_model_path = None
        
        print(f"开始训练 {self.model_type} 模型...")
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # 训练模式
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                # 累计损失
                train_loss += loss.item() * inputs.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                progress_bar.set_postfix(loss=loss.item(), acc=train_correct/train_total)
                
            # 计算训练集平均损失和准确率
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            
            # 验证模式
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            if val_loader:
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        
                        # 累计损失
                        val_loss += loss.item() * inputs.size(0)
                        
                        # 计算准确率
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # 计算验证集平均损失和准确率
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = val_correct / val_total
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = f"models/{self.model_type}_best_model.pth"
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Epoch {epoch+1}: 保存最佳模型，验证损失: {val_loss:.4f}")
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            if val_loader:
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            # 打印进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.num_epochs} - {epoch_time:.1f}s - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}", end="")
            if val_loader:
                print(f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                print("")
        
        # 保存最终模型
        final_model_path = f"models/{self.model_type}_final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"训练结束，保存最终模型: {final_model_path}")
        
        if best_model_path:
            print(f"最佳模型: {best_model_path}")
            # 加载最佳模型
            self.model.load_state_dict(torch.load(best_model_path))
        
        return history
    
    def evaluate(self, test_loader):
        """
        评估模型性能
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            metrics: 评估指标
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # 收集预测结果和标签
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # 打印评估指标
        print("\n模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1得分: {f1:.4f}")
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("混淆矩阵")
        plt.savefig(f"models/{self.model_type}_confusion_matrix.png")
        plt.show()
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix
        }
    
    def plot_training_history(self, history):
        """
        绘制训练历史
        
        Args:
            history: 训练历史记录
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        if 'val_acc' in history and history['val_acc']:
            plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"models/{self.model_type}_training_history.png")
        plt.show()
    
    def predict(self, data):
        """
        使用模型进行预测
        
        Args:
            data: 输入数据，形状为 [样本数, 序列长度] 或 [样本数, 通道数, 序列长度]
            
        Returns:
            predictions: 预测结果
            probabilities: 预测概率
        """
        # 确保数据是三维的 [样本数, 通道数, 序列长度]
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], 1, data.shape[1])
        
        # 转换为PyTorch张量
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # 评估模式
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def save_model(self, path=None):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if path is None:
            path = f"models/{self.model_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'input_size': self.input_size,
            'seq_length': self.seq_length,
            'num_classes': self.num_classes
        }, path)
        
        print(f"模型已保存到: {path}")
        
        return path
    
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 更新模型参数
        if 'model_type' in checkpoint:
            self.model_type = checkpoint['model_type']
        if 'input_size' in checkpoint:
            self.input_size = checkpoint['input_size']
        if 'seq_length' in checkpoint:
            self.seq_length = checkpoint['seq_length']
        if 'num_classes' in checkpoint:
            self.num_classes = checkpoint['num_classes']
        
        # 重新初始化模型
        self.model = self._initialize_model()
        self.model.to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果有优化器状态，也加载
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"模型已从 {path} 加载")
    
    def export_onnx(self, path=None, sample_input=None):
        """
        将模型导出为ONNX格式
        
        Args:
            path: 导出路径
            sample_input: 样本输入，用于跟踪模型
        """
        if path is None:
            path = f"models/{self.model_type}_model.onnx"
            
        if sample_input is None:
            sample_input = torch.randn(1, self.input_size, self.seq_length).to(self.device)
            
        self.model.eval()
        
        torch.onnx.export(
            self.model,               # 模型
            sample_input,             # 样本输入
            path,                     # 输出路径
            export_params=True,       # 存储模型训练参数
            opset_version=11,         # ONNX版本
            do_constant_folding=True, # 是否执行常量折叠优化
            input_names=['input'],    # 输入名称
            output_names=['output'],  # 输出名称
            dynamic_axes={            # 动态尺寸
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"模型已导出为ONNX格式: {path}")
        
        return path 