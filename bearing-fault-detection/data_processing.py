import os
import numpy as np
import pandas as pd
import scipy.io
import requests
import zipfile
import io
from scipy import signal
from sklearn.model_selection import train_test_split


class DatasetProcessor:
    def __init__(self, data_dir="./datasets"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # 存储数据集路径
        self.cwru_dir = os.path.join(data_dir, "CWRU")
        self.mfpt_dir = os.path.join(data_dir, "MFPT")

        # 创建数据集目录
        os.makedirs(self.cwru_dir, exist_ok=True)
        os.makedirs(self.mfpt_dir, exist_ok=True)

        # 数据集下载链接
        self.cwru_download_urls = {
            "normal_0": "https://engineering.case.edu/sites/default/files/97.mat",  # 正常数据
            "inner_007": "https://engineering.case.edu/sites/default/files/105.mat",  # 内圈故障 0.007英寸
            "outer_007": "https://engineering.case.edu/sites/default/files/130.mat",  # 外圈故障 0.007英寸
            "ball_007": "https://engineering.case.edu/sites/default/files/118.mat",  # 滚动体故障 0.007英寸
        }

        self.mfpt_download_urls = {
            "base": "https://www.mfpt.org/fault-data-sets/",
            # MFPT数据集需要从官方网站手动下载
        }

    def download_cwru_dataset(self, force_download=False):
        """下载CWRU轴承数据集"""
        print("开始下载CWRU轴承数据集...")

        for name, url in self.cwru_download_urls.items():
            target_file = os.path.join(self.cwru_dir, f"{name}.mat")

            if os.path.exists(target_file) and not force_download:
                print(f"文件 {name}.mat 已存在，跳过下载")
                continue

            try:
                print(f"正在下载 {name}.mat...")
                response = requests.get(url)
                with open(target_file, "wb") as f:
                    f.write(response.content)
                print(f"下载完成: {name}.mat")
            except Exception as e:
                print(f"下载 {name}.mat 失败: {e}")

        print("CWRU数据集下载完成")

    def load_cwru_data(self):
        """加载CWRU数据集"""
        X = []
        y = []
        labels = {
            "normal_0": 0,  # 正常
            "inner_007": 1,  # 内圈故障
            "outer_007": 2,  # 外圈故障
            "ball_007": 3,  # 滚动体故障
        }

        sample_size = 1024  # 每个样本的数据点数

        for name, label in labels.items():
            file_path = os.path.join(self.cwru_dir, f"{name}.mat")

            if not os.path.exists(file_path):
                print(f"文件 {file_path} 不存在，请先下载数据集")
                continue

            try:
                mat_data = scipy.io.loadmat(file_path)

                # 提取振动数据，CWRU数据集有不同的键名
                data_keys = [
                    k for k in mat_data.keys() if "DE" in k and not k.startswith("__")
                ]
                if not data_keys:
                    print(f"在 {name}.mat 中未找到振动数据")
                    continue

                vibration_data = mat_data[data_keys[0]].flatten()

                # 将长信号分割成固定长度的样本
                num_samples = len(vibration_data) // sample_size

                for i in range(num_samples):
                    start_idx = i * sample_size
                    end_idx = start_idx + sample_size

                    sample = vibration_data[start_idx:end_idx]

                    if len(sample) == sample_size:
                        X.append(sample)
                        y.append(label)

                print(f"从 {name}.mat 提取了 {num_samples} 个样本")

            except Exception as e:
                print(f"处理 {name}.mat 时出错: {e}")

        return np.array(X), np.array(y)

    def extract_features(self, X_raw):
        """从原始振动信号中提取特征"""
        num_samples = X_raw.shape[0]
        features = np.zeros((num_samples, 8))  # 8个时域特征

        for i in range(num_samples):
            x = X_raw[i]

            # 时域特征
            # 1. 均方根值(RMS)
            rms = np.sqrt(np.mean(np.square(x)))

            # 2. 峰值
            peak = np.max(np.abs(x))

            # 3. 峰值因子(Crest Factor)
            crest_factor = peak / rms if rms > 0 else 0

            # 4. 峭度(Kurtosis)
            kurtosis = (
                np.mean(np.power(x - np.mean(x), 4)) / np.power(np.std(x), 4)
                if np.std(x) > 0
                else 0
            )

            # 5. 偏度(Skewness)
            skewness = (
                np.mean(np.power(x - np.mean(x), 3)) / np.power(np.std(x), 3)
                if np.std(x) > 0
                else 0
            )

            # 6. 方差(Variance)
            variance = np.var(x)

            # 7. 峰峰值(Peak-to-Peak)
            p2p = np.max(x) - np.min(x)

            # 8. 脉冲因子(Impulse Factor)
            impulse_factor = peak / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else 0

            features[i] = [
                rms,
                peak,
                crest_factor,
                kurtosis,
                skewness,
                variance,
                p2p,
                impulse_factor,
            ]

        return features

    def prepare_data_for_training(
        self, window_size=1024, overlap=512, extract_features=False
    ):
        """准备用于训练的数据"""
        # 加载原始数据
        X_raw, y = self.load_cwru_data()

        if extract_features:
            # 提取特征
            X = self.extract_features(X_raw)
        else:
            # 使用原始波形
            X = X_raw

        # 划分训练集、验证集和测试集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def simulate_phm_data(self, num_samples=100, fault_ratio=0.3):
        """
        模拟PHM平台数据，用于算法验证

        Args:
            num_samples: 要生成的样本数
            fault_ratio: 故障样本的比例

        Returns:
            模拟的PHM平台数据
        """
        # 轴承特征频率参数
        bpfi_coef = 5.4  # 内圈故障特征频率系数
        bpfo_coef = 3.6  # 外圈故障特征频率系数
        bsf_coef = 2.8  # 滚动体故障特征频率系数

        sample_size = 1024
        sample_rate = 12000  # 采样率(Hz)

        X = []
        y = []

        for i in range(num_samples):
            rpm = np.random.uniform(1000, 3000)  # 随机转速
            base_freq = rpm / 60  # 基础频率(Hz)

            t = np.linspace(0, sample_size / sample_rate, sample_size)

            # 基础振动信号(包含噪声)
            base_signal = np.sin(2 * np.pi * base_freq * t) + 0.1 * np.random.randn(
                sample_size
            )

            # 随机决定样本类型
            rand_val = np.random.random()

            if rand_val < fault_ratio:
                # 故障样本
                fault_type = np.random.randint(1, 4)  # 1=内圈, 2=外圈, 3=滚动体

                if fault_type == 1:
                    # 内圈故障
                    bpfi = base_freq * bpfi_coef
                    fault_signal = (
                        0.5
                        * np.sin(2 * np.pi * bpfi * t)
                        * (1 + 0.2 * np.sin(2 * np.pi * base_freq * t))
                    )
                    signal = base_signal + fault_signal
                    y.append(1)
                elif fault_type == 2:
                    # 外圈故障
                    bpfo = base_freq * bpfo_coef
                    fault_signal = 0.4 * np.sin(2 * np.pi * bpfo * t)
                    signal = base_signal + fault_signal
                    y.append(2)
                else:
                    # 滚动体故障
                    bsf = base_freq * bsf_coef
                    fault_signal = (
                        0.3
                        * np.sin(2 * np.pi * bsf * t)
                        * (1 + 0.1 * np.sin(4 * np.pi * base_freq * t))
                    )
                    signal = base_signal + fault_signal
                    y.append(3)
            else:
                # 正常样本
                signal = base_signal
                y.append(0)

            X.append(signal)

        return np.array(X), np.array(y)
