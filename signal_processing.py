import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt


class SignalProcessor:
    def __init__(self, sample_rate=12000):
        self.sample_rate = sample_rate

    def compute_fft(self, signal_data, n_points=None):
        """
        计算信号的快速傅里叶变换(FFT)

        Args:
            signal_data: 输入信号
            n_points: FFT点数，默认为信号长度

        Returns:
            freq: 频率数组
            amplitude: 幅值数组
        """
        if n_points is None:
            n_points = len(signal_data)

        # 计算FFT
        yf = fft(signal_data, n_points)
        amplitude = 2.0 / n_points * np.abs(yf[: n_points // 2])
        freq = fftfreq(n_points, 1 / self.sample_rate)[: n_points // 2]

        return freq, amplitude

    def compute_envelope_spectrum(self, signal_data):
        """
        计算信号的包络谱

        Args:
            signal_data: 输入信号

        Returns:
            freq: 频率数组
            amplitude: 包络谱幅值数组
        """
        # 计算信号的希尔伯特变换
        analytic_signal = signal.hilbert(signal_data)

        # 计算包络
        envelope = np.abs(analytic_signal)

        # 计算包络的FFT
        n_points = len(envelope)
        yf = fft(envelope, n_points)
        amplitude = 2.0 / n_points * np.abs(yf[: n_points // 2])
        freq = fftfreq(n_points, 1 / self.sample_rate)[: n_points // 2]

        return freq, amplitude

    def compute_wavelet(self, signal_data, wavelet="db4", level=5):
        """
        计算信号的小波变换

        Args:
            signal_data: 输入信号
            wavelet: 小波类型
            level: 分解级别

        Returns:
            coeffs: 小波系数
        """
        # 小波分解
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)

        return coeffs

    def extract_time_domain_features(self, signal_data):
        """
        提取时域特征

        Args:
            signal_data: 输入信号

        Returns:
            features: 时域特征字典
        """
        features = {}

        # 均值
        features["mean"] = np.mean(signal_data)

        # 均方根值(RMS)
        features["rms"] = np.sqrt(np.mean(np.square(signal_data)))

        # 峰值
        features["peak"] = np.max(np.abs(signal_data))

        # 峰峰值
        features["peak_to_peak"] = np.max(signal_data) - np.min(signal_data)

        # 标准差
        features["std"] = np.std(signal_data)

        # 方差
        features["variance"] = np.var(signal_data)

        # 偏度
        features["skewness"] = (
            np.mean(np.power(signal_data - features["mean"], 3))
            / np.power(features["std"], 3)
            if features["std"] > 0
            else 0
        )

        # 峭度
        features["kurtosis"] = (
            np.mean(np.power(signal_data - features["mean"], 4))
            / np.power(features["std"], 4)
            if features["std"] > 0
            else 0
        )

        # 峰值因子
        features["crest_factor"] = (
            features["peak"] / features["rms"] if features["rms"] > 0 else 0
        )

        # 裕度因子
        features["clearance_factor"] = (
            features["peak"] / np.power(np.mean(np.sqrt(np.abs(signal_data))), 2)
            if np.mean(np.sqrt(np.abs(signal_data))) > 0
            else 0
        )

        # 脉冲因子
        features["impulse_factor"] = (
            features["peak"] / np.mean(np.abs(signal_data))
            if np.mean(np.abs(signal_data)) > 0
            else 0
        )

        # 波形因子
        features["shape_factor"] = (
            features["rms"] / np.mean(np.abs(signal_data))
            if np.mean(np.abs(signal_data)) > 0
            else 0
        )

        return features

    def extract_frequency_domain_features(self, signal_data):
        """
        提取频域特征

        Args:
            signal_data: 输入信号

        Returns:
            features: 频域特征字典
        """
        features = {}

        # 计算FFT
        freq, amplitude = self.compute_fft(signal_data)

        # 频率质心
        features["spectral_centroid"] = (
            np.sum(freq * amplitude) / np.sum(amplitude) if np.sum(amplitude) > 0 else 0
        )

        # 频率方差
        features["spectral_variance"] = (
            np.sum(np.power(freq - features["spectral_centroid"], 2) * amplitude)
            / np.sum(amplitude)
            if np.sum(amplitude) > 0
            else 0
        )

        # 频率偏度
        features["spectral_skewness"] = (
            np.sum(np.power(freq - features["spectral_centroid"], 3) * amplitude)
            / (np.sum(amplitude) * np.power(features["spectral_variance"], 1.5))
            if features["spectral_variance"] > 0 and np.sum(amplitude) > 0
            else 0
        )

        # 频率峭度
        features["spectral_kurtosis"] = (
            np.sum(np.power(freq - features["spectral_centroid"], 4) * amplitude)
            / (np.sum(amplitude) * np.power(features["spectral_variance"], 2))
            if features["spectral_variance"] > 0 and np.sum(amplitude) > 0
            else 0
        )

        # 频谱能量
        features["spectral_energy"] = np.sum(np.square(amplitude))

        # 频谱熵
        norm_amplitude = (
            amplitude / np.sum(amplitude) if np.sum(amplitude) > 0 else amplitude
        )
        entropy = -np.sum(norm_amplitude * np.log2(norm_amplitude + 1e-10))
        features["spectral_entropy"] = entropy

        return features

    def extract_wavelet_features(self, signal_data, wavelet="db4", level=3):
        """
        提取小波特征

        Args:
            signal_data: 输入信号
            wavelet: 小波类型
            level: 分解级别

        Returns:
            features: 小波特征字典
        """
        features = {}

        # 小波分解
        coeffs = self.compute_wavelet(signal_data, wavelet, level)

        # 对每个小波系数计算统计特征
        for i, coef in enumerate(coeffs):
            if i == 0:
                suffix = f"_a{level}"  # 近似系数
            else:
                suffix = f"_d{level - i + 1}"  # 细节系数

            # 能量
            features[f"wavelet_energy{suffix}"] = np.sum(np.square(coef))

            # 均值
            features[f"wavelet_mean{suffix}"] = np.mean(coef)

            # 标准差
            features[f"wavelet_std{suffix}"] = np.std(coef)

            # 最大值
            features[f"wavelet_max{suffix}"] = np.max(np.abs(coef))

        return features

    def extract_all_features(self, signal_data):
        """
        提取所有特征

        Args:
            signal_data: 输入信号

        Returns:
            features: 特征字典
        """
        # 时域特征
        time_features = self.extract_time_domain_features(signal_data)

        # 频域特征
        freq_features = self.extract_frequency_domain_features(signal_data)

        # 小波特征
        wavelet_features = self.extract_wavelet_features(signal_data)

        # 合并所有特征
        features = {**time_features, **freq_features, **wavelet_features}

        return features

    def filter_signal(
        self, signal_data, filter_type="bandpass", lowcut=100, highcut=1000, order=5
    ):
        """
        滤波处理

        Args:
            signal_data: 输入信号
            filter_type: 滤波器类型 ('lowpass', 'highpass', 'bandpass')
            lowcut: 低截止频率
            highcut: 高截止频率
            order: 滤波器阶数

        Returns:
            filtered_signal: 滤波后的信号
        """
        nyq = 0.5 * self.sample_rate

        if filter_type == "lowpass":
            normal_cutoff = highcut / nyq
            b, a = signal.butter(order, normal_cutoff, btype="low")
        elif filter_type == "highpass":
            normal_cutoff = lowcut / nyq
            b, a = signal.butter(order, normal_cutoff, btype="high")
        else:  # bandpass
            low = lowcut / nyq
            high = highcut / nyq
            b, a = signal.butter(order, [low, high], btype="band")

        filtered_signal = signal.filtfilt(b, a, signal_data)

        return filtered_signal

    def plot_signal(self, signal_data, title="Signal", figsize=(12, 4)):
        """
        绘制信号波形

        Args:
            signal_data: 输入信号
            title: 图表标题
            figsize: 图表尺寸
        """
        t = np.arange(len(signal_data)) / self.sample_rate

        plt.figure(figsize=figsize)
        plt.plot(t, signal_data)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_fft(self, signal_data, title="FFT Spectrum", figsize=(12, 4)):
        """
        绘制FFT频谱

        Args:
            signal_data: 输入信号
            title: 图表标题
            figsize: 图表尺寸
        """
        freq, amplitude = self.compute_fft(signal_data)

        plt.figure(figsize=figsize)
        plt.plot(freq, amplitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_spectrogram(self, signal_data, title="Spectrogram", figsize=(12, 6)):
        """
        绘制时频图

        Args:
            signal_data: 输入信号
            title: 图表标题
            figsize: 图表尺寸
        """
        plt.figure(figsize=figsize)
        plt.specgram(
            signal_data, Fs=self.sample_rate, NFFT=256, noverlap=128, cmap="viridis"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.colorbar(label="Intensity (dB)")
        plt.tight_layout()
        plt.show()
