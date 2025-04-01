import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import json
import torch
from datetime import datetime

# 创建必要的目录
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 导入自定义模块
from data_processing import DatasetProcessor
from signal_processing import SignalProcessor
from enhanced_model_training import (
    ModelTrainer,
    BearingFaultDetector,
    CNNModel,
    RNNModel,
)
from data_simulator import VibrationDataSimulator


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="轴承故障检测系统")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "simulate", "all"],
        help="运行模式: train(训练), evaluate(评估), simulate(模拟), all(全部)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_lstm",
        choices=["cnn", "lstm", "cnn_lstm", "rnn", "resnet", "all"],
        help="模型类型: cnn, lstm, cnn_lstm, rnn, resnet, all(全部)",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="cwru",
        choices=["cwru", "simulated", "both"],
        help="数据类型: cwru(真实数据), simulated(模拟数据), both(两者)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument(
        "--extract_features", action="store_true", help="是否提取特征而不是使用原始波形"
    )
    return parser.parse_args()


def download_and_prepare_data(args):
    """下载并准备数据集"""
    print("=== 数据集准备 ===")
    processor = DatasetProcessor()

    datasets = {}

    if args.data_type in ["cwru", "both"]:
        # 下载CWRU数据集
        processor.download_cwru_dataset()

        # 加载并预处理数据
        print("\n准备CWRU训练数据...")
        X_train, y_train, X_val, y_val, X_test, y_test = (
            processor.prepare_data_for_training(extract_features=args.extract_features)
        )

        print(f"训练集: {X_train.shape}, {y_train.shape}")
        print(f"验证集: {X_val.shape}, {y_val.shape}")
        print(f"测试集: {X_test.shape}, {y_test.shape}")

        # 显示样本类别分布
        print("\nCWRU训练集样本分布:")
        for label, count in zip(*np.unique(y_train, return_counts=True)):
            print(f"类别 {label}: {count} 个样本")

        datasets["cwru"] = (X_train, y_train, X_val, y_val, X_test, y_test)

    if args.data_type in ["simulated", "both"]:
        # 生成模拟数据
        print("\n生成模拟PHM数据...")
        X_sim, y_sim = processor.simulate_phm_data(num_samples=2000, fault_ratio=0.3)

        # 划分数据集
        from sklearn.model_selection import train_test_split

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_sim, y_sim, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        print(f"模拟训练集: {X_train.shape}, {y_train.shape}")
        print(f"模拟验证集: {X_val.shape}, {y_val.shape}")
        print(f"模拟测试集: {X_test.shape}, {y_test.shape}")

        # 显示样本类别分布
        print("\n模拟数据集样本分布:")
        for label, count in zip(*np.unique(y_train, return_counts=True)):
            print(f"类别 {label}: {count} 个样本")

        datasets["simulated"] = (X_train, y_train, X_val, y_val, X_test, y_test)

    # 可视化一些样本
    if args.data_type == "cwru":
        visualize_samples(*datasets["cwru"][:2])
    elif args.data_type == "simulated":
        visualize_samples(*datasets["simulated"][:2])
    else:
        visualize_samples(*datasets["cwru"][:2], title="CWRU数据样本")
        visualize_samples(*datasets["simulated"][:2], title="模拟数据样本")

    return datasets


def train_models(datasets, args):
    """训练并评估多种模型"""
    print("\n=== 模型训练与评估 ===")

    results = {}
    models = {}

    # 确定要训练的模型
    model_types = (
        ["cnn", "lstm", "cnn_lstm", "rnn", "resnet"]
        if args.model == "all"
        else [args.model]
    )

    # 确定要使用的数据集
    data_keys = list(datasets.keys())

    for data_key in data_keys:
        print(f"\n使用{data_key}数据集训练模型...")
        X_train, y_train, X_val, y_val, X_test, y_test = datasets[data_key]

        for model_type in model_types:
            print(f"\n训练{model_type}模型...")

            # 初始化模型训练器
            trainer = ModelTrainer(
                model_type=model_type,
                input_size=1,
                seq_length=X_train.shape[1],
                num_classes=len(np.unique(y_train)),
                learning_rate=0.001,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
            )

            # 准备数据
            train_loader, val_loader, test_loader = trainer.prepare_data(
                X_train, y_train, X_val, y_val, X_test, y_test
            )

            # 训练模型
            history = trainer.train(train_loader, val_loader)

            # 绘制训练历史
            trainer.plot_training_history(history)

            # 评估模型
            metrics = trainer.evaluate(test_loader)

            # 保存模型
            model_path = trainer.save_model(f"models/{data_key}_{model_type}_model.pth")

            # 保存最佳模型
            best_model_path = f"models/{data_key}_{model_type}_best_model.pth"
            trainer.save_model(best_model_path)

            # 导出ONNX模型
            onnx_path = trainer.export_onnx(
                f"models/{data_key}_{model_type}_model.onnx"
            )

            # 记录结果
            results[f"{data_key}_{model_type}"] = metrics
            models[f"{data_key}_{model_type}"] = trainer

    # 比较不同模型的性能
    if len(models) > 1:
        compare_models_performance(results)

    return models, results


def compare_models_performance(results):
    """比较不同模型的性能"""
    print("\n=== 模型性能比较 ===")

    # 提取准确率、精确率、召回率和F1分数
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for model_name, metrics in results.items():
        model_names.append(model_name)
        accuracies.append(metrics["accuracy"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_scores.append(metrics["f1"])

    # 绘制比较图
    plt.figure(figsize=(12, 8))

    x = np.arange(len(model_names))
    width = 0.2

    plt.bar(x - 1.5 * width, accuracies, width, label="准确率")
    plt.bar(x - 0.5 * width, precisions, width, label="精确率")
    plt.bar(x + 0.5 * width, recalls, width, label="召回率")
    plt.bar(x + 1.5 * width, f1_scores, width, label="F1分数")

    plt.xlabel("模型")
    plt.ylabel("分数")
    plt.title("不同模型性能比较")
    plt.xticks(x, model_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/model_comparison.png")
    plt.show()


def visualize_samples(X, y, num_samples=5, title="数据样本可视化"):
    """可视化数据样本"""
    print(f"\n=== {title} ===")

    # 创建信号处理器
    processor = SignalProcessor()

    # 为每个类别选择样本
    labels = np.unique(y)

    plt.figure(figsize=(15, 10))
    for i, label in enumerate(labels):
        # 获取当前类别的索引
        indices = np.where(y == label)[0]

        # 随机选择样本
        if len(indices) > num_samples:
            indices = np.random.choice(indices, num_samples, replace=False)

        # 绘制样本
        for j, idx in enumerate(indices):
            plt.subplot(len(labels), num_samples, i * num_samples + j + 1)
            plt.plot(X[idx])
            plt.title(f"类别 {label} - 样本 {j+1}")
            plt.xticks([])

    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_')}.png")
    plt.show()

    # 也为第一个样本绘制频谱
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(labels):
        indices = np.where(y == label)[0]
        if len(indices) > 0:
            idx = indices[0]
            plt.subplot(len(labels), 2, 2 * i + 1)
            plt.plot(X[idx])
            plt.title(f"类别 {label} - 时域波形")

            plt.subplot(len(labels), 2, 2 * i + 2)
            freq, amp = processor.compute_fft(X[idx])
            plt.plot(freq, amp)
            plt.title(f"类别 {label} - 频域谱")

    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_')}_spectra.png")
    plt.show()


def simulate_real_time_detection(models, args):
    """模拟实时故障检测"""
    print("\n=== 模拟实时故障检测 ===")

    # 选择一个模型进行故障检测
    if args.model != "all":
        model_key = (
            f"{args.data_type}_{args.model}"
            if args.data_type != "both"
            else f"cwru_{args.model}"
        )
        if model_key in models:
            trainer = models[model_key]
        else:
            print(f"未找到{model_key}模型，使用第一个可用模型")
            trainer = list(models.values())[0]
    else:
        # 使用性能最好的模型
        print("使用CNN-LSTM混合模型进行实时检测")
        model_key = (
            f"cwru_cnn_lstm" if "cwru_cnn_lstm" in models else list(models.keys())[0]
        )
        trainer = models[model_key]

    # 创建数据模拟器
    simulator = VibrationDataSimulator()

    # 创建信号处理器
    processor = SignalProcessor()

    # 故障类型映射
    fault_types = ["正常", "内圈故障", "外圈故障", "滚动体故障"]

    # 模拟实时采集和检测
    print("\n开始模拟实时故障检测 (按 Ctrl+C 停止)...")
    try:
        sample_buffer = []
        sample_size = 1024

        # 记录预测结果
        prediction_history = []

        plt.figure(figsize=(12, 8))
        plt.ion()  # 打开交互模式

        for i in range(300):  # 模拟300个时间步
            # 随机设置故障类型，使模拟更有趣
            if i % 30 == 0:
                simulator.fault_type = np.random.choice(fault_types)
                print(f"\n===== 切换故障类型: {simulator.fault_type} =====")

            # 生成模拟数据
            data = simulator.generate_vibration_data()

            # 将振动数据添加到缓冲区
            sample_buffer.append(data["acceleration_x"])

            # 当缓冲区达到样本大小时进行检测
            if len(sample_buffer) >= sample_size:
                # 准备数据
                signal = np.array(sample_buffer[-sample_size:])

                # 进行预测
                predictions, probabilities = trainer.predict(np.array([signal]))
                predicted_class = predictions[0]
                confidence = np.max(probabilities[0])

                # 记录预测结果
                prediction_history.append(
                    {
                        "time": time.time(),
                        "prediction": predicted_class,
                        "confidence": confidence,
                        "rpm": data["rpm"],
                        "temperature": data["temperature"],
                        "true_fault": fault_types.index(simulator.fault_type),
                    }
                )

                # 清除旧图
                plt.clf()

                # 绘制信号
                plt.subplot(2, 2, 1)
                plt.plot(signal)
                plt.title(
                    f"振动信号 - 转速: {data['rpm']:.1f} RPM, 温度: {data['temperature']:.1f}°C"
                )
                plt.xlabel("采样点")
                plt.ylabel("振幅")

                # 绘制频谱
                plt.subplot(2, 2, 2)
                freq, amplitude = processor.compute_fft(signal)
                plt.plot(freq, amplitude)
                plt.title(
                    f"频谱分析 - 诊断结果: {fault_types[predicted_class]} (置信度: {confidence:.2f})"
                )
                plt.xlabel("频率 (Hz)")
                plt.ylabel("幅值")

                # 绘制包络谱
                plt.subplot(2, 2, 3)
                env_freq, env_amp = processor.compute_envelope_spectrum(signal)
                plt.plot(env_freq, env_amp)
                plt.title("包络谱分析")
                plt.xlabel("频率 (Hz)")
                plt.ylabel("幅值")

                # 绘制预测历史
                plt.subplot(2, 2, 4)
                if len(prediction_history) > 1:
                    history_length = min(50, len(prediction_history))
                    pred_history = prediction_history[-history_length:]
                    plt.plot(
                        [i for i in range(history_length)],
                        [p["prediction"] for p in pred_history],
                        "bo-",
                        label="预测",
                    )
                    plt.plot(
                        [i for i in range(history_length)],
                        [p["true_fault"] for p in pred_history],
                        "r--",
                        label="实际",
                    )
                    plt.yticks(range(len(fault_types)), fault_types)
                    plt.title("故障预测历史")
                    plt.xlabel("时间步")
                    plt.legend()

                plt.tight_layout()
                plt.pause(0.05)

                # 输出结果
                print(
                    f"\r诊断结果: {fault_types[predicted_class]}, 置信度: {confidence:.2f}, 实际状态: {simulator.fault_type}",
                    end="",
                )

                # 每30步保存一次预测历史图
                if i % 30 == 0 and i > 0:
                    plt.savefig(f"results/real_time_detection_{i}.png")

            # 模拟时间延迟
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n模拟已停止")
    finally:
        plt.ioff()
        plt.close()

        # 保存最终预测历史图
        plt.figure(figsize=(10, 6))
        if len(prediction_history) > 1:
            plt.plot(
                [i for i in range(len(prediction_history))],
                [p["prediction"] for p in prediction_history],
                "bo-",
                label="预测",
            )
            plt.plot(
                [i for i in range(len(prediction_history))],
                [p["true_fault"] for p in prediction_history],
                "r--",
                label="实际",
            )
            plt.yticks(range(len(fault_types)), fault_types)
            plt.title("完整故障预测历史")
            plt.xlabel("时间步")
            plt.legend()
            plt.tight_layout()
            plt.savefig("results/complete_prediction_history.png")
            plt.show()

        # 计算预测准确率
        if len(prediction_history) > 0:
            correct = sum(
                1 for p in prediction_history if p["prediction"] == p["true_fault"]
            )
            accuracy = correct / len(prediction_history)
            print(f"\n模拟测试准确率: {accuracy:.4f}")

        # 保存预测历史到文件
        with open("results/prediction_history.json", "w") as f:
            json.dump(prediction_history, f)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    print("\n" + "=" * 50)
    print("轴承故障检测系统")
    print("=" * 50)

    # 确保目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 根据运行模式执行相应操作
    if args.mode in ["train", "all"]:
        # 下载并准备数据
        datasets = download_and_prepare_data(args)

        # 训练模型
        models, results = train_models(datasets, args)

        if args.mode == "train":
            return

    if args.mode in ["evaluate", "all"]:
        # 如果只是评估模式，尝试加载已训练的模型
        if args.mode == "evaluate":
            # 查找模型文件
            model_files = [
                f
                for f in os.listdir("models")
                if f.endswith(".pth") and "best_model" in f
            ]
            if not model_files:
                print("未找到训练好的模型，请先运行训练模式")
                return

            models = {}
            for model_file in model_files:
                model_type = model_file.split("_")[1]
                data_type = model_file.split("_")[0]

                trainer = ModelTrainer(model_type=model_type)
                trainer.load_model(os.path.join("models", model_file))

                models[f"{data_type}_{model_type}"] = trainer

            # 为了评估，也需要加载数据
            datasets = download_and_prepare_data(args)

            # 评估模型
            results = {}
            for model_name, trainer in models.items():
                data_type = model_name.split("_")[0]
                if data_type in datasets:
                    _, _, _, _, X_test, y_test = datasets[data_type]
                    test_loader = trainer.prepare_data(
                        None, None, None, None, X_test, y_test
                    )[2]
                    results[model_name] = trainer.evaluate(test_loader)

            # 比较模型性能
            if len(results) > 1:
                compare_models_performance(results)

    if args.mode in ["simulate", "all"]:
        # 如果只是模拟模式，尝试加载已训练的模型
        if args.mode == "simulate" and "models" not in locals():
            # 查找模型文件
            model_files = [
                f
                for f in os.listdir("models")
                if f.endswith(".pth") and "best_model" in f
            ]
            if not model_files:
                print("未找到训练好的模型，请先运行训练模式")
                return

            models = {}
            for model_file in model_files:
                parts = model_file.split("_")
                data_type = parts[0]
                model_type = parts[1]

                trainer = ModelTrainer(model_type=model_type)
                trainer.load_model(os.path.join("models", model_file))

                models[f"{data_type}_{model_type}"] = trainer

        # 模拟实时故障检测
        simulate_real_time_detection(models, args)


if __name__ == "__main__":
    main()
