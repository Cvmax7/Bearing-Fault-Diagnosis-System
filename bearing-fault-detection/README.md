# 轴承故障检测与预测性维护系统
## 系统概述
这是一个基于振动信号分析和深度学习的轴承故障检测系统，可以实时监测轴承运行状态并进行故障诊断，提高设备的可靠性和运行效率。系统支持西储大学轴承振动数据和PHM平台振动数据的处理和分析。

## 系统架构
系统由以下几个主要模块组成：
1.数据采集模块：从传感器采集振动数据或生成模拟数据
2.数据处理模块：对原始数据进行预处理和特征提取
3.信号处理模块：实现FFT、小波变换、包络分析等信号处理功能
4.深度学习模块：支持多种深度学习模型用于故障诊断
5.后端服务模块：基于FastAPI提供REST API和WebSocket服务
6.前端可视化模块：基于Vue.js的Web界面展示监测结果

## 功能特点
1.数据多样性：支持处理CWRU轴承数据集和模拟PHM数据集
2.多种模型选择：包括CNN、LSTM、CNN-LSTM混合模型、RNN等深度学习模型
3.丰富的信号处理功能：
    -时域分析：RMS、峰值、峭度、波形因子等
    -频域分析：FFT频谱、包络谱等
    -时频分析：小波变换、时频图
4.实时监测与离线分析：支持实时数据流处理和历史数据分析
5.可视化界面：波形图、频谱图、故障概率分布等多种可视化方式
6.故障诊断：能够检测和分类正常、内圈故障、外圈故障、滚动体故障等轴承状态
7.预警系统：当检测到故障时发出警报，并提供故障详情


## 安装和配置
环境要求：

- Python 3.7+
- PyTorch 1.7+
- Node.js 12+ (前端开发)
- MQTT代理 (如Mosquitto，用于数据传输)

# 安装依赖

## 克隆仓库

git clone https://github.com/yourusername/bearing-fault-detection.git
cd bearing-fault-detection

## 安装Python依赖

pip install -r requirements.txt

## 安装前端依赖

cd frontend
npm install

# 使用指南

## 数据准备
系统支持两种数据来源：

- CWRU轴承数据集: 程序会自动下载和处理
- 模拟数据: 系统自动生成用于测试

# 模型训练

## 使用CWRU数据训练CNN-LSTM混合模型

python main.py --mode train --model cnn_lstm --data_type cwru

## 使用模拟数据训练所有模型
python main.py --mode train --model all --data_type simulated

## 使用原始波形而非提取特征进行训练
python main.py --mode train --model cnn_lstm --data_type both --extract_features


# 模型评估

## 评估已训练模型在CWRU数据上的性能
python main.py --mode evaluate --data_type cwru

## 比较不同模型在模拟数据上的性能
python main.py --mode evaluate --data_type simulated

# 实时仿真与监测
## 使用训练好的模型进行实时故障检测模拟
python main.py --mode simulate --model cnn_lstm

## 运行完整工作流程(训练、评估、模拟)
python main.py --mode all --model cnn_lstm --data_type both

# 启动Web界面
## 启动后端服务
python app.py

## 启动前端开发服务器
cd frontend
npm run serve

## 访问 http://localhost:8080 查看Web界面

# 系统组件详解

数据处理模块 (data_processing.py)

- 下载和处理CWRU轴承数据集
- 生成模拟PHM数据
- 提取时域特征和频域特征
- 数据分割和预处理

信号处理模块 (signal_processing.py)

- FFT频谱分析
- 包络谱分析
- 小波变换
- 时域和频域特征提取
- 信号滤波
- 可视化功能

模型训练模块 (enhanced_model_training.py)
-支持CNN、LSTM、CNN-LSTM、RNN等多种模型
-高级训练策略
-模型评估与比较
-模型保存与加载
ONNX导出支持

数据采集/模拟模块 (data_collection.py / data_simulator.py)
-串口通信读取传感器数据
-MQTT消息发布与订阅
-模拟不同类型的轴承故障信号

后端服务 (app.py)
-RESTful API
-WebSocket实时通信
-数据持久化
故障诊断服务

前端界面 (src/App.vue)
-实时数据展示
-故障诊断结果可视化
-历史数据查询
-设备参数控制





# 项目结构

bearing-fault-detection/
├── app.py                    # FastAPI后端服务
├── data_collection.py        # 数据采集模块
├── data_processing.py        # 数据处理模块
├── data_simulator.py         # 数据模拟模块
├── enhanced_model_training.py # 模型训练模块
├── main.py                   # 主程序
├── signal_processing.py      # 信号处理模块
├── requirements.txt          # Python依赖
├── README.md                 # 项目说明
├── datasets/                 # 数据集目录
├── models/                   # 模型保存目录
├── results/                  # 结果输出目录
└── frontend/                 # 前端源码
    ├── src/
    │   ├── App.vue           # 主界面组件
    │   ├── components/       # 界面组件
    │   ├── assets/           # 静态资源
    │   └── ...
    └── ...

​    