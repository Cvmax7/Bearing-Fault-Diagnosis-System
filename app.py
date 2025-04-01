import uvicorn
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json
import asyncio
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import paho.mqtt.client as mqtt
import os
import nest_asyncio
from fastapi.responses import JSONResponse

# 初始化FastAPI应用
app = FastAPI(title="轴承故障检测系统API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 数据库模型
Base = declarative_base()


class BearingData(Base):
    __tablename__ = "bearing_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    acceleration_x = Column(Float)
    acceleration_y = Column(Float)
    acceleration_z = Column(Float)
    temperature = Column(Float)
    rpm = Column(Float)
    prediction = Column(String)
    confidence = Column(Float)


# 数据库连接
SQLALCHEMY_DATABASE_URL = "sqlite:///./bearing.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 确保表存在
Base.metadata.create_all(bind=engine)


# 依赖项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 全局变量
connected_clients = set()
latest_data = {}
model = None

# 保存主事件循环引用
main_event_loop = None

# 在文件顶部添加
nest_asyncio.apply()


# 加载模型
def load_model():
    global model
    model_path = "./bearing_fault_model.pth"

    if not os.path.exists(model_path):
        print(f"警告: 模型文件 {model_path} 不存在，请先训练模型")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从正确的模块导入BearingFaultDetector类
    from enhanced_model_training import BearingFaultDetector

    # 加载整个checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # 从checkpoint中提取模型参数
    if "model_state_dict" in checkpoint:
        # 创建模型
        model = BearingFaultDetector(
            input_size=checkpoint.get("input_size", 3),
            seq_length=checkpoint.get("seq_length", 1024),
            num_classes=checkpoint.get("num_classes", 4),
        )
        # 加载模型参数
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # 如果是直接的state_dict
        model = BearingFaultDetector()
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("模型加载成功！")


# MQTT客户端回调
def on_mqtt_connect(client, userdata, flags, rc):
    print(f"MQTT连接成功，返回码: {rc}")
    client.subscribe("bearing/vibration")


def on_mqtt_message(client, userdata, msg):
    global latest_data, main_event_loop

    try:
        data = json.loads(msg.payload)

        # 确保数据包含所有必要字段
        if (
            "acceleration_x" not in data
            or "acceleration_y" not in data
            or "acceleration_z" not in data
        ):
            print(f"警告: 收到的数据缺少加速度字段: {data}")
            return

        print(
            f"收到MQTT数据: {data['acceleration_x']:.2f}, {data['acceleration_y']:.2f}, {data['acceleration_z']:.2f}"
        )

        # 确保有预测结果
        if "prediction" not in data:
            data["prediction"] = "正常"
        if "confidence" not in data:
            data["confidence"] = 0.95

        latest_data = data

        # 进行故障预测
        prediction_result = predict_fault(data)
        data.update(prediction_result)

        # 保存到数据库
        save_to_database(data)

        # 使用主事件循环运行协程
        if main_event_loop and main_event_loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_data(data), main_event_loop)
        else:
            print("主事件循环不可用")
    except Exception as e:
        print(f"处理MQTT消息时出错: {e}")


# 初始化MQTT客户端
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message
mqtt_client.connect("localhost", 1883, 60)
mqtt_client.loop_start()


# 故障预测函数
def predict_fault(data):
    if model is None:
        return {"prediction": "未知", "confidence": 0.0}

    try:
        # 创建固定长度的特征向量
        sample_size = 1024  # 应与训练时相同

        # 如果接收到的是单个值，则需要将其转换为适当格式
        if isinstance(data["acceleration_x"], (int, float)):
            # 单个值扩展为数组
            x_data = [data["acceleration_x"]] * sample_size
            y_data = [data["acceleration_y"]] * sample_size
            z_data = [data["acceleration_z"]] * sample_size

            # 合并三轴数据（取平均值）
            combined_data = [(x + y + z) / 3 for x, y, z in zip(x_data, y_data, z_data)]
            features = np.array([combined_data]).reshape(1, 1, sample_size)
        else:
            # 已经是数组形式，同样需要调整为1个通道
            x_data = data["acceleration_x"]
            y_data = data["acceleration_y"]
            z_data = data["acceleration_z"]

            # 合并三轴数据
            combined_data = [(x + y + z) / 3 for x, y, z in zip(x_data, y_data, z_data)]
            features = np.array([combined_data]).reshape(1, 1, -1)

            # 确保长度正确
            if features.shape[2] > sample_size:
                features = features[:, :, :sample_size]
            elif features.shape[2] < sample_size:
                # 如果长度不足，用零填充
                pad_size = sample_size - features.shape[2]
                features = np.pad(features, ((0, 0), (0, 0), (0, pad_size)), "constant")

        # 转换为PyTorch张量
        tensor_data = torch.FloatTensor(features)

        # 使用模型预测
        with torch.no_grad():
            outputs = model(tensor_data)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        # 故障类型映射
        fault_types = {0: "正常", 1: "内圈故障", 2: "外圈故障", 3: "滚动体故障"}

        return {
            "prediction": fault_types[predicted.item()],
            "confidence": confidence.item(),
        }
    except Exception as e:
        print(f"预测过程中出错: {e}")
        return {"prediction": "预测错误", "confidence": 0.0}


# 保存数据到数据库
def save_to_database(data):
    try:
        db = SessionLocal()
        db_data = BearingData(
            timestamp=datetime.fromtimestamp(data["timestamp"]),
            acceleration_x=data["acceleration_x"],
            acceleration_y=data["acceleration_y"],
            acceleration_z=data["acceleration_z"],
            temperature=data["temperature"],
            rpm=data["rpm"],
            prediction=data["prediction"],
            confidence=data["confidence"],
        )
        db.add(db_data)
        db.commit()
    except Exception as e:
        print(f"保存数据到数据库时出错: {e}")
    finally:
        db.close()


# 广播数据到所有WebSocket客户端
async def broadcast_data(data):
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(data))
        except Exception as e:
            print(f"发送数据到WebSocket客户端出错: {e}")


# WebSocket端点
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket连接请求...")

    # 尝试接受连接
    try:
        await websocket.accept()
        print("WebSocket连接已接受")

        # 添加到连接集合
        connected_clients.add(websocket)

        # 发送测试数据
        test_data = {
            "test": True,
            "message": "连接测试成功",
            "time": datetime.now().isoformat(),
        }
        await websocket.send_text(json.dumps(test_data))
        print(f"已发送测试数据: {test_data}")

        # 发送最新数据（如果有）
        if latest_data:
            await websocket.send_text(json.dumps(latest_data))
            print("已发送最新数据")

        # 处理消息循环
        try:
            while True:
                data = await websocket.receive_text()
                print(f"收到WebSocket消息: {data}")

                try:
                    message = json.loads(data)

                    # 处理前端发来的命令
                    if message.get("type") == "set_rpm":
                        # 发送MQTT命令调整转速
                        mqtt_client.publish(
                            "bearing/command",
                            json.dumps(
                                {"type": "set_rpm", "value": message.get("value", 1500)}
                            ),
                        )
                        print(f"发送转速设置命令: {message.get('value', 1500)}")

                    elif message.get("type") == "set_fault":
                        # 发送MQTT命令设置故障类型
                        mqtt_client.publish(
                            "bearing/command",
                            json.dumps(
                                {
                                    "type": "set_fault",
                                    "value": message.get("value", "正常"),
                                }
                            ),
                        )
                        print(f"发送故障类型设置命令: {message.get('value', '正常')}")

                    # 回复确认消息
                    await websocket.send_text(
                        json.dumps({"status": "ok", "message": "命令已处理"})
                    )

                except json.JSONDecodeError:
                    print(f"收到的消息不是有效的JSON: {data}")
                except Exception as e:
                    print(f"处理WebSocket消息时出错: {e}")

        except Exception as e:
            print(f"WebSocket接收消息循环中断: {e}")

    except Exception as e:
        print(f"WebSocket连接建立出错: {e}")

    finally:
        # 确保客户端从集合中移除
        if websocket in connected_clients:
            connected_clients.remove(websocket)
            print("WebSocket连接已关闭并从集合中移除")


# API端点
@app.get("/api/data/recent")
def get_recent_data(limit: int = 100, db: Session = Depends(get_db)):
    """获取最近的传感器数据"""
    data = (
        db.query(BearingData).order_by(BearingData.timestamp.desc()).limit(limit).all()
    )
    return data


@app.get("/api/data/stats")
def get_stats(db: Session = Depends(get_db)):
    """获取统计信息"""
    # 总记录数
    total = db.query(BearingData).count()

    # 故障分布
    fault_counts = (
        db.query(BearingData.prediction, db.func.count(BearingData.id))
        .group_by(BearingData.prediction)
        .all()
    )

    # 最近故障
    latest_fault = (
        db.query(BearingData)
        .filter(BearingData.prediction != "正常")
        .order_by(BearingData.timestamp.desc())
        .first()
    )

    return {
        "total_records": total,
        "fault_distribution": dict(fault_counts),
        "latest_fault": latest_fault,
    }


@app.get("/api/data/latest")
def get_latest_data():
    """获取最新的传感器数据"""
    return latest_data


@app.post("/api/command/set_rpm")
async def set_rpm(data: dict):
    mqtt_client.publish(
        "bearing/command",
        json.dumps({"type": "set_rpm", "value": data.get("value", 1500)}),
    )
    return {"status": "ok"}


@app.post("/api/command/set_fault")
async def set_fault(data: dict):
    mqtt_client.publish(
        "bearing/command",
        json.dumps({"type": "set_fault", "value": data.get("value", "正常")}),
    )
    return {"status": "ok"}


# 启动服务器时加载模型
@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()
    load_model()


@app.middleware("http")
async def add_charset_middleware(request, call_next):
    response = await call_next(request)
    if isinstance(response, JSONResponse):
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


# 主入口
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
