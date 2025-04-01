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

# 初始化FastAPI应用
app = FastAPI(title="轴承故障检测系统API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
SQLALCHEMY_DATABASE_URL = "postgresql://username:password@localhost/bearing_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
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


# 加载模型
def load_model():
    global model
    model_path = "bearing_fault_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设BearingFaultDetector类已经定义
    from model_training import BearingFaultDetector

    model = BearingFaultDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("模型加载成功！")


# MQTT客户端回调
def on_mqtt_connect(client, userdata, flags, rc):
    print(f"MQTT连接成功，返回码: {rc}")
    client.subscribe("bearing/vibration")


def on_mqtt_message(client, userdata, msg):
    global latest_data

    try:
        data = json.loads(msg.payload)
        latest_data = data

        # 进行故障预测
        prediction_result = predict_fault(data)
        data.update(prediction_result)

        # 保存到数据库
        save_to_database(data)

        # 广播到所有连接的WebSocket客户端
        asyncio.create_task(broadcast_data(data))
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
        # 提取特征
        features = np.array(
            [data["acceleration_x"], data["acceleration_y"], data["acceleration_z"]]
        ).reshape(1, 3, -1)

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
    if connected_clients:
        message = json.dumps(data)
        await asyncio.gather(
            *[client.send_text(message) for client in connected_clients]
        )


# WebSocket端点
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        # 发送最新数据
        if latest_data:
            await websocket.send_text(json.dumps(latest_data))

        # 持续接收消息
        while True:
            data = await websocket.receive_text()
            # 处理来自客户端的消息
            command = json.loads(data)
            if command.get("type") == "set_rpm":
                # 发送到MQTT
                mqtt_client.publish("bearing/command", json.dumps(command))
    except Exception as e:
        print(f"WebSocket错误: {e}")
    finally:
        connected_clients.remove(websocket)


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


# 启动服务器时加载模型
@app.on_event("startup")
def startup_event():
    load_model()


# 主入口
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
