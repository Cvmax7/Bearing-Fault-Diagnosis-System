import time
import json
import math
import random
import numpy as np
import paho.mqtt.client as mqtt

# 初始化MQTT客户端
client = mqtt.Client()

# 全局状态
rpm = 1500  # 初始转速
fault_type = "正常"  # 初始故障类型


# MQTT消息回调
def on_message(client, userdata, msg):
    global rpm, fault_type
    try:
        data = json.loads(msg.payload)
        if data.get("type") == "set_rpm":
            rpm = data.get("value", rpm)
            print(f"转速已设置为: {rpm} RPM")
        elif data.get("type") == "set_fault":
            fault_type = data.get("value", fault_type)
            print(f"故障类型已设置为: {fault_type}")
    except Exception as e:
        print(f"处理命令时出错: {e}")


# 连接配置
def on_connect(client, userdata, flags, rc):
    print(f"已连接到MQTT代理，返回码: {rc}")
    client.subscribe("bearing/command")


client.on_connect = on_connect
client.on_message = on_message

# 连接到MQTT代理
try:
    client.connect("localhost", 1883, 60)
    client.loop_start()
    print(f"开始模拟振动数据，初始转速: {rpm} RPM")
except Exception as e:
    print(f"连接MQTT代理失败: {e}")
    exit(1)

# 主循环，模拟数据
while True:
    try:
        # 基础振动幅度与转速相关
        base_amplitude = rpm / 1000

        # 根据故障类型调整振动特性
        if fault_type == "正常":
            x_factor, y_factor, z_factor = 1.0, 1.0, 1.0
        elif fault_type == "内圈故障":
            x_factor, y_factor, z_factor = 2.5, 1.2, 1.0
        elif fault_type == "外圈故障":
            x_factor, y_factor, z_factor = 1.2, 2.5, 1.0
        elif fault_type == "滚动体故障":
            x_factor, y_factor, z_factor = 1.5, 1.5, 2.0
        else:
            x_factor, y_factor, z_factor = 1.0, 1.0, 1.0

        # 生成振动数据
        t = time.time()
        freq = rpm / 60  # 转换为赫兹

        # 添加一些噪声和谐波
        noise_x = np.random.normal(0, 0.2 * base_amplitude)
        noise_y = np.random.normal(0, 0.2 * base_amplitude)
        noise_z = np.random.normal(0, 0.2 * base_amplitude)

        acc_x = x_factor * base_amplitude * math.sin(2 * math.pi * freq * t) + noise_x
        acc_y = (
            y_factor * base_amplitude * math.sin(2 * math.pi * freq * t + math.pi / 3)
            + noise_y
        )
        acc_z = (
            z_factor * base_amplitude * math.sin(2 * math.pi * freq * t + math.pi / 6)
            + noise_z
        )

        # 随机温度，与转速有关
        temperature = 25 + (rpm / 500) + random.uniform(-0.5, 0.5)

        # 创建数据包
        data = {
            "timestamp": time.time(),
            "acceleration_x": acc_x,
            "acceleration_y": acc_y,
            "acceleration_z": acc_z,
            "temperature": temperature,
            "rpm": rpm,
        }

        # 发布数据到MQTT并确认发送成功
        result = client.publish("bearing/vibration", json.dumps(data))
        if result.rc != 0:
            print(f"发送数据失败，错误码: {result.rc}")
            # 尝试重新连接MQTT
            try:
                client.reconnect()
                print("已重新连接到MQTT服务器")
            except Exception as e:
                print(f"MQTT重连失败: {e}")
        else:
            print(
                f"已发送数据: {data['acceleration_x']:.2f}, {data['acceleration_y']:.2f}, {data['acceleration_z']:.2f}"
            )

        time.sleep(0.1)
    except Exception as e:
        print(f"模拟数据生成出错: {e}")
        time.sleep(1)  # 出错后暂停较长时间
