import time
import json
import math
import random
import numpy as np
import paho.mqtt.client as mqtt


class VibrationDataSimulator:
    def __init__(self):
        # 初始化MQTT客户端
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect("localhost", 1883, 60)

        # 订阅命令主题
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.subscribe("bearing/command")
        self.mqtt_client.loop_start()

        # 初始化参数
        self.rpm = 1500
        self.timestamp = time.time()
        self.fault_type = "正常"  # 可以是 "正常", "内圈故障", "外圈故障", "滚动体故障"
        self.is_running = True

        # 模拟周期性信号的参数
        self.base_freq = self.rpm / 60  # 转速频率 (Hz)
        self.sample_rate = 1000  # 采样率 (Hz)
        self.t = 0  # 时间累加器

        # 内圈、外圈和滚动体特征频率的系数
        # 这些系数基于轴承结构，实际应用中需要根据具体轴承类型调整
        self.bpfi_coef = 5.4  # 内圈特征频率系数 (Ball Pass Frequency Inner)
        self.bpfo_coef = 3.6  # 外圈特征频率系数 (Ball Pass Frequency Outer)
        self.bsf_coef = 2.8  # 滚动体特征频率系数 (Ball Spin Frequency)

    def on_message(self, client, userdata, msg):
        """处理接收到的命令"""
        try:
            command = json.loads(msg.payload)
            if command.get("type") == "set_rpm":
                self.rpm = command.get("value", self.rpm)
                self.base_freq = self.rpm / 60
                print(f"转速已设置为 {self.rpm} RPM")
            elif command.get("type") == "set_fault":
                self.fault_type = command.get("value", self.fault_type)
                print(f"故障类型已设置为 {self.fault_type}")
        except Exception as e:
            print(f"处理命令时出错: {e}")

    def generate_vibration_data(self):
        """生成振动数据"""
        # 时间增量
        dt = 1.0 / self.sample_rate
        self.t += dt

        # 基础振动（包含轴承旋转基频）
        base_vibration = math.sin(2 * math.pi * self.base_freq * self.t)

        # 添加随机噪声
        noise = random.uniform(-0.1, 0.1)

        # 根据故障类型添加特征振动
        fault_vibration = 0
        if self.fault_type == "内圈故障":
            # 内圈故障特征频率
            bpfi = self.base_freq * self.bpfi_coef
            fault_vibration = (
                0.5
                * math.sin(2 * math.pi * bpfi * self.t)
                * (1 + 0.2 * math.sin(2 * math.pi * self.base_freq * self.t))
            )
        elif self.fault_type == "外圈故障":
            # 外圈故障特征频率
            bpfo = self.base_freq * self.bpfo_coef
            fault_vibration = 0.4 * math.sin(2 * math.pi * bpfo * self.t)
        elif self.fault_type == "滚动体故障":
            # 滚动体故障特征频率
            bsf = self.base_freq * self.bsf_coef
            fault_vibration = (
                0.3
                * math.sin(2 * math.pi * bsf * self.t)
                * (1 + 0.1 * math.sin(4 * math.pi * self.base_freq * self.t))
            )

        # 合成振动信号
        vib_x = base_vibration + fault_vibration + noise

        # y轴和z轴可以是x轴的变种，但有不同的幅度和相位
        vib_y = 0.7 * base_vibration + 0.5 * fault_vibration + random.uniform(-0.1, 0.1)
        vib_z = 0.5 * base_vibration + 0.3 * fault_vibration + random.uniform(-0.1, 0.1)

        # 模拟温度（随着转速略微升高）
        temperature = 25 + (self.rpm / 3000) * 5 + random.uniform(-0.5, 0.5)

        return {
            "timestamp": time.time(),
            "acceleration_x": vib_x,
            "acceleration_y": vib_y,
            "acceleration_z": vib_z,
            "temperature": temperature,
            "rpm": self.rpm,
        }

    def run(self, interval=0.1):
        """运行模拟器"""
        try:
            print(f"开始模拟振动数据，初始转速: {self.rpm} RPM")
            while self.is_running:
                data = self.generate_vibration_data()
                self.mqtt_client.publish("bearing/vibration", json.dumps(data))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("模拟器已停止")
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.is_running = False


if __name__ == "__main__":
    simulator = VibrationDataSimulator()

    # 可以随机设置故障类型以模拟不同情况
    fault_types = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
    simulator.fault_type = random.choice(fault_types)
    print(f"初始故障类型: {simulator.fault_type}")

    simulator.run()
