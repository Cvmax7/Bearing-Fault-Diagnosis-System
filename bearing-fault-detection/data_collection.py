import serial
import time
import json
import paho.mqtt.client as mqtt
import numpy as np


class VibrationDataCollector:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600):
        # 初始化串口连接
        self.serial_conn = serial.Serial(port, baudrate)

        # 初始化MQTT客户端
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect("localhost", 1883, 60)

    def read_sensor_data(self):
        """读取传感器数据"""
        if self.serial_conn.in_waiting:
            data = self.serial_conn.readline().decode("utf-8").strip()
            return self._parse_data(data)
        return None

    def _parse_data(self, data_str):
        """解析传感器数据"""
        try:
            data_parts = data_str.split(",")
            timestamp = time.time()

            # 假设数据格式为: x轴加速度,y轴加速度,z轴加速度,温度,转速
            return {
                "timestamp": timestamp,
                "acceleration_x": float(data_parts[0]),
                "acceleration_y": float(data_parts[1]),
                "acceleration_z": float(data_parts[2]),
                "temperature": float(data_parts[3]),
                "rpm": float(data_parts[4]),
            }
        except Exception as e:
            print(f"数据解析错误: {e}")
            return None

    def send_data_to_server(self, data):
        """将数据发送到MQTT服务器"""
        if data:
            self.mqtt_client.publish("bearing/vibration", json.dumps(data))

    def set_rpm(self, rpm_value):
        """设置轴承转速"""
        command = f"SET_RPM:{rpm_value}\n"
        self.serial_conn.write(command.encode())

    def start_collection(self, interval=0.1):
        """开始数据采集循环"""
        try:
            while True:
                data = self.read_sensor_data()
                if data:
                    self.send_data_to_server(data)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("数据采集已停止")
            self.serial_conn.close()
            self.mqtt_client.disconnect()


if __name__ == "__main__":
    collector = VibrationDataCollector()
    collector.start_collection()
