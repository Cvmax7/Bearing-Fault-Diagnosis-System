<template>
  <div id="app">
    <el-container>
      <el-header>
        <h1>轴承故障检测与预测性维护系统</h1>
        <div class="system-status">
          <span>转速: {{ rpmValue }} RPM</span>
          <span>温度: {{ temperature }}°C</span>
          <span class="status" :class="statusClass">
            状态: {{ currentStatus }}
          </span>
        </div>
      </el-header>
      
      <el-main>
        <el-row :gutter="20">
          <!-- 实时振动数据图表 -->
          <el-col :span="16">
            <el-card class="chart-card">
              <div slot="header">
                <span>三轴振动加速度</span>
                <el-button 
                  style="float: right; margin-left: 10px" 
                  size="small"
                  @click="toggleRealtime">
                  {{ isRealtime ? '暂停' : '实时' }}
                </el-button>
              </div>
              <div ref="vibrationChart" style="height: 300px"></div>
            </el-card>
          </el-col>
          
          <!-- 故障诊断结果 -->
          <el-col :span="8">
            <el-card class="status-card">
              <div slot="header">故障诊断</div>
              <div class="diagnosis-result">
                <div class="current-status">
                  <h3>当前状态</h3>
                  <div class="status-indicator" :class="statusClass">
                    {{ currentStatus }}
                  </div>
                  <div class="confidence">
                    可信度: {{ (confidence * 100).toFixed(2) }}%
                  </div>
                </div>
                <el-divider></el-divider>
                <div class="fault-history">
                  <h3>故障历史</h3>
                  <el-table :data="faultHistory" style="width: 100%" size="mini">
                    <el-table-column prop="timestamp" label="时间" width="140"></el-table-column>
                    <el-table-column prop="prediction" label="故障类型"></el-table-column>
                  </el-table>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
        
        <el-row :gutter="20" style="margin-top: 20px">
          <!-- 频域分析图表 -->
          <el-col :span="12">
            <el-card class="chart-card">
              <div slot="header">频域分析</div>
              <div ref="fftChart" style="height: 300px"></div>
            </el-card>
          </el-col>
          
          <!-- 控制面板 -->
          <el-col :span="12">
            <el-card class="control-card">
              <div slot="header">系统控制</div>
              <div class="control-panel">
                <div class="rpm-control">
                  <span>转速控制:</span>
                  <el-slider 
                    v-model="rpmValue" 
                    :min="0" 
                    :max="3000" 
                    :step="50"
                    show-input>
                  </el-slider>
                  <el-button type="primary" @click="setRPM">设置转速</el-button>
                </div>
                <el-divider></el-divider>
                <div class="data-export">
                  <el-button type="success" @click="exportData">导出数据</el-button>
                  <el-button type="info" @click="showStats">查看统计</el-button>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-main>
    </el-container>
    
    <!-- 统计对话框 -->
    <el-dialog title="系统统计" :visible.sync="statsDialogVisible" width="60%">
      <div v-if="stats" class="stats-container">
        <div class="stats-item">
          <h3>总记录数</h3>
          <div class="stats-value">{{ stats.total_records }}</div>
        </div>
        <div class="stats-item">
          <h3>故障分布</h3>
          <div ref="faultDistChart" style="height: 300px; width: 100%"></div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import * as echarts from 'echarts';
import axios from 'axios';

export default {
  name: 'App',
  data() {
    return {
      // 图表实例
      vibrationChart: null,
      fftChart: null,
      faultDistChart: null,
      
      // 数据状态
      vibrationData: {
        timestamps: [],
        x: [],
        y: [],
        z: []
      },
      
      // FFT数据
      fftData: {
        frequency: [],
        amplitude: []
      },
      
      // 系统状态
      currentStatus: '正常',
      confidence: 1.0,
      rpmValue: 1500,
      temperature: 25.0,
      isRealtime: true,
      
      // WebSocket连接
      ws: null,
      
      // 故障历史
      faultHistory: [],
      
      // 统计数据
      stats: null,
      statsDialogVisible: false,
    };
  },
  computed: {
    statusClass() {
      if (this.currentStatus === '正常') return 'status-normal';
      return 'status-fault';
    }
  },
  mounted() {
    // 初始化图表
    this.initCharts();
    
    // 连接WebSocket
    this.connectWebSocket();
    
    // 获取初始数据
    this.fetchInitialData();
  },
  beforeDestroy() {
    // 关闭WebSocket连接
    if (this.ws) {
      this.ws.close();
    }
    
    // 销毁图表实例
    if (this.vibrationChart) {
      this.vibrationChart.dispose();
    }
    if (this.fftChart) {
      this.fftChart.dispose();
    }
  },
  methods: {
    initCharts() {
      // 初始化振动图表
      this.vibrationChart = echarts.init(this.$refs.vibrationChart);
      const vibrationOption = {
        title: {
          text: '实时振动加速度'
        },
        tooltip: {
          trigger: 'axis'
        },
        legend: {
          data: ['X轴', 'Y轴', 'Z轴']
        },
        xAxis: {
          type: 'category',
          data: []
        },
        yAxis: {
          type: 'value',
          name: '加速度 (g)'
        },
        series: [
          {
            name: 'X轴',
            type: 'line',
            data: [],
            smooth: true,
            lineStyle: {
              width: 2
            }
          },
          {
            name: 'Y轴',
            type: 'line',
            data: [],
            smooth: true,
            lineStyle: {
              width: 2
            }
          },
          {
            name: 'Z轴',
            type: 'line',
            data: [],
            smooth: true,
            lineStyle: {
              width: 2
            }
          }
        ]
      };
      this.vibrationChart.setOption(vibrationOption);
      
      // 初始化FFT图表
      this.fftChart = echarts.init(this.$refs.fftChart);
      const fftOption = {
        title: {
          text: '振动信号频谱分析'
        },
        tooltip: {
          trigger: 'axis'
        },
        xAxis: {
          type: 'value',
          name: '频率 (Hz)'
        },
        yAxis: {
          type: 'value',
          name: '幅值'
        },
        series: [
          {
            type: 'line',
            data: [],
            smooth: true,
            areaStyle: {}
          }
        ]
      };
      this.fftChart.setOption(fftOption);
    },
    
    connectWebSocket() {
      this.ws = new WebSocket('ws://localhost:8000/ws');
      
      this.ws.onopen = () => {
        console.log('WebSocket连接已建立');
      };
      
      this.ws.onmessage = (event) => {
        if (!this.isRealtime) return;
        
        const data = JSON.parse(event.data);
        this.updateData(data);
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket连接已关闭');
        // 尝试重新连接
        setTimeout(() => {
          this.connectWebSocket();
        }, 5000);
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket错误:', error);
      };
    },
    
    updateData(data) {
      // 更新状态信息
      this.currentStatus = data.prediction || '正常';
      this.confidence = data.confidence || 1.0;
      this.temperature = data.temperature;
      
      // 格式化时间
      const time = new Date(data.timestamp * 1000).toLocaleTimeString();
      
      // 更新振动数据
      this.vibrationData.timestamps.push(time);
      this.vibrationData.x.push(data.acceleration_x);
      this.vibrationData.y.push(data.acceleration_y);
      this.vibrationData.z.push(data.acceleration_z);
      
      // 限制数据点数量
      if (this.vibrationData.timestamps.length > 50) {
        this.vibrationData.timestamps.shift();
        this.vibrationData.x.shift();
        this.vibrationData.y.shift();
        this.vibrationData.z.shift();
      }
      
      // 计算FFT（简单示例 - 实际应用中应使用更复杂的FFT算法）
      if (data.fft_data) {
        this.fftData.frequency = data.fft_data.frequency;
        this.fftData.amplitude = data.fft_data.amplitude;
      } else {
        // 如果后端没有提供FFT数据，可以在前端简单模拟
        this.simulateFFT(this.vibrationData.x.slice(-128));
      }
      
      // 更新图表
      this.updateCharts();
      
      // 如果是故障，添加到历史记录
      if (this.currentStatus !== '正常') {
        this.faultHistory.unshift({
          timestamp: time,
          prediction: this.currentStatus
        });
        
        // 限制历史记录数量
        if (this.faultHistory.length > 5) {
          this.faultHistory.pop();
        }
      }
    },
    
    simulateFFT(signal) {
      // 简单的FFT模拟
      const n = signal.length;
      const frequencies = [];
      const amplitudes = [];
      
      // 采样率假设为1000Hz
      const fs = 1000;
      
      for (let i = 0; i < n/2; i++) {
        frequencies.push(i * fs / n);
        
        // 模拟幅值计算（实际应用中应使用正确的FFT算法）
        let amplitude = 0;
        for (let j = 0; j < n; j++) {
          amplitude += signal[j] * Math.cos(2 * Math.PI * i * j / n);
          amplitude += signal[j] * Math.sin(2 * Math.PI * i * j / n);
        }
        amplitude = Math.sqrt(amplitude * amplitude) / n;
        amplitudes.push(amplitude);
      }
      
      this.fftData.frequency = frequencies;
      this.fftData.amplitude = amplitudes;
    },
    
    updateCharts() {
      // 更新振动图表
      this.vibrationChart.setOption({
        xAxis: {
          data: this.vibrationData.timestamps
        },
        series: [
          { data: this.vibrationData.x },
          { data: this.vibrationData.y },
          { data: this.vibrationData.z }
        ]
      });
      
      // 更新FFT图表
      this.fftChart.setOption({
        series: [
          {
            data: this.fftData.frequency.map((f, i) => [f, this.fftData.amplitude[i]])
          }
        ]
      });
    },
    
    fetchInitialData() {
      axios.get('http://localhost:8000/api/data/recent')
        .then(response => {
          const data = response.data;
          if (data && data.length > 0) {
            // 更新故障历史
            this.faultHistory = data
              .filter(item => item.prediction !== '正常')
              .slice(0, 5)
              .map(item => ({
                timestamp: new Date(item.timestamp).toLocaleString(),
                prediction: item.prediction
              }));
            
            // 获取最新状态
            const latest = data[0];
            this.currentStatus = latest.prediction;
            this.confidence = latest.confidence;
            this.temperature = latest.temperature;
            this.rpmValue = latest.rpm;
          }
        })
        .catch(error => {
          console.error('获取初始数据失败:', error);
        });
    },
    
    toggleRealtime() {
      this.isRealtime = !this.isRealtime;
    },
    
    setRPM() {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'set_rpm',
          value: this.rpmValue
        }));
        
        this.$message({
          message: `转速已设置为 ${this.rpmValue} RPM`,
          type: 'success'
        });
      } else {
        this.$message.error('WebSocket连接已断开，无法设置转速');
      }
    },
    
    exportData() {
      axios.get('http://localhost:8000/api/data/recent', { params: { limit: 1000 } })
        .then(response => {
          const data = response.data;
          if (data && data.length > 0) {
            // 将数据转换为CSV格式
            const csvContent = this.convertToCSV(data);
            
            // 创建下载链接
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `bearing_data_${new Date().toISOString()}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
          } else {
            this.$message.warning('没有数据可导出');
          }
        })
        .catch(error => {
          console.error('导出数据失败:', error);
          this.$message.error('导出数据失败');
        });
    },
    
    convertToCSV(data) {
      const headers = ['Timestamp', 'Acceleration X', 'Acceleration Y', 'Acceleration Z', 'Temperature', 'RPM', 'Prediction', 'Confidence'];
      
      const rows = data.map(item => [
        new Date(item.timestamp).toISOString(),
        item.acceleration_x,
        item.acceleration_y,
        item.acceleration_z,
        item.temperature,
        item.rpm,
        item.prediction,
        item.confidence
      ]);
      
      return [
        headers.join(','),
        ...rows.map(row => row.join(','))
      ].join('\n');
    },
    
    showStats() {
      axios.get('http://localhost:8000/api/data/stats')
        .then(response => {
          this.stats = response.data;
          this.statsDialogVisible = true;
          
          // 在下一个渲染周期初始化图表
          this.$nextTick(() => {
            if (this.faultDistChart) {
              this.faultDistChart.dispose();
            }
            
            this.faultDistChart = echarts.init(this.$refs.faultDistChart);
            
            const data = Object.entries(this.stats.fault_distribution).map(([name, value]) => ({
              name,
              value
            }));
            
            this.faultDistChart.setOption({
              title: {
                text: '故障类型分布',
                left: 'center'
              },
              tooltip: {
                trigger: 'item',
                formatter: '{a} <br/>{b} : {c} ({d}%)'
              },
              series: [
                {
                  name: '故障类型',
                  type: 'pie',
                  radius: '60%',
                  center: ['50%', '50%'],
                  data: data,
                  emphasis: {
                    itemStyle: {
                      shadowBlur: 10,
                      shadowOffsetX: 0,
                      shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                  }
                }
              ]
            });
          });
        })
        .catch(error => {
          console.error('获取统计数据失败:', error);
          this.$message.error('获取统计数据失败');
        });
    }
  }
};
</script>

<style>
.el-header {
  background-color: #409EFF;
  color: white;
  line-height: 60px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
}

.system-status {
  display: flex;
  gap: 20px;
}

.status {
  font-weight: bold;
}

.status-normal {
  color: #67C23A;
}

.status-fault {
  color: #F56C6C;
}

.chart-card, .status-card, .control-card {
  margin-bottom: 20px;
}

.status-indicator {
  font-size: 24px;
  font-weight: bold;
  text-align: center;
  padding: 10px;
  border-radius: 5px;
  margin: 10px 0;
}

.confidence {
  text-align: center;
  margin-bottom: 10px;
}

.control-panel {
  padding: 20px;
}

.rpm-control {
  margin-bottom: 20px;
}

.stats-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.stats-item {
  text-align: center;
}

.stats-value {
  font-size: 24px;
  font-weight: bold;
}
</style> 