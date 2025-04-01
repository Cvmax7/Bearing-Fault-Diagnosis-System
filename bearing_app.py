import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import altair as alt  # 更换为Altair绘图库，更适合流式数据

# 页面配置
st.set_page_config(
    page_title="轴承故障检测系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://www.example.com/help",
        "Report a bug": "https://www.example.com/bug",
        "About": "# 轴承故障检测系统 \n 基于机器学习的实时轴承状态监测",
    },
)

# 使用专业的深色主题
st.markdown(
    """
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.2rem;
        color: #4c9aff;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.2);
    }
    .status-box {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-normal {
        background-color: rgba(76, 175, 80, 0.15);
        border-left: 5px solid #4CAF50;
    }
    .status-fault {
        background-color: rgba(244, 67, 54, 0.15);
        border-left: 5px solid #F44336;
    }
    .status-title {
        font-size: 1.5rem;
        margin-bottom: 0.8rem;
        color: #fafafa;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        flex: 1;
        min-width: 120px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
    }
    .control-panel {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .control-title {
        color: #4c9aff;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .info-message {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 5px solid #2196F3;
        padding: 0.8rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    /* 自定义滑块颜色 */
    .stSlider > div > div > div {
        background-color: #4c9aff !important;
    }
    /* 自定义按钮样式 */
    .stButton button {
        background-color: #4c9aff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #3a7bd5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)

# 初始化会话状态
if "vibration_data" not in st.session_state:
    st.session_state.vibration_data = {"time": [], "x": [], "y": [], "z": []}
if "df" not in st.session_state:  # 为Altair创建DataFrame
    st.session_state.df = pd.DataFrame(columns=["time", "value", "axis"])
if "status" not in st.session_state:
    st.session_state.status = "正常"
if "rpm" not in st.session_state:
    st.session_state.rpm = 1500
if "temperature" not in st.session_state:
    st.session_state.temperature = 25.0
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.95
if "update_time" not in st.session_state:
    st.session_state.update_time = datetime.now()
if "last_error_time" not in st.session_state:
    st.session_state.last_error_time = None

# API配置
API_BASE_URL = "http://localhost:8000"


# 获取最新数据
def get_latest_data():
    try:
        response = requests.get(f"{API_BASE_URL}/api/data/latest", timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data and "acceleration_x" in data:
                # 更新会话状态
                current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                st.session_state.update_time = datetime.now()

                # 添加新数据点
                st.session_state.vibration_data["time"].append(current_time)
                st.session_state.vibration_data["x"].append(data["acceleration_x"])
                st.session_state.vibration_data["y"].append(data["acceleration_y"])
                st.session_state.vibration_data["z"].append(data["acceleration_z"])

                # 限制数据点数量，保持最近的100个点
                max_points = 100
                if len(st.session_state.vibration_data["time"]) > max_points:
                    st.session_state.vibration_data["time"] = (
                        st.session_state.vibration_data["time"][-max_points:]
                    )
                    st.session_state.vibration_data["x"] = (
                        st.session_state.vibration_data["x"][-max_points:]
                    )
                    st.session_state.vibration_data["y"] = (
                        st.session_state.vibration_data["y"][-max_points:]
                    )
                    st.session_state.vibration_data["z"] = (
                        st.session_state.vibration_data["z"][-max_points:]
                    )

                # 为Altair创建长格式DataFrame（更适合动态可视化）
                new_data = []
                for i in range(
                    len(st.session_state.vibration_data["time"][-30:])
                ):  # 仅使用最近30个点
                    idx = -(30 - i)
                    t = st.session_state.vibration_data["time"][idx]
                    new_data.append(
                        {
                            "time": t,
                            "value": st.session_state.vibration_data["x"][idx],
                            "axis": "X轴",
                        }
                    )
                    new_data.append(
                        {
                            "time": t,
                            "value": st.session_state.vibration_data["y"][idx],
                            "axis": "Y轴",
                        }
                    )
                    new_data.append(
                        {
                            "time": t,
                            "value": st.session_state.vibration_data["z"][idx],
                            "axis": "Z轴",
                        }
                    )

                st.session_state.df = pd.DataFrame(new_data)

                # 更新状态和指标
                st.session_state.status = data.get("prediction", "正常")
                st.session_state.rpm = data.get("rpm", 0)
                st.session_state.temperature = data.get("temperature", 0)
                st.session_state.confidence = data.get("confidence", 0)

                return True
            return False
        else:
            st.error(f"API错误: {response.status_code}")
            return False
    except Exception as e:
        current_time = time.time()
        # 避免频繁显示相同错误
        if (
            st.session_state.last_error_time is None
            or current_time - st.session_state.last_error_time > 10
        ):
            st.error(f"获取数据出错: {e}")
            st.session_state.last_error_time = current_time
        return False


# 设置转速
def set_rpm(rpm_value):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/command/set_rpm",
            json={"value": rpm_value},
            headers={"Content-Type": "application/json"},
            timeout=2,
        )
        if response.status_code == 200:
            st.success(f"✅ 转速已设置为 {rpm_value} RPM")
            return True
        else:
            st.error(f"❌ 设置转速失败: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ 设置转速出错: {e}")
        return False


# 设置故障类型
def set_fault_type(fault_type):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/command/set_fault",
            json={"value": fault_type},
            headers={"Content-Type": "application/json"},
            timeout=2,
        )
        if response.status_code == 200:
            st.success(f"✅ 故障类型已设置为 {fault_type}")
            return True
        else:
            st.error(f"❌ 设置故障类型失败: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ 设置故障类型出错: {e}")
        return False


# 创建振动数据图表 (使用Altair)
def create_vibration_chart_altair():
    if len(st.session_state.df) == 0:
        # 如果没有数据，返回空白图表
        return (
            alt.Chart(pd.DataFrame({"time": [0], "value": [0], "axis": ["X轴"]}))
            .mark_line()
            .encode(x="time", y="value")
            .properties(title="等待数据加载...")
        )

    # 使用Altair创建平滑线图
    chart = (
        alt.Chart(st.session_state.df)
        .mark_line(
            interpolate="basis",  # 使用基样条插值实现平滑曲线
            strokeWidth=2.5,
            opacity=0.8,
        )
        .encode(
            x=alt.X("time:N", axis=alt.Axis(title="时间", labelAngle=-45)),
            y=alt.Y("value:Q", axis=alt.Axis(title="加速度")),
            color=alt.Color(
                "axis:N",
                scale=alt.Scale(
                    domain=["X轴", "Y轴", "Z轴"],
                    range=["#FF4B4B", "#4CAF50", "#3B82F6"],
                ),
                legend=alt.Legend(title="轴向"),
            ),
            tooltip=["time", "value", "axis"],
        )
        .properties(title="实时振动数据", height=400)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridOpacity=0.2)
        .configure_title(fontSize=18, color="#4c9aff", anchor="start")
    )

    return chart


# 使用Plotly创建备选图表（如果需要）
def create_vibration_chart_plotly():
    # 创建一个美观的Plotly图表
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # 添加三轴加速度数据
    fig.add_trace(
        go.Scatter(
            x=st.session_state.vibration_data["time"][-40:],  # 最近40个点
            y=st.session_state.vibration_data["x"][-40:],
            mode="lines",
            name="X轴",
            line=dict(
                color="#FF4B4B",
                width=2.5,
                shape="spline",  # 使用样条曲线实现平滑
                smoothing=1.3,  # 增加平滑程度
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=st.session_state.vibration_data["time"][-40:],
            y=st.session_state.vibration_data["y"][-40:],
            mode="lines",
            name="Y轴",
            line=dict(
                color="#4CAF50", width=2.5, shape="spline", smoothing=1.3  # 平滑曲线
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=st.session_state.vibration_data["time"][-40:],
            y=st.session_state.vibration_data["z"][-40:],
            mode="lines",
            name="Z轴",
            line=dict(
                color="#3B82F6", width=2.5, shape="spline", smoothing=1.3  # 平滑曲线
            ),
        )
    )

    # 更新布局
    fig.update_layout(
        title={
            "text": "实时振动数据 (平滑曲线)",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 22, "color": "#4c9aff"},
        },
        xaxis_title="时间",
        yaxis_title="加速度",
        height=450,
        margin=dict(l=20, r=20, t=70, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font={"size": 12},
        ),
        plot_bgcolor="rgba(17, 25, 40, 0.3)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": "#fafafa"},
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)", tickfont={"size": 10}
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.1)",
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.2)",
            tickfont={"size": 10},
        ),
        hovermode="x unified",
    )

    return fig


# 页面主体
def main():
    # 标题
    st.markdown(
        "<h1 class='main-header'>基于CNN-LSTM的轴承故障检测系统</h1>",
        unsafe_allow_html=True,
    )

    # 布局
    col1, col2 = st.columns([2, 1])

    with col1:
        # 状态显示 - 改进的HTML格式
        status_class = (
            "status-normal" if st.session_state.status == "正常" else "status-fault"
        )
        st.markdown(
            f"""
            <div class="status-box {status_class}">
                <h2 class="status-title">当前状态: <span style="color: {'#4CAF50' if st.session_state.status == '正常' else '#F44336'};">{st.session_state.status}</span></h2>
                <div class="metric-container">
                    <div class="metric-box">
                        <div class="metric-value">{st.session_state.rpm:.1f}</div>
                        <div class="metric-label">转速 (RPM)</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{st.session_state.temperature:.1f}°C</div>
                        <div class="metric-label">温度</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{st.session_state.confidence * 100:.2f}%</div>
                        <div class="metric-label">可信度</div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # 振动图表 - 使用Altair或Plotly（取其一）
        st.plotly_chart(create_vibration_chart_plotly(), use_container_width=True)

    with col2:
        # 控制面板 - 改进的样式
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown('<h3 class="control-title">控制面板</h3>', unsafe_allow_html=True)

        # 转速控制
        st.write("设置转速")

        # 移除嵌套列，直接使用滑块
        rpm_value = st.slider(
            "转速调节",
            min_value=0,
            max_value=3000,
            value=int(st.session_state.rpm),
            step=50,
            key="rpm_slider",
        )

        # 显示当前值
        st.write(
            f"<div style='text-align: center; font-weight: bold; font-size: 1.2rem;'>{rpm_value} RPM</div>",
            unsafe_allow_html=True,
        )

        if st.button("应用转速设置", key="rpm_button"):
            set_rpm(rpm_value)

        st.markdown("<hr style='margin: 20px 0; opacity: 0.2'>", unsafe_allow_html=True)

        # 故障类型设置
        st.write("设置故障类型")
        fault_options = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
        fault_type = st.selectbox(
            "选择故障类型",
            fault_options,
            index=(
                fault_options.index(st.session_state.status)
                if st.session_state.status in fault_options
                else 0
            ),
            key="fault_selector",
        )
        if st.button("应用故障类型", key="fault_button"):
            set_fault_type(fault_type)

        st.markdown("</div>", unsafe_allow_html=True)

        # 系统信息部分
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown('<h3 class="control-title">系统信息</h3>', unsafe_allow_html=True)

        # 显示上次更新时间
        st.markdown(
            f"""
            <div class="info-message">
                上次数据更新时间: {st.session_state.update_time.strftime('%H:%M:%S.%f')[:-3]}
            </div>
        """,
            unsafe_allow_html=True,
        )

        # 尝试获取统计数据
        try:
            stats_response = requests.get(f"{API_BASE_URL}/api/data/stats", timeout=2)
            if stats_response.status_code == 200:
                stats = stats_response.json()

                # 不要在col2中再嵌套列
                st.metric("总记录数", stats.get("total_records", 0))

                # 故障分布
                if "fault_distribution" in stats:
                    st.write("故障分布")
                    fault_df = pd.DataFrame(
                        {
                            "故障类型": list(stats["fault_distribution"].keys()),
                            "数量": list(stats["fault_distribution"].values()),
                        }
                    )
                    st.dataframe(fault_df, hide_index=True)
            else:
                st.info("统计数据暂时不可用")
        except:
            st.info("统计API连接失败")

        st.markdown("</div>", unsafe_allow_html=True)


# 运行主函数
if __name__ == "__main__":
    # 获取初始数据
    get_latest_data()

    # 显示主界面
    main()

    # 添加手动刷新按钮和自动刷新选项 - 这里可以使用顶层列
    col_refresh, col_auto = st.columns([1, 2])

    with col_refresh:
        if st.button("🔄 刷新数据", key="refresh_button"):
            get_latest_data()

    with col_auto:
        auto_refresh = st.checkbox(
            "启用自动刷新 (每2秒)", value=True, key="auto_refresh"
        )

    # 如果启用自动刷新
    if auto_refresh:
        refresh_container = st.empty()
        with refresh_container.container():
            next_refresh_time = st.session_state.update_time + timedelta(seconds=2)
            st.markdown(
                f"""
                <div class="info-message">
                    自动刷新已启用 - 下次刷新: {next_refresh_time.strftime('%H:%M:%S')}
                </div>
            """,
                unsafe_allow_html=True,
            )

        # 获取最新数据
        get_latest_data()

        # 使用JavaScript自动刷新页面
        st.markdown(
            """
            <script>
                setTimeout(function() {
                    window.location.reload();
                }, 2000);  // 2秒后刷新
            </script>
            """,
            unsafe_allow_html=True,
        )
