from fastapi import FastAPI, WebSocket
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 配置更宽松的CORS以允许WebSocket连接
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        print("WebSocket连接请求...")
        await websocket.accept()
        print("WebSocket连接已接受")

        # 发送一条测试消息
        await websocket.send_text("服务器连接成功")

        # 持续接收消息
        while True:
            message = await websocket.receive_text()
            print(f"收到消息: {message}")
            await websocket.send_text(f"服务器回应: {message}")
    except Exception as e:
        print(f"WebSocket错误: {e}")


@app.get("/")
async def root():
    return {"message": "服务器运行中"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
