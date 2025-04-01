import torch

# 加载checkpoint
checkpoint = torch.load("models/simulated_cnn_lstm_best_model.pth")

# 提取model_state_dict
model_state_dict = checkpoint["model_state_dict"]

# 保存为单独的state_dict
torch.save(model_state_dict, "bearing_fault_model.pth")

print("模型已成功提取并保存为bearing_fault_model.pth")
