import torch
import os
tx = 5511
ty = 101

model = torch.nn.Sequential(
    torch.nn.Conv1d(1,196, 15,4),
    torch.nn.ReLU(),
    torch.nn.LSTM(196,128),
    torch.nn.Linear(128,7)
)

print(model)
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
model.eval()

model.qconfig = torch.quantization.default_qconfig

print(model.qconfig)
torch.quantization.prepare(model, inplace=True)
print_size_of_model(model)
qmodel = torch.quantization.convert(model)
print_size_of_model(qmodel)
print(qmodel)


import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8
)
print(quantized_model)
print_size_of_model(quantized_model)