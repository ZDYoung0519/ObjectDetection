import torch
import torchvision.models as models

AlexNet = models.AlexNet()
print(AlexNet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AlexNet.to(device)

batch_size = 64
epotch = 10

# 数据集
# 预训练AlexNet


