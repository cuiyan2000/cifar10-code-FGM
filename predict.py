import torch
import torchvision
from PIL import Image
from model import LeNet

CLASS = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
image_path = "4(1).png"
image = Image.open(image_path)
# 如果不用image.convert('RGB')，直接打开是1通道的
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LeNet().to(device)


model.load_state_dict(torch.load('Lenet.pth'))

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)

ret = output.argmax(1)
ret = ret.numpy()
print("预测结果为:{}".format(CLASS[ret[0]]))