import torch
from matplotlib import pyplot as plt

from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

from model import fashionNet
import random
# making classes where i have to pick 
classes = [
      "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]
# device finding
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()
test_dataset = FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

model = fashionNet().to(device)
model.load_state_dict(
      torch.load('fashion_model.pth')
)

model.eval()

idex = random.randint(0,len(test_dataset) -1)
images,label = test_dataset[idex]

with torch.no_grad():
      output = model(images.unsqueeze(0).to(device))
      prediction = output.argmax(1)
plt.imshow(images.squeeze(),cmap="gray")
plt.title(f'predicted:{classes[prediction]}')
plt.show()
print('actual',classes[label])