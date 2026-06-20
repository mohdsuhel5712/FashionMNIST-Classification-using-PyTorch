import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from model import fashionNet
# import all the required libararies

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()

train_dataset = FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

model = fashionNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10
# traing loop 
for epoch in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

torch.save(
    model.state_dict(),
    "fashion_model.pth"
)

print("Model Saved Successfully")