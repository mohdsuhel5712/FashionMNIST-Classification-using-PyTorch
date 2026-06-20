# evaluation means (checkimg = testing the model)
import torch
from matplotlib import pyplot as plt 
import seaborn as sns

from sklearn.metrics import confusion_matrix

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from model import fashionNet
# check the decive EITHER CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device",device)
# device selection

transform = transforms.ToTensor()
# tarin data set 
test_dataset = FashionMNIST(root="data",train=False,download=True,transform=transform)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

model = fashionNet().to(device)

model.load_state_dict(
      torch.load("fashion_model.pth")
)

model.eval()
correct= 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
      # testing loop
      for images,labels in test_loader:
             images = images.to(device)
             labels = labels.to(device)
             
             output = model(images)
             preds = output.argmax(1)
             total +=labels.size(0)
             
             correct += (
                   preds == labels
             ).sum().item()
             
             all_preds.extend(
                   preds.cpu().numpy()
             )
             
             all_labels.extend(
                   labels.numpy()
             )
accucary = (100*correct)/total

cm = confusion_matrix(
      all_labels,all_preds
)

plt.figure(figsize=(10,8))
sns.heatmap(
      cm,annot=True,
      fmt="d"
)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

             