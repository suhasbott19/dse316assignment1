import torch
from torch import nn
from PIL import Image
from datasets import Dataset
from datasets import load_dataset
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader


def to_tensor(x):
    return {
        "image": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(x["image"]),
        "label": torch.tensor(x["label"], dtype=torch.long)
    }

# load only the cropped digits subset from svhn
svhn = load_dataset("svhn", "cropped_digits", split="train").train_test_split(test_size=0.5, seed=42)

X = svhn['train'].train_test_split(test_size=0.25, seed=42)
X_train = X['train']
X_test = X['test']

# apply the transformation to the dataset
X_train = X_train.map(to_tensor)
X_test = X_test.map(to_tensor)

# importing alexnet
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# train the entire classifier with one added classifier (4096 to 1000)
# ! the children part doesnt work!, removes the classifier


# importing vgg11
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
# # ! the children part doesnt work!, removes the classifier
# # train the entire classifier with one added classifier (4096 to 1000)

# # importing resnet18
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# # has one fc from 512 features to 1000 features, might have to restructure...

# # importing resnet50
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# # has one fc from 2048 features to 1000 features, might have to restructure...

# importing resnet101
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# has one fc from 2048 features to 1000 features, might have to restructure...

# add a classifier to the model
model.fc = nn.Linear(2048, 10)

# set model to freeze backbone
for param in model.parameters():
    param.requires_grad = False

# freeze the classifier
for param in model.fc.parameters():
    param.requires_grad = True

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

# run training loop for model
for epoch in range(10):
    for data in X_train:
        image = torch.Tensor(data['image'])
        label = data['label']
        output = model(image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)

# evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in X_test:
        image = torch.Tensor(data['image'])
        label = data['label']
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
