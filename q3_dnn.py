from torchvision import transforms
from torch.nn import Linear
from torch import nn
import torch
from PIL import Image
from torch.nn import functional as F

# We consider these hyperparameters: 

# a) Network Hyperparameters
#   n_hidden_layers: Number of hidden layers in the network
#   activation: Activation function for each layer


# b) Training Hyperparameters
#   learning_rate: Learning rate for the optimizer
#   batch_size: Batch size for training
#   epochs: Number of epochs for training
#   weight_decay: Weight decay for the optimizer
#   regularization methods: L1, L2
#   optimizer: Optimizer to use for training

# Assumptions:

# the input image is a 28x28 grayscale image, hence normalization is done in the only channel
# The input layer has 784 neurons, corresponding to the 28x28 input image.
# The output layer has 3 neuron, assuming there are 3 classes (random assumption)
# The activation function for the output layer is softmax
# The activation function is same for the rest of the layers
# The Data Augmentation Strategy is not considered.
# Dense Neural Networks are fully connected networks and there are no skip connections or dropout layers


hyperparameters_dict = {
    'n_hidden_layers': [1, 2, 3],
    'activation': ['relu', 'sigmoid', 'tanh'],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64, 128],
    'epochs': [10, 25, 50, 100],
    'weight_decay': [0.001, 0.01, 0.1],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'augmentation': ['flip', 'rotate', 'crop']
}

def create_model(n_hidden_layers):
    if n_hidden_layers == 1:
        return Linear(784, 512)
    elif n_hidden_layers == 2:
        return [Linear(784, 512), Linear(512, 256)]
    elif n_hidden_layers == 3:
        return [Linear(784, 512), Linear(512, 256), Linear(256, 64)]

def create_transforms(augmentation):
    if augmentation == 'flip':
        return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    elif augmentation == 'rotate':
        return transforms.Compose([transforms.RandomRotation(30)])
    elif augmentation == 'crop':
        return transforms.Compose([transforms.RandomResizedCrop(28)])

def create_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()

def create_optimizer(model, optimizer, learning_rate, weight_decay):
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_model(n_hidden_layers, activation):
    model = nn.Sequential()
    model.add_module('input', nn.Linear(784, 512))
    model.add_module('activation', create_activation(activation))
    for i in range(n_hidden_layers):
        model.add_module(f'hidden_{i}', nn.Linear(512//(2**i), 512//(2**(i+1))))
        model.add_module(f'activation_{i}', create_activation(activation))
    model.add_module('output', nn.Linear(512//(2**n_hidden_layers), 3))
    model.add_module('softmax', nn.Softmax(dim=1))
    return model

def train_model(model, optimizer, criterion, train_loader, epochs):
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def preprocess_by_transforms(image, man_transforms, augmentation_transforms):
    image = augmentation_transforms(image)
    image = man_transforms(image)
    return image
    
def inference(model, image):
    return model(image)

def main():
    model = create_model(1, 'relu')
    my_transforms = transforms.Compose([transforms.Resize((28,28)),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5]),
                                        transforms.Lambda(lambda x: x.view(1, -1))
                                        ])
    augmentation_transforms = create_transforms('flip')
    optimizer = create_optimizer(model, 'adam', 0.001, 0.001)
    criterion = nn.CrossEntropyLoss()
    random_image = Image.open(r'C:\Users\suhas\Codes\test\lenna.png')
    input = preprocess_by_transforms(random_image, my_transforms, augmentation_transforms)
    output = inference(model, input)
    print('Probabilities: ', F.softmax(output, dim=1))

main()