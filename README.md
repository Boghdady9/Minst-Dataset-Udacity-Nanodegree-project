# Minst-Dataset-Udacity-Nanodegree-project
# MNIST Dataset with PyTorch

This is a tutorial on how to use the MNIST dataset with PyTorch. The MNIST dataset is a collection of 70,000 handwritten digits, split into 60,000 training examples and 10,000 test examples. Each example is a grayscale image of size 28x28, and the goal is to classify each image into one of 10 possible classes (0-9).

## Requirements

- Python 3.x
- PyTorch

You can install PyTorch using pip:

```
pip install torch
```

## Downloading the Data

The MNIST dataset is available for download from the official PyTorch website. You can download the dataset using the following code:

```python
import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
```

In this code, we first define a transform to normalize the data using `transforms.Normalize()`. We then download and load the training data using `datasets.MNIST()` with `train=True` and the transform we defined. We create a `DataLoader` for the training data using `torch.utils.data.DataLoader()` with a batch size of 64 and shuffle set to True.

We then download and load the test data using `datasets.MNIST()` with `train=False` and the same transform as the training data. We create a `DataLoader` for the test data using `torch.utils.data.DataLoader()` with the same batch size and shuffle settings as the training data.

## Using the Data

Once you have downloaded and loaded the MNIST dataset, you can use it to train and test machine learning models. Here's an example of how to train a simple feedforward neural network on the MNIST dataset using PyTorch:

```python
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the neural network and move it to the GPU if available
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Train the neural network
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
```

In this code, we first define a simple feedforward neural network called `Net` with three fully connected layers. We define the `forward()` method to specify how the data flows through the network.

We then create an instance of the `Net` class and move it to the GPU if available using `net.to(device)`.

We define the loss function as cross-entropy loss and the optimizer as stochastic gradient descent (SGD) with momentum. We then use a `for` loop to train the neural network for 400 epochs. In each epoch, we iterate over the training data using a `for` loop and use the `optimizer` to update the weights of the neural network based on the loss calculated by the `criterion`. We print the training loss every 64 batches.

## Conclusion

In this tutorial, we showed you how to downloadand load the MNIST dataset using PyTorch, and how to use the dataset to train a simple feedforward neural network. You can use this code as a starting point to train more complex models on the MNIST dataset, or to explore other computer vision tasks using PyTorch.

When working with the MNIST dataset, it's important to keep in mind that it is a relatively simple dataset, and that more complex datasets may require more sophisticated models and techniques. Nonetheless, MNIST is a great dataset for getting started with machine learning and computer vision, and is widely used in the research community for benchmarking new models and techniques.

If you're new to PyTorch or machine learning in general, we recommend exploring the PyTorch documentation and tutorials to learn more about the framework and how to use it for machine learning tasks. There are also many resources online, including blogs, videos, and forums, that can help you learn more about machine learning and computer vision.
