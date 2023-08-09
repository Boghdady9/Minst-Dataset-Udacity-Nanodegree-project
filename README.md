# Minst-Dataset-Udacity-Nanodegree-project
# MNIST Dataset with PyTorch

In this project, I build a neural network that predicts the label of the image. The MNIST dataset is a collection of 70,000 handwritten digits, split into 60,000 training examples and 10,000 test examples. Each example is a grayscale image of size 28x28, and the goal is to classify each image into one of 10 possible classes (0-9).

## Requirements

- Python 3
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
train = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

# Download and load the test data
test = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test, batch_size=200, shuffle=True)
```

In this code, we first define a transform to normalize the data using `transforms.Normalize()`. We then download and load the training data using `datasets.MNIST()` with `train=True` and the transform we defined. We create a `DataLoader` for the training data using `torch.utils.data.DataLoader()` with a batch size of 64 and shuffle set to True.

We then download and load the test data using `datasets.MNIST()` with `train=False` and the same transform as the training data. We create a `DataLoader` for the test data using `torch.utils.data.DataLoader()` with the same batch size and shuffle settings as the training data.

## Using the Data

Once you have downloaded and loaded the MNIST dataset, you can use it to train and test machine-learning models. Here's an example of how to train a simple feedforward neural network on the MNIST dataset using PyTorch:

```python

class Net(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.activation = F.relu
        self.dropout_prob = dropout_prob

        self.fc1 = nn.Linear(28*28, 512)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)

        x = self.activation(self.fc2(x))
        x = self.dropout2(x)

        x = F.softmax(self.fc3(x))
        return x
net=Net()    
net.to(device)

def train_model(net, train_loader, test_loader, num_epoch=300, lr=0.001, momentum=0.5):
    optimizer = optim.SGD(net.parameters(), lr=lr,momentum=momentum)
    criterion = nn.NLLLoss()

    train_loss_history = []
    test_loss_history = []

    for epoch in range(num_epoch):
        net.train()
        train_loss = 0
        train_correct = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(output.data, 1)
            train_correct += (preds == labels).sum().item()
            train_loss += loss.item()

        train_loss_history.append(train_loss / len(train_loader))
        train_acc = train_correct / len(train_loader.dataset)

        net.eval()
        test_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs.data, 1)
                test_correct += (preds == labels).sum().item()
                test_loss += loss.item()

        test_loss_history.append(test_loss / len(test_loader))
        test_acc = test_correct / len(test_loader.dataset)

        print(f'Epoch {epoch + 1} training accuracy: {train_acc:.2f}% training loss: {train_loss / len(train_loader):.5f}')
        print(f'Epoch {epoch + 1} test accuracy: {test_acc:.2f}% test loss: {test_loss / len(test_loader):.5f}')

    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(test_loss_history, label="test Loss")
    plt.legend()
    plt.show()
```

In this code, we first define a simple feedforward neural network called `Net` with three fully connected layers. We define the `forward()` method to specify how the data flows through the network.

We then create an instance of the `Net` class and move it to the GPU if available using `net.to(device)`.

We define the loss function as cross-entropy loss and the optimizer as stochastic gradient descent (SGD) with momentum. We then use a `for` loop to train the neural network for 400 epochs. In each epoch, we iterate over the training data using a `for` loop and use the `optimizer` to update the weights of the neural network based on the loss calculated by the `criterion`. We print the training loss every batch.




