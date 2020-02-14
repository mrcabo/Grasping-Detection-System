import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from cornell_dataset import scale_values
from util import calculate_similarity


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.fc1 = nn.Linear(256, 1024)
    self.fc2 = nn.Linear(1024, 5)

  def forward(self, x):
    x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (3, 3))
    x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), (3, 3))
    x = x.view(x.shape[0], -1)
    x = nn.functional.relu(self.fc1(x))
    x = torch.sigmoid(self.fc2(x))
    return x


class CNN:
    def __init__(self, dest_path, train_loader, valid_loader, test_loader):
        self.dest_path = dest_path
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        # self.model = ResNet18(pre_trained=pre_trained)
        self.model = Net()
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        # See if we use CPU or GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()

    def train(self):
        total_loss = 0
        self.model.train()
        if self.cuda_available:
            self.model.cuda()
        for i, data in enumerate(self.train_loader):
            X, y = data['image'].to(self.device), data['rectangle'].to(self.device)
            # training step for single batch
            y = scale_values(y, 'down')
            self.model.zero_grad()
            outputs = self.model(X)
            loss = self.loss_function(outputs, y)
            loss.backward()
            self.optimizer.step()

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss

    def evaluate(self, data_loader):
        val_losses = []
        accuracies = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                X, y = data['image'].to(self.device), data['rectangle'].to(self.device)
                outputs = self.model(X)  # this get's the prediction from the network
                val_losses.append(self.loss_function(outputs, scale_values(y, 'down')))
                accuracy = calculate_similarity(scale_values(outputs, 'up'), y, self.device)
                accuracies.append(accuracy)

        return val_losses, accuracies

    def validate(self):
        return self.evaluate(self.valid_loader)

    def test(self):
        return self.evaluate(self.test_loader)

    def get_prediction(self, data_loader):
        self.model.eval()
        predictions = []
        images = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                X, y = data['image'].to(self.device), data['rectangle'].to(self.device)
                outputs = self.model(X)  # this get's the prediction from the network
                # print(f"predicted: {outputs}, label: {y}")
                outputs = scale_values(outputs, 'up')
                predictions.append(outputs)
                images.append(X)
        return images, predictions

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path, device):
        self.model.load_state_dict(torch.load(path, map_location=device), strict=False)

    @staticmethod
    def save_experiment(path, metrics):
        filename = path.parts[-1] + '-accuracy'
        filename = (path.parent / filename).as_posix()
        train_loss, valid_loss, val_accuracy, test_accuracy = list(zip(*metrics))
        plt.plot(val_accuracy)
        plt.plot(test_accuracy)
        plt.title("ResNet50 Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Validation set", "Test set"], loc="upper left")
        plt.savefig(filename)
        plt.close()

        filename = path.parts[-1] + '-loss'
        filename = (path.parent / filename).as_posix()
        plt.plot(train_loss)
        plt.plot(valid_loss)
        plt.title("ResNet50 Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Training set", "Validation set"], loc="upper left")
        plt.savefig(filename)
        plt.close()

    def free(self):
        del self.model
        del self.train_loader
        del self.valid_loader
        del self.test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
