import torch
import torchvision
from torch import nn, optim
import matplotlib.pyplot as plt

from cornell_dataset import scale_values
from util import calculate_similarity


class ResNet18(nn.Module):
    def __init__(self, pre_trained=True):
        super().__init__()
        res_net = torchvision.models.resnet18(pretrained=pre_trained)
        res_net.fc = nn.Linear(512, 256)  # 512 for resnet18
        self.add_module('resnet', res_net)

        fc1 = nn.Linear(256, 128)
        self.add_module('fc1', fc1)

        fc_reg = nn.Linear(128, 5)
        self.add_module('fc_reg', fc_reg)

    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc_reg(x))
        return x


class ResNet50(nn.Module):
    def __init__(self, pre_trained=True):
        super().__init__()
        res_net = torchvision.models.resnet50(pretrained=pre_trained)
        res_net.fc = nn.Linear(2048, 512)  # 2048 for resnet50
        self.add_module('resnet', res_net)

        fc1 = nn.Linear(512, 128)
        self.add_module('fc1', fc1)

        fc_reg = nn.Linear(128, 5)
        self.add_module('fc_reg', fc_reg)

    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc_reg(x))
        return x


class MobileNet(nn.Module):

    def __init__(self, pre_trained=True):
        super().__init__()
        mobile_net = torchvision.models.mobilenet_v2(pretrained=pre_trained)
        mobile_net.classifier[1] = nn.Linear(1280, 320)  # reshape to fit correct number of classes
        self.add_module('mobileNet', mobile_net)

        fc1 = nn.Linear(320, 5)
        self.add_module('fc1', fc1)

    def forward(self, x):
        x = self.mobileNet(x)
        x = nn.functional.relu(self.fc1(x))
        return x


class SqueezeNet(nn.Module):

    def __init__(self, pre_trained=True):
        super().__init__()
        squeeze_net = torchvision.models.squeezenet1_1(pretrained=pre_trained)
        self.add_module('squeezenet', squeeze_net)

        fc1 = nn.Linear(1000, 256)
        self.add_module('fc1', fc1)
        fc_reg = nn.Linear(256, 5)
        self.add_module('fc_reg', fc_reg)

    def forward(self, x):
        x = self.squeezenet(x)
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc_reg(x))
        return x


class OurNet:
    def __init__(self, dest_path, train_loader, valid_loader, test_loader, network_name, pre_trained=True, **kwargs):
        self.dest_path = dest_path
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        if network_name == "squeezenet1_1":
            self.model = SqueezeNet(pre_trained=pre_trained)
        elif network_name == "mobilenet_v2":
            self.model = MobileNet(pre_trained=pre_trained)
        elif network_name == "resnet18":
            self.model = ResNet18(pre_trained=pre_trained)
        elif network_name == "resnet50":
            self.model = ResNet50(pre_trained=pre_trained)
        else:
            raise NameError('{} is not a possible network'.format(network_name))
        print(f"Model used: {network_name}")

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        # See if we use CPU or GPU
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
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
