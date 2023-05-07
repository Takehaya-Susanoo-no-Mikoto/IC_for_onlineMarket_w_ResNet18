import os
from torch.utils.data import DataLoader
from torch import nn, optim
import torchvision.models as models
import torch
from torchvision import transforms, datasets

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "ResNet"
    net_path = f"E:\\pythonProject4\\{model_name}.pt"
    test_root = 'E:\\images\\test'
    train_root = 'E:\\images\\train_1'
    num_epochs = 50
    lr = 0.001
    batch_size = 64

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.RandomRotation(30),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_root, transform=data_transform)
    test_dataset = datasets.ImageFolder(test_root, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=3)


    class ModelsMgmt:
        def save(self, model, path, model_name):
            torch.save(model.state_dict(), os.path.join(path, f"{model_name}.pt"))

        def load(self, path, model, model_name):
            model.load_state_dict(torch.load(os.path.join(path, f"{model_name}.pt")))
            return model


    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train()

        running_loss = 0.0
        correct = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            model.to(device)
            model.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / len(dataloader.dataset)
        return epoch_loss, epoch_acc


    def evaluate_loss_acc(model, criterion, test_loader, device):
        model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                model.to(device)

                output = model(data)

                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        return test_loss, accuracy


    # Проверяем, есть ли сохраненные веса для модели
    if os.path.exists(net_path):
        # Если веса есть, загружаем их в модель
        net = models.resnet18()
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 874)
        net.load_state_dict(torch.load(net_path))
        for param in net.parameters():
            param.requires_grad = False
        for param in net.fc.parameters():
            param.requires_grad = True
    else:
        # Если весов нет, создаем новую модель
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 874)
        for param in net.parameters():
            param.requires_grad = False
        for param in net.fc.parameters():
            param.requires_grad = True

    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)

    net = net.to(device)
    print(net)

    verbose = True
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_loss_acc(net, criterion, test_loader, device)

        ModelsMgmt().save(net, "E:\\pythonProject4", model_name)

        if verbose:
            print(('Epoch [%d/%d], Loss (train/test) : %.4f/%.4f,' + 'Acc (train/test): %.4f/%.4f')
                  % (epoch + 1, num_epochs, train_loss, val_loss, train_acc, val_acc))
