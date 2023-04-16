from model import MobileNetV2
import torch
import torchvision
from torch import nn, Tensor
import torchvision.transforms as transforms
import math
import torch.optim as optim
import time
import os

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MobileNetV2(num_classes=10)
    num_epochs = 2
    loss_fn = nn.CrossEntropyLoss()
    batch_size=16
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(),
     transforms.Resize((224,224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
    
    save_dir = 'checkpointsAdam/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    


    epoch_train_loss = []
    epoch_train_acc = []
    model.to(device)
    train_start_time = time.time()
    for epoch in range(num_epochs):
        print("+"*50+" Epoch "+str(epoch)+" "+"+"*50)
        epoch_start_time = time.time()
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                _,preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
            epoch_end_time = time.time()
            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(labels==preds).item()
        if epoch%10 == 0:
            checkpoint_name = f'epoch_{epoch}.pth'
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)
        train_loss = running_loss/len(trainloader.sampler)
        train_acc = running_corrects/len(trainloader.sampler)
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_time = epoch_end_time - epoch_start_time
        print('Training: Loss: {} Accuracy: {} Time: {}s'.format(train_loss, train_acc, epoch_time))

    train_end_time = time.time()
    total_train_time = train_end_time-train_start_time
    print('Total Training time: {}'.format(total_train_time))
