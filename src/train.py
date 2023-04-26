from model import MobileNetV2
import torch
import torchvision
from torch import nn, Tensor
import torchvision.transforms as transforms
import math
import torch.optim as optim
import time
import os
from utils import save_vars
import argparse

def train(batch_size=16, lr=0.001, num_epochs=2,dataset='cifar10'):
    """
        Trains the model for a selected dataset
        parameters
        batch_size: [default 16]
        lr: learning rate [default 0.001]
        num_epochs: epochs to train the model [default 2]
        dataset: string with values 'cifar10' and 'cifar100'

        Model trains on selected dataset and creates checkpoints for every
        10 epochs.
        Dumps the loss and accuracy vs epoch data in a pickle file
    """
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MobileNetV2(num_classes=10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        save_dir = 'checkpoints/original/cifar10/'
        metrics_dir = 'metrics/original/cifar10/'

    elif dataset=='cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
        save_dir = 'checkpoints/original/cifar100/'
        metrics_dir = 'metrics/original/cifar100/'

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_train_time = []
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
            # checkpoint_name = f'epoch_{epoch}.pth'
            # checkpoint_path = os.path.join(save_dir, checkpoint_name)
            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'epoch': epoch,
            # }, checkpoint_path)
            save_name = f'model_{epoch}.pt'
            save_path = os.path.join(save_dir,save_name)
            torch.save(model, save_path)
        train_loss = running_loss/len(trainloader.sampler)
        train_acc = running_corrects/len(trainloader.sampler)
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        epoch_time = epoch_end_time - epoch_start_time
        epoch_train_time.append(epoch_time)

        # Dumping metrics
        loss_name = f'loss_{epoch}.pkl'
        save_path = os.path.join(metrics_dir, loss_name)
        save_vars(epoch_train_loss, save_path)

        acc_name = f'acc_{epoch}.pkl'
        save_path = os.path.join(metrics_dir, acc_name)
        save_vars(epoch_train_acc, save_path)

        time_name = f'time_{epoch}.pkl'
        save_path = os.path.join(metrics_dir, time_name)
        save_vars(epoch_train_time, save_path)
        
        print('Training: Loss: {} Accuracy: {} Time: {}s'.format(train_loss, train_acc, epoch_time))

    train_end_time = time.time()
    total_train_time = train_end_time-train_start_time
    print('Total Training time: {}'.format(total_train_time))


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-b", "--batch_size", help="Batch size")
    argParser.add_argument("-lr", "--learning_rate", help="Learning rate")
    argParser.add_argument("-e", "--epochs", help="Number of epochs")
    argParser.add_argument("-d", "--data_set", help="Data set (cifar10/cifar10)")

    args = argParser.parse_args()
    b = int(args.batch_size)
    lr = float(args.learning_rate)
    ep = int(args.epochs)
    train(b,lr,ep,args.data_set)
