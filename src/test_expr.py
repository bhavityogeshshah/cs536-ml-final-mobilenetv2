import argparse
import torch
import torchvision
import sklearn.metrics as skm
import matplotlib as plt
import torchvision.transforms as transforms
import os
from exper_model import MobileNetV2

def test_expr(dataset='cifar10',epoch='40',batch_size = 16):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  #test accuracy here
  y_true = []
  y_pred = []
  correct = 0
  total = 0
  transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
  if dataset=='cifar10':
    save_dir = 'checkpoints/exp/cifar10/'
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
  elif dataset=='cifar100':
    save_dir = 'checkpoints/exp/cifar100/'
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
    
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                      shuffle=True, num_workers=2)  
  
  checkpoint_name = f'model_{epoch}.pt'
  checkpoint_path = os.path.join(save_dir, checkpoint_name)
  model = torch.load(checkpoint_path)
  model.to(device)
  model.eval()
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          y_true = y_true + labels.detach().numpy().tolist()
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          y_pred = y_pred + predicted.to('cpu').detach().numpy().tolist()
          correct += (predicted == labels).sum().item()

  cm = skm.confusion_matrix(y_true, y_pred)
  skm.ConfusionMatrixDisplay(cm).plot()
  print(f'Accuracy of the network on the test images: {100 * correct // total} %')


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", "--epochs", help="Number of epochs")
    argParser.add_argument("-d", "--data_set", help="Data set (cifar10/cifar100)")

    args = argParser.parse_args()
    ep = int(args.epochs)
    test_expr(args.data_set,ep)