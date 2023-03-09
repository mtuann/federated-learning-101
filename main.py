from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision
import argparse
import json
from easydict import EasyDict
import tqdm

# from models import *
from models.vgg import get_vgg_model
from models.resnet import ResNet18
from utils import Net
import sys

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# write a class that accumulate the loss, and accuracy for each epoch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val,  n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        train_loss.update(loss.item(), data.size(0))
        
        _, pred = torch.max(output, 1)
        correct = torch.eq(pred, target).sum().float().item()
        
        train_acc.update(correct / data.size(0), data.size(0))

        loss.backward()
        optimizer.step()
        pbar.set_description("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch, train_loss.avg, train_acc.avg))
        # if batch_idx == 5:
        #     import IPython
        #     IPython.embed()
        #     exit(0)
            
        # if batch_idx % args.log_interval == 0:
        #     logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, criterion):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    if args.dataset == "EMNIST":
        classes = [str(i) for i in range(10)]
    elif args.dataset == "MNIST":
        classes = [str(i) for i in range(10)]
    elif args.dataset == "Cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        target_class = 2
    elif args.dataset == "TinyImageNet": # 200 class-
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        target_class = 2

    model.eval()
    test_loss = 0
    correct = 0
    test_losses = AverageMeter()
    test_accs = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss = criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            test_losses.update(test_loss, data.size(0))
            test_accs.update(correct / data.size(0), data.size(0))
            pbar.set_description("Test Loss: {:.4f}, Acc: {:.4f}".format(test_losses.avg, test_accs.avg))
            
            
            # for image_index in range(args.test_batch_size):
            for image_index in range(len(target)):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
                class_total[label] += 1

    test_loss /= len(test_loader.dataset)

    for i in range(10):
        logger.info('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_args(json_file):
    # get value from json file
    with open(json_file, 'r') as f:
        args = json.load(f)
    return EasyDict(args)

def get_dataset(args, kwargs):
    if args.dataset == "EMNIST":
        train_dataset = datasets.EMNIST('./data', split="digits", train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        test_dataset = datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    elif args.dataset == "MNIST":
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    
    elif args.dataset == "Cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

def get_model(args, device):
    if args.model == "LeNet":
        model = Net(num_classes=10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    elif args.model in ("vgg9", "vgg11", "vgg13", "vgg16"):
        model = get_vgg_model(args.model).to(device)
        #model = VGG(args.model.upper()).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[e for e in [151, 251]], gamma=0.1)
        
    elif args.model in ("ResNet18"):
        model = ResNet18().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[e for e in [151, 251]], gamma=0.1)
        
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion

def main(json_config="./configs/fedml_config_yaml.json"):
    # Training settings
    args_dict = get_args(json_config)
    
    args = args_dict.fed_training
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print("kwargs = ", kwargs, "device = ", device, "use_cuda = ", use_cuda, "args.device = ", args.device)
    # return
    # prepare dataset
    
    train_loader, test_loader = get_dataset(args, kwargs)
    model, optimizer, scheduler, criterion = get_model(args, device)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(args, model, device, test_loader, criterion)

        for param_group in optimizer.param_groups:
            logger.info(param_group['lr'])
        scheduler.step()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "./checkpoint/{}_{}_{}epoch.pt".format(args.dataset, args.model.upper(), args.epochs))


if __name__ == '__main__':
    # get name of file json from command line
    parser =  argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--config', type=str, default="./configs/cifar10_030823", help='config file')
    args = parser.parse_args()
    main(json_config=args.config)
    
