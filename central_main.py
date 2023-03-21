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
        

def train(args, model, device, data_loader, optimizer, criterion, epoch, is_train=True):
    
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
        
    
    # model.eval()
    
    # model_train = model.train()
    # model_eval = model.eval()
    # calculate total sum different between model_train and model_eval
    # sum_diff = 0
    # for p1, p2 in zip(model_train.parameters(), model_eval.parameters()):
    #     sum_diff += torch.sum(torch.abs(p1 - p2))
    # print(sum_diff)
    # exit(0)
    
    if is_train:
        model.train()
    else:
        model.eval()
        
    # model.eval()
    
    # print(sum(p.numel() for p in model_train.parameters() if p.requires_grad))

    
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    total_correct = 0
    
    # using torch.no_grad() when is_train == True
    # https://discuss.pytorch.org/t/what-does-model-eval-do-for-pytorch-models/7146/2
    
    with torch.set_grad_enabled(is_train):
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output, target)
            # loss = F.nll_loss(output, target)
            if is_train:
                loss.backward()
                optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            correct_pred = pred.eq(target.view_as(pred))
            correct = correct_pred.sum().item()
            total_correct += correct
            
            train_loss.update(loss.item(), data.size(0))    
            train_acc.update(correct / data.size(0), data.size(0))
            # if batch_idx % 10 == 0:
            #     print(loss.item())
            pbar.set_description("{} Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format("Train" if is_train else "Test", epoch, train_loss.avg, train_acc.avg))
                
            for image_index in range(len(target)):
                label = target[image_index]
                class_correct[label] += correct_pred[image_index].item()
                class_total[label] += 1
            
    for i in range(10):
        logger.info('%s Accuracy of %5s : %2d %%' % ("Train" if is_train else "Test",
            classes[i], 100 * class_correct[i] / class_total[i]))
        
    logger.info('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format("Train" if is_train else "Test",
        train_loss.avg, total_correct, len(data_loader.dataset),
        100. * total_correct / len(data_loader.dataset)))


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
        test_dataset = datasets.MNIST('./data', train=False,
                            transform=transforms.Compose([
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
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_model(args, device):
    if args.model == "LeNet":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = Net(num_classes=10).to(device)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        # load pretrained model (./checkpoint/mnist_cnn.pt)
        # model.load_state_dict(torch.load('./checkpoint/mnist_cnn.pt'))
            
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
    # criterion = nn.NLLLoss()
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
    print(len(train_loader.dataset), len(test_loader.dataset))
    print(len(train_loader), len(test_loader))
    # exit(0)
    model, optimizer, scheduler, criterion = get_model(args, device)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch, is_train=True)
        train(args, model, device, test_loader, criterion, criterion, epoch, is_train=False)

        for param_group in optimizer.param_groups:
            logger.info(param_group['lr'])
        scheduler.step()

        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), "./checkpoint/{}_{}_{}epoch.pt".format(args.dataset, args.model.upper(), args.epochs))


if __name__ == '__main__':
    # get name of file json from command line
    parser =  argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--config', type=str, default="./configs/cifar10_030823", help='config file')
    parser.add_argument('--config', type=str, default="./configs/mnist_030823.json", help='config file')
    args = parser.parse_args()
    main(json_config=args.config)
    
