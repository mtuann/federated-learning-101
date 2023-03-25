from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import json
from easydict import EasyDict
import tqdm
import numpy as np

# from models import *
from models.vgg import get_vgg_model
from models.resnet import ResNet18
from utils import Net
import sys
from copy import deepcopy

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
		

def train_test_model(args, test_model, device, data_loader, optimizer, criterion, epoch, is_train=True):
	
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	if args.dataset == "EMNIST":
		classes = [str(i) for i in range(10)]
	elif args.dataset == "MNIST":
		classes = [str(i) for i in range(10)]
	elif args.dataset == "Cifar10":
		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  
	elif args.dataset == "TinyImageNet": # 200 class
		# get the name of 200 classes of TinyImageNet (wnids.txt)
		with open("./data/tiny-imagenet-200/words.txt", "r") as f:
			name_classes = f.readlines()
		name_classes = [x.strip() for x in name_classes]
		
		map_class = {}
		for i, c in enumerate(name_classes):
			cc = c.split("\t")
			map_class[cc[0]] = cc[1]
			
		with open("./data/tiny-imagenet-200/wnids.txt", "r") as f:
			id_classes = f.readlines()
		
		id_classes = [x.strip() for x in id_classes]
		classes = [map_class[x.strip()] for x in id_classes]
		
	if is_train:
		test_model.train()
	else:
		test_model.eval()
		
	
	running_loss = AverageMeter()
	running_acc = AverageMeter()
	
	pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
	total_correct = 0
	
	# using torch.no_grad() when is_train == True
	# https://discuss.pytorch.org/t/what-does-model-eval-do-for-pytorch-models/7146/2
	
	running_str = "Train" if is_train else "Test"
	test_model.to(device)
	# with torch.set_grad_enabled(is_train):
	for batch_idx, (data, target) in pbar:
		data, target = data.to(device), target.to(device)
		if is_train:
			optimizer.zero_grad()
		# torch.manual_seed(0)
		# torch.cuda.manual_seed(0)
		output = test_model(data)
		# op2 = model2(data)
		# import IPython; IPython.embed();exit(0)
		# if batch_idx == 0:
		# 	print(output)
		loss = criterion(output, target)

		if is_train:
			loss.backward()
			optimizer.step()
		
		pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
		# if batch_idx == 0:
		# 	print(list(pred))

		correct_pred = pred.eq(target.view_as(pred))
		correct = correct_pred.sum().item()
		total_correct += correct
		
		running_loss.update(loss.item(), data.size(0))    
		running_acc.update(correct / data.size(0), data.size(0))
		
		pbar.set_description("{} Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(running_str, epoch, running_loss.avg, running_acc.avg))
		
		if not is_train:
			for image_index in range(len(target)):
				label = target[image_index]
				class_correct[label] += correct_pred[image_index].item()
				class_total[label] += 1
					
	if not is_train:
		for i in range(10):
			logger.info('%s Accuracy of %5s : %2d %%' % (running_str, classes[i], 100 * class_correct[i] / class_total[i]))
		
	logger.info('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(running_str, running_loss.avg, total_correct, len(data_loader.dataset),
		100. * total_correct / len(data_loader.dataset)))
	
	return running_loss, running_acc


def loss_dataset(model, dataset, loss_func):
	# loss_func = nn.CrossEntropyLoss()
	loss = 0
	for x, y in dataset:
		y_pred = model(x)
		loss += loss_func(y_pred, y)
	return loss

def accuracy_dataset(model, dataset):
	# loss_func = nn.CrossEntropyLoss()
	acc = 0
	for x, y in dataset:
		y_pred = model(x)
		acc += (y_pred.argmax(dim=1) == y).float().mean()
	return acc / len(dataset)



def get_optimizer(optimizer_name, client_model, lr, weight_decay):
	if optimizer_name == "sgd":
		return optim.SGD(client_model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == "adam":
		return optim.Adam(client_model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == "adagrad":
		return optim.Adagrad(client_model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == "adadelta":
		return optim.Adadelta(client_model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == "rmsprop":
		return optim.RMSprop(client_model.parameters(), lr=lr, weight_decay=weight_decay)
	else:
		raise Exception("Optimizer not supported")


def difference_models_norm_2(model_1, model_2):
	"""Return the norm 2 difference between the two model parameters"""
	
	tensor_1=list(model_1.parameters())
	tensor_2=list(model_2.parameters())
	
	norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
		for i in range(len(tensor_1))])
	
	return norm


def client_learning(model, mu, args, criterion, train_loader, test_loader, device, local_ep, logger):
	original_model = deepcopy(model)
	
	client_optimizer = get_optimizer(args.client_optimizer, model, args.lr, args.weight_decay)
	
	total_loss = 0
	total_sample = 0
	total_correct = 0
	model.train()
 
	
	# local_ep = 1
 
	with torch.set_grad_enabled(True):
		for e in range(local_ep):
			# print("Epoch: ", e)
			total_loss = 0
			total_sample = 0
			total_correct = 0
   
			pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
			for batch_idx, (data, target) in pbar:
				data, target = data.to(device), target.to(device)
				client_optimizer.zero_grad()
				
				output = model(data)
				# p = pred.argmax(dim=1,keepdim=True)
				# correct_pred = p.eq(labels.view_as(p))
				# correct_pred.sum()
				# if idx > 0 and idx % 10 == 0:
				# 	import IPython
				# 	IPython.embed()
				# if idx == 50:
				# 	exit(0)
		
				diff_norm = mu/2* difference_models_norm_2(model, original_model)
				loss = criterion(output, target) + diff_norm
				# loss = criterion(output, target)
				# .item()*len(labels)
				
				loss.backward()
				client_optimizer.step()
				pred = output.argmax(dim=1, keepdim=True)
				correct_pred = pred.eq(target.view_as(pred))
				correct = correct_pred.sum().item()
				total_correct += correct
	
				total_loss += loss.item()*len(data) + diff_norm
				total_sample += len(data)
				# print("here", batch_idx, len(data), total_loss, total_sample, total_correct)
			# print(total_loss, total_sample, total_correct)
			pbar.set_description(f"Local Epoch: {e + 1}/ {local_ep}, Loss: {total_loss / total_sample:.4f}, Acc: {total_correct/ total_sample:.4f}")
				
	return total_loss / total_sample, total_correct/ total_sample

def generate_iid_data(train_dataset, test_dataset, args):
	N = args.client_num_in_total  # number of clients in total = client_num_in_total
	all_range = list(range(len(train_dataset)))
	data_len_per_client = len(train_dataset) // N
	# print("data_len_per_client: ", data_len_per_client)
	train_indices = [all_range[i * data_len_per_client: (i + 1) * data_len_per_client] for i in range(N)]
	train_loaders = [torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices)) for indices in train_indices]
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
	
	return train_loaders, train_indices, test_loader


def train_test_federated_fedprox(args, model, device, training_sets, test_set, optimizer, criterion, epoch, is_train=True):
	
	N = args.client_num_per_round  # number of clients in each round = client_num_per_round
	C = args.comm_round  # number of communication rounds = comm_round
	B = args.batch_size  # batch size = batch_size
	E = args.local_ep  # the number of local epochs = local_ep
	lr = args.learning_rate  # learning rate = learning_rate
	mu = args.mu  # mu = mu
	decay = args.weight_decay  # weight decay = weight_decay
	optimizer_name = args.client_optimizer  # optimizer = client_optimizer

	K = len(training_sets)  # number of clients = client_num_in_total
	n_samples = sum([len(db.dataset) for db in training_sets])
	weights = [len(db.dataset) / n_samples for db in training_sets]
	print("Clients' weights:", weights)
	loss_hist = [float(loss_dataset(model, ds, criterion)) for ds in training_sets]
	acc_hist = [float(accuracy_dataset(model, ds)) for ds in training_sets]
	
	for com_rnd_idx in range(args.comm_round):
		clients_params = []
		clients_optimizers = []
		clients_losses = []
		clients_accs = []
		clients_models = []
		for client_idx in range(args.client_num_per_round):
			client_model = deepcopy(model)
			client_model.train()
			client_optimizer = get_optimizer(optimizer_name, client_model, lr=lr, weight_decay=decay)
			client_optimizer.zero_grad()
			client_params = client_model.state_dict()
			clients_params.append(client_params)
			clients_optimizers.append(client_optimizer)
			clients_models.append(client_model)
			clients_losses.append(0)
			clients_accs.append(0)
		# training for each client
		for client_idx in range(args.client_num_per_round):
			client_model = clients_models[client_idx]
			client_optimizer = clients_optimizers[client_idx]
			client_params = clients_params[client_idx]
			client_loss = clients_losses[client_idx]
			client_acc = clients_accs[client_idx]
			# client_optimizer = get_optimizer(optimizer_name, client_model, lr=lr, weight_decay=decay)
			# client_optimizer.zero_grad()
			# client_params = client_model.state_dict()
			# client_loss = 0
			# client_acc = 0
			for local_ep in range(E):
				# print("Client", client_idx, "Local Epoch", local_ep)
				# client_model.train()
				# client_optimizer.zero_grad()
				# client_params = client_model.state_dict()
				# client_loss = 0
				# client_acc = 0
				train_loader = DataLoader(training_sets[client_idx].dataset, batch_size=B, shuffle=True)
				for batch_idx, (data, target) in enumerate(train_loader):
					data, target = data.to(device), target.to(device)
					client_optimizer.zero_grad()
					output = client_model(data)
					loss = criterion(output, target)
					loss.backward()
					client_optimizer.step()
					pred = output.argmax(dim=1, keepdim=True)
					correct = pred.eq(target.view_as(pred)).sum().item()
					client_loss += loss.item() * len(data)
					client_acc += correct
	 
			client_loss /= len(train_loader.dataset)
			client_acc /= len(train_loader.dataset)
   
	pass


def get_args(json_file):
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
		
	elif args.dataset == "TinyImageNet":
		train_dataset = torchvision.datasets.ImageFolder(
			root='./data/tiny-imagenet-200/train',
			transform=transforms.Compose([
				transforms.RandomResizedCrop(64),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225]),
			]))

		test_dataset = torchvision.datasets.ImageFolder(
			root='./data/tiny-imagenet-200/val',
			transform=transforms.Compose([
				transforms.Resize(64),
				transforms.CenterCrop(64),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225]),
			]))
		
	train_loader = torch.utils.data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, **kwargs)
	
	test_loader = torch.utils.data.DataLoader(test_dataset,
		batch_size=args.test_batch_size, shuffle=False, **kwargs)

	return train_dataset, test_dataset, train_loader, test_loader

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

def set_to_zero_model_weights(model):
	"""Set all the parameters of a model to 0"""

	for layer_weigths in model.parameters():
		layer_weigths.data.sub_(layer_weigths.data)

def average_models(model, clients_models_hist:list , weights:list):
	

	"""Creates the new model of a given iteration with the models of the other
	clients"""
	
	new_model=deepcopy(model)
	set_to_zero_model_weights(new_model)

	for k, client_hist in enumerate(clients_models_hist):
		list_params = list(client_hist.parameters())
		for idx, layer_weights in enumerate(new_model.parameters()):
			# import IPython
			# IPython.embed()
			contribution=list_params[idx].data*weights[k]
			layer_weights.data.add_(contribution)
			
	return new_model

# def fed_avg_aggregator(net_list, net_freq, model):
#     #net_avg = VGG('VGG11').to(device)
#     net_avg = deepcopy(model)
#     whole_aggregator = []
	
#     for p_index, p in enumerate(net_list[0].parameters()):
#         # initial 
#         params_aggregator = torch.zeros(p.size()).to(device)
#         for net_index, net in enumerate(net_list):
#             # we assume the adv model always comes to the beginning
#             params_aggregator = params_aggregator + net_freq[net_index] * list(net.parameters())[p_index].data
#         whole_aggregator.append(params_aggregator)

#     for param_index, p in enumerate(net_avg.parameters()):
#         p.data = whole_aggregator[param_index]
#     return net_avg


def fed_avg_aggregator(net_list, net_freq, device, model):
	# diff = dict()
	# for name, data_layer in 
	print(f"Aggregating models with FedAvg {net_freq}")
	whole_aggregator = []
	# new_model=deepcopy(model)
	# set_to_zero_model_weights(new_model)
	weight_accumulator = {}
	for name, params in model.state_dict().items():
		weight_accumulator[name] = torch.zeros(params.size()).to(device)
	
	for net_index, net in enumerate(net_list):
		# diff = {}
		for name, data in net.state_dict().items():
			# diff[name] = data - model.state_dict()[name]
			weight_accumulator[name].add_(net_freq[net_index] * (data - model.state_dict()[name]))
			# print("net_freq", name, (net_freq[net_index] * (data - model.state_dict()[name]))[0][0][0])
			# print("weight_accumulator[name]: ", name, weight_accumulator[name][0][0][0])
			# print("model.state_dict()[name]: ", name, model.state_dict()[name][0][0][0])
			# break
		# for name, params in net.state_dict().items():
			# weight_accumulator[name].add_(net_freq[net_index] * params)
	for name, params in model.state_dict().items():
		update_per_layer = weight_accumulator[name]
		if params.type() != update_per_layer.type():
			params.add_(update_per_layer.to(torch.int64))
		else:
			params.add_(update_per_layer)
	# import IPython
	# IPython.embed()
	# for name, params in model.state_dict().items():
	# 	print("Model 1: ", net_list[0].state_dict()[name][0][0][0])
	# 	print("Model 2: ", net_list[1].state_dict()[name][0][0][0])
	# 	print("Main Model: ", model.state_dict()[name][0][0][0])
	# 	break
	# for p_index, p in enumerate(net_list[0].parameters()):
	# 	# initial 
	# 	params_aggregator = torch.zeros(p.size()).to(device)
	# 	for net_index, net in enumerate(net_list):
	# 		# we assume the adv model always comes to the beginning
	# 		params_aggregator = params_aggregator + net_freq[net_index] * list(net.parameters())[p_index].data
	# 		if p_index == 3:
	# 			# get the sum of all weight for layer p_index
	# 			# import IPython
	# 			# IPython.embed()
	# 			sum_weight = list(net.parameters())[p_index].data.sum()
	# 			print(sum_weight)
	# 	whole_aggregator.append(params_aggregator)
	# for name, data in model.state_dict().items():
	# 	data.add_()	
	
	# exit(0)
	# for param_index, p in enumerate(model.parameters()):
	# 	p.data =  whole_aggregator[param_index]
	# 	if param_index == 0:
	# 		print(p.data)
  
	# for key_item_1 in model.state_dict().items():
	# 	print(key_item_1)
	# 	break

	# for key_item_1 in net_list[0].state_dict().items():
	# 	print(key_item_1)
	# 	break
	# import IPython
	# IPython.embed()
	# for id, pr in enumerate(model.state_dict().items()):
	# 	# if sum of weight in this layer is 0 then print id
	# 	if (pr[1].sum() == 0):
	# 		print(id, pr[0])
	# 	# print(id, pr)
	# 	# break
	# print("Tuan end here")
	# return model
	#     print(param_index, whole_aggregator[param_index])
		# import IPython
		# IPython.embed()
	# check new_model and net_list[0] are the same
	# for p_index, p in enumerate(net_list[0].parameters()):
	#     print(p_index, torch.equal(p.data, list(new_model.parameters())[p_index].data))
	
	# return new_model

def main(json_config="./configs/fedml_config_yaml.json"):
	# set seed for training
	torch.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	np.random.seed(0)
	# random.seed(0)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	# Training settings
	args_dict = get_args(json_config)
	
	args = args_dict.fed_training
	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(args.seed)


	device = torch.device(args.device if use_cuda else "cpu")

	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	print("kwargs = ", kwargs, "device = ", device, "use_cuda = ", use_cuda, "args.device = ", args.device)
	
	train_dataset, test_dataset, train_loader, test_loader = get_dataset(args, kwargs)
 
	train_loaders, train_indices, test_loader = generate_iid_data(train_dataset, test_dataset, args)
	# import IPython
	# IPython.embed()
 
	# print("train_indices = ", train_indices)
	
	# print(len(train_loader.dataset), len(test_loader.dataset))
	# print(len(train_loader), len(test_loader))
	# exit(0)
	
	model, optimizer, scheduler, criterion = get_model(args, device)
	mu = args.mu
 
	# TODO here
	
	
	# check number of class for each clients
	# for i in range(args.client_num_in_total):
	# 	print(f"client {i} has {len(train_indices[i])} samples")
	# 	cnt_classes = [0 for _ in range(10)]
	# 	# get the list of label from train_loaders[i]
	# 	for idx, (images, labels) in enumerate(train_loaders[i]):
	# 		for j in range(len(labels)):
	# 			cnt_classes[labels[j]] += 1
	# 	# for j in range(len(train_indices[i])):
	# 	# 	import IPython
	# 	# 	IPython.embed()
	# 	# 	cnt_classes[train_indices[i][j][1]] += 1
	# 	print(f"client {i} has {cnt_classes} samples")
	# exit(0)
 
	comm_round = args.comm_round
	n_client_per_round = args.client_num_per_round
 
	client_weight = [1/n_client_per_round for _ in range(args.client_num_in_total)]
	model.to(device)
	# for cr_index in range(1, args.comm_round + 1):
	# 	train_test_model(args, model, device, train_loaders[0], optimizer, criterion, cr_index, is_train=True)
	# 	train_test_model(args, model, device, test_loader, optimizer, criterion, cr_index, is_train=False)
  
	# exit(0)
	for cr_index in range(1, args.comm_round + 1):
		clients_in_round = np.random.choice(args.client_num_in_total, n_client_per_round, replace=False)
		print(f"round: {cr_index} clients_in_round: {clients_in_round}")
  
		client_model_list = [deepcopy(model) for _ in range(n_client_per_round)]
		total_sample_per_round = 0
		for id_0, client_id in enumerate(clients_in_round):
			total_sample_per_round += len(train_indices[client_id])
		models_weight = [len(train_indices[client_id])/total_sample_per_round for client_id in clients_in_round]
		print("Set models_weight = ", models_weight)
		for id_0, client_id in enumerate(clients_in_round):
			# if client_id not in clients_in_round:
			# 	continue
			print(f"Start training round {cr_index} for client number {client_id}")
   
			client_loss, client_acc = client_learning(client_model_list[id_0], mu, args, criterion, train_loaders[client_id], test_loader, device, args.lc_epoch, logger)
			# client_model_list.append(client_model)
			print(f"End training round {cr_index} for client number {client_id} loss = {client_loss} acc = {client_acc:.2f}")
			print("----------------------------------------")
   
	 		# train_test_model(args, model, device, train_loaders[client_id], optimizer, criterion, epoch, is_train=True)
		print(f"Done training round {cr_index} for all clients")
		# aggregate_weights = [1/args.client_num_in_total for _ in range(args.client_num_in_total)]
		# global_model = average_models(model, client_model_list, client_weight)
		# model = average_models(client_model_list, client_weight, device, model)
		# model = average_models(model, client_model_list, models_weight)
  
		fed_avg_aggregator(client_model_list, models_weight, device, model)
		# net_list, net_freq, device, model):
		print("Start testing")
		train_test_model(args, model, device, test_loader, optimizer, criterion, cr_index, is_train=False)
		# model = global_model
  
		# print("Testing by the first client model")
  
		# def compare_models(model_1, model_2):
		# 	models_differ = 0
		# 	for idlyer, (key_item_1, key_item_2) in enumerate(zip(model_1.state_dict().items(), model_2.state_dict().items())):
		# 		if torch.equal(key_item_1[1], key_item_2[1]):
		# 			print(idlyer, 'Match yes found at', key_item_1[0])
		# 			pass
		# 		else:
		# 			models_differ += 1
		# 			if (key_item_1[0] == key_item_2[0]):
		# 				print(idlyer, 'Mismtach found at', key_item_1[0], key_item_2[0])
		# 				# print(key_item_1[1], key_item_2[1])
		# 				# exit(0)
		# 			else:
		# 				print(idlyer, 'Match found at', key_item_1[0])
		# 				# raise Exception
		# 	if models_differ == 0:
		# 		print('Models match perfectly! :)')
    
		# compare_models(client_model_list[0], model)
		# exit(0)
		# import IPython
		# IPython.embed()
		# for p_index, p in enumerate(model.parameters()):
		# 	print(p_index, torch.equal(p.data, list(client_model_list[0].parameters())[p_index].data))	
		# train_test_model(args, client_model_list[0], device, test_loader, optimizer, criterion, cr_index, is_train=False)
		# train_test_model(args, model, device, deepcopy(test_loader), optimizer, criterion, cr_index, is_train=False)
  
		# get iteration of test_loader
		# import IPython
		# IPython.embed()
		# getsp = next(iter(test_loader))
  
		# op1 = client_model_list[0](getsp)
		# print("Testing by aggregating all clients model")
		# train_test_model(args, model, device, test_loader, optimizer, criterion, cr_index, is_train=False)
  
		# exit(0)
		# train_test_model(args, model, device, test_loader, optimizer, criterion, cr_index, is_train=False)
		
		# op2 = model(getsp)
		# print("Testing by the first client model again")
		# print(op1)
		# print(op2)


	
		# train_test_model(args, model, device, test_loader, criterion, criterion, cr_index, is_train=False)
  
	# 	train_test_model(args, model, device, train_loader, optimizer, criterion, epoch, is_train=True)
	# 	train_test_model(args, model, device, test_loader, criterion, criterion, epoch, is_train=False)

	# 	for param_group in optimizer.param_groups:
	# 		logger.info(param_group['lr'])
	# 	scheduler.step()

		# if epoch % 5 == 0:
		#     torch.save(model.state_dict(), "./checkpoint/{}_{}_{}epoch.pt".format(args.dataset, args.model.upper(), args.epochs))


if __name__ == '__main__':
	# get name of file json from command line
	parser =  argparse.ArgumentParser(description='PyTorch MNIST Example')
	# parser.add_argument('--config', type=str, default="./configs/cifar10_030823", help='config file')
	parser.add_argument('--config', type=str, default="./configs/cifar10_030823.json", help='config file')
	args = parser.parse_args()
	main(json_config=args.config)
	
