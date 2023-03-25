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

# sample here: https://github.com/Xtra-Computing/NIID-Bench
# write a class that accumulate the loss, and accuracy for each epoch
# https://arxiv.org/pdf/2209.15595.pdf

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def train_test_model_one_epoch(args, train_test_model, device, data_loader, optimizer, criterion, epoch, is_train=True):

	class_correct = list(0.0 for i in range(10))
	class_total = list(0.0 for i in range(10))
	if args.dataset == "EMNIST":
		classes = [str(i) for i in range(10)]
	elif args.dataset == "MNIST":
		classes = [str(i) for i in range(10)]
	elif args.dataset == "Cifar10":
		classes = (
			"plane",
			"car",
			"bird",
			"cat",
			"deer",
			"dog",
			"frog",
			"horse",
			"ship",
			"truck",
		)

	elif args.dataset == "TinyImageNet":  # 200 class
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
		train_test_model.train()
	else:
		train_test_model.eval()

	running_loss = AverageMeter()
	running_acc = AverageMeter()

	pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
	total_correct = 0

	# using torch.no_grad() when is_train == True
	# https://discuss.pytorch.org/t/what-does-model-eval-do-for-pytorch-models/7146/2

	running_str = "Train" if is_train else "Test"
	train_test_model.to(device)
	# with torch.set_grad_enabled(is_train):
	for batch_idx, (data, target) in pbar:
		data, target = data.to(device), target.to(device)
		if is_train:
			optimizer.zero_grad()
		output = train_test_model(data)
		loss = criterion(output, target)

		if is_train:
			loss.backward()
			optimizer.step()

		pred = output.argmax(
			dim=1, keepdim=True
		) 
		correct_pred = pred.eq(target.view_as(pred))
		correct = correct_pred.sum().item()
		total_correct += correct

		running_loss.update(loss.item(), data.size(0))
		running_acc.update(correct / data.size(0), data.size(0))

		pbar.set_description(
			"{} Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(
				running_str, epoch, running_loss.avg, running_acc.avg
			)
		)

		if not is_train:
			for image_index in range(len(target)):
				label = target[image_index]
				class_correct[label] += correct_pred[image_index].item()
				class_total[label] += 1

	if not is_train:
		for i in range(10):
			logger.info(
				"%s Accuracy of %5s : %2d %%"
				% (running_str, classes[i], 100 * class_correct[i] / class_total[i])
			)

	logger.info(
		"\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
			running_str,
			running_loss.avg,
			total_correct,
			len(data_loader.dataset),
			100.0 * total_correct / len(data_loader.dataset),
		)
	)

	return running_loss, running_acc



def get_optimizer(optimizer_name, client_model, lr, weight_decay):
	if optimizer_name == "sgd":
		return optim.SGD(client_model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == "adam":
		return optim.Adam(client_model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optimizer_name == "adagrad":
		return optim.Adagrad(
			client_model.parameters(), lr=lr, weight_decay=weight_decay
		)
	elif optimizer_name == "adadelta":
		return optim.Adadelta(
			client_model.parameters(), lr=lr, weight_decay=weight_decay
		)
	elif optimizer_name == "rmsprop":
		return optim.RMSprop(
			client_model.parameters(), lr=lr, weight_decay=weight_decay
		)
	else:
		raise Exception("Optimizer not supported")


def difference_models_norm_2(model_1, model_2):
	"""Return the norm 2 difference between the two model parameters"""

	tensor_1 = list(model_1.parameters())
	tensor_2 = list(model_2.parameters())

	norm = sum(
		[torch.sum((tensor_1[i] - tensor_2[i]) ** 2) for i in range(len(tensor_1))]
	)

	return norm


def client_learning(
	model, mu, args, criterion, train_loader, test_loader, device, local_ep, logger
):
	original_model = deepcopy(model)

	client_optimizer = get_optimizer(
		args.client_optimizer, model, args.lr, args.weight_decay
	)

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
			   
				diff_norm = mu / 2 * difference_models_norm_2(model, original_model)
				loss = criterion(output, target) + diff_norm
				# loss = criterion(output, target)
				# .item()*len(labels)

				loss.backward()
				client_optimizer.step()
				pred = output.argmax(dim=1, keepdim=True)
				correct_pred = pred.eq(target.view_as(pred))
				correct = correct_pred.sum().item()
				total_correct += correct

				total_loss += loss.item() * len(data) + diff_norm
				total_sample += len(data)
				# print("here", batch_idx, len(data), total_loss, total_sample, total_correct)
			# print(total_loss, total_sample, total_correct)
			pbar.set_description(
				f"Local Epoch: {e + 1}/ {local_ep}, Loss: {total_loss / total_sample:.4f}, Acc: {total_correct/ total_sample:.4f}"
			)

	return total_loss / total_sample, total_correct / total_sample


def generate_iid_data(train_dataset, test_dataset, args):
	N = args.client_num_in_total  # number of clients in total = client_num_in_total
	all_range = list(range(len(train_dataset)))
	data_len_per_client = len(train_dataset) // N
	# print("data_len_per_client: ", data_len_per_client)
	train_indices = [
		all_range[i * data_len_per_client : (i + 1) * data_len_per_client]
		for i in range(N)
	]
	train_loaders = [
		torch.utils.data.DataLoader(
			train_dataset,
			batch_size=args.batch_size,
			sampler=SubsetRandomSampler(indices),
		)
		for indices in train_indices
	]
	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=args.batch_size, shuffle=False
	)

	return train_loaders, train_indices, test_loader

def generata_non_iid_data(train_dataset, test_dataset, args):
	
	partition_method = args.partition_method
	# list object to numpy
	y_train = np.array(train_dataset.targets)
	# y_train = train_dataset.targets.umpy()
 
	number_of_classs = len(np.unique(y_train))
	n_train = len(train_dataset)
	n_nets = args.client_num_in_total
		
	if partition_method == "hetero-dir":
		partition_alpha = args.partition_alpha
		min_size = 0
		min_required_size = 10
		K = number_of_classs # number of classes
		dataset = args.dataset
		net_dataidx_map = {}

		while (min_size < min_required_size) or (dataset == 'mnist' and min_size < 100):
			idx_batch = [[] for _ in range(n_nets)]
			# for each class in the dataset
			for k in range(K):
				idx_k = np.where(y_train == k)[0]
				np.random.shuffle(idx_k)
				proportions = np.random.dirichlet(np.repeat(partition_alpha, n_nets))
				## Balance
				proportions = np.array([p*(len(idx_j) < n_train/n_nets) for p,idx_j in zip(proportions,idx_batch)])
				proportions = proportions/proportions.sum()
				proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
				idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
				min_size = min([len(idx_j) for idx_j in idx_batch])

		for j in range(n_nets):
			np.random.shuffle(idx_batch[j])
			net_dataidx_map[j] = idx_batch[j]
	elif partition_method == "homo":
		print("Go to this {} method".format(partition_method))
		idxs = np.random.permutation(n_train)
		batch_idxs = np.array_split(idxs, n_nets)
		net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
	# count how many samples in each client
	total_sample = 0
	for j in range(n_nets):
		print("Client %d: %d samples" % (j, len(net_dataidx_map[j])))
		cnt_class = {}
		for i in net_dataidx_map[j]:
			label = y_train[i]
			if label not in cnt_class:
				cnt_class[label] = 0
			cnt_class[label] += 1
		total_sample += len(net_dataidx_map[j])
		print("Client %d: %s" % (j, str(cnt_class)))
		print("--------"*10)
	print("Total training: %d samples" % total_sample)
	print("Total testing: %d samples" % len(test_dataset))
	# import IPython; IPython.embed()
	train_loaders = [
		torch.utils.data.DataLoader(
			train_dataset,
			batch_size=args.batch_size,
			sampler=SubsetRandomSampler(indices),
		)
		for _, indices in net_dataidx_map.items()
	]
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
	# print(len(train_loaders[0]), len(test_loader))
	# exit(0)
	return train_loaders, test_loader, net_dataidx_map


def get_args(json_file):
	with open(json_file, "r") as f:
		args = json.load(f)
	return EasyDict(args)


def get_dataset(args, kwargs):
	if args.dataset == "EMNIST":
		train_dataset = datasets.EMNIST(
			"./data",
			split="digits",
			train=True,
			download=True,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		)
		test_dataset = datasets.EMNIST(
			"./data",
			split="digits",
			train=False,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		)
	elif args.dataset == "MNIST":
		train_dataset = datasets.MNIST(
			"./data",
			train=True,
			download=True,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		)
		test_dataset = datasets.MNIST(
			"./data",
			train=False,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		)

	elif args.dataset == "Cifar10":
		transform_train = transforms.Compose(
			[
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(
					(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
				),
			]
		)

		transform_test = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize(
					(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
				),
			]
		)

		train_dataset = torchvision.datasets.CIFAR10(
			root="./data", train=True, download=True, transform=transform_train
		)

		test_dataset = torchvision.datasets.CIFAR10(
			root="./data", train=False, download=True, transform=transform_test
		)

	elif args.dataset == "TinyImageNet":
		train_dataset = torchvision.datasets.ImageFolder(
			root="./data/tiny-imagenet-200/train",
			transform=transforms.Compose(
				[
					transforms.RandomResizedCrop(64),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize(
						mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
					),
				]
			),
		)

		test_dataset = torchvision.datasets.ImageFolder(
			root="./data/tiny-imagenet-200/val",
			transform=transforms.Compose(
				[
					transforms.Resize(64),
					transforms.CenterCrop(64),
					transforms.ToTensor(),
					transforms.Normalize(
						mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
					),
				]
			),
		)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
	)

	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
	)

	return train_dataset, test_dataset, train_loader, test_loader


def get_model(args, device):
	if args.model == "LeNet":
		# torch.manual_seed(0)
		# torch.cuda.manual_seed_all(0)
		model = Net(num_classes=10).to(device)
		# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
		# optimizer = optim.Adam(model.parameters(), lr=args.lr)
		optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
		scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
		# load pretrained model (./checkpoint/mnist_cnn.pt)
		# model.load_state_dict(torch.load('./checkpoint/mnist_cnn.pt'))

	elif args.model in ("vgg9", "vgg11", "vgg13", "vgg16"):
		model = get_vgg_model(args.model).to(device)
		# model = VGG(args.model.upper()).to(device)
		optimizer = optim.SGD(
			model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
		)
		scheduler = MultiStepLR(
			optimizer, milestones=[e for e in [151, 251]], gamma=0.1
		)

	elif args.model in ("ResNet18"):
		model = ResNet18().to(device)
		optimizer = optim.SGD(
			model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
		)
		scheduler = MultiStepLR(
			optimizer, milestones=[e for e in [151, 251]], gamma=0.1
		)

	criterion = nn.CrossEntropyLoss()
	# criterion = nn.NLLLoss()
	return model, optimizer, scheduler, criterion




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
			weight_accumulator[name].add_(
				net_freq[net_index] * (data - model.state_dict()[name])
			)
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

	kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
	print(
		"kwargs = ",
		kwargs,
		"device = ",
		device,
		"use_cuda = ",
		use_cuda,
		"args.device = ",
		args.device,
	)

	train_dataset, test_dataset, train_loader, test_loader = get_dataset(args, kwargs)

	# train_loaders, train_indices, test_loader = generate_iid_data(
	# 	train_dataset, test_dataset, args
	# )
	train_loaders, test_loader, net_dataidx_map = generata_non_iid_data(train_dataset, test_dataset, args)

	# exit(0)
 
	model, optimizer, scheduler, criterion = get_model(args, device)
	mu = args.mu



	comm_round = args.comm_round
	n_client_per_round = args.client_num_per_round

	model.to(device)

	for cr_index in range(1, args.comm_round + 1):
		clients_in_round = np.random.choice(
			args.client_num_in_total, n_client_per_round, replace=False
		)
		print(f"round: {cr_index} clients_in_round: {clients_in_round}")

		client_model_list = [deepcopy(model) for _ in range(n_client_per_round)]
		# global_model_param = model.state_dict()
		# client_model_list = []
		total_sample_per_round = 0
		for id_0, client_id in enumerate(clients_in_round):
			total_sample_per_round += len(net_dataidx_map[client_id])
		models_weight = [
			len(net_dataidx_map[client_id]) / total_sample_per_round
			for client_id in clients_in_round
		]
		print("Set models_weight = ", models_weight)
		for id_0, client_id in enumerate(clients_in_round):
			# if client_id not in clients_in_round:
			# 	continue
			print(f"Start training round {cr_index} for client number {client_id}")

			client_loss, client_acc = client_learning(
				client_model_list[id_0],
				mu,
				args,
				criterion,
				train_loaders[client_id],
				test_loader,
				device,
				args.lc_epoch,
				logger,
			)
			# client_model_list.append(client_model)
			print(
				f"End training round {cr_index} for client number {client_id} loss = {client_loss} acc = {client_acc:.2f}"
			)
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
		train_test_model_one_epoch(
			args,
			model,
			device,
			test_loader,
			optimizer,
			criterion,
			cr_index,
			is_train=False,
		)
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


if __name__ == "__main__":
	# get name of file json from command line
	parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
	# parser.add_argument('--config', type=str, default="./configs/cifar10_030823", help='config file')
	parser.add_argument(
		"--config",
		type=str,
		default="./configs/cifar10_030823.json",
		help="config file",
	)
	args = parser.parse_args()
	main(json_config=args.config)
