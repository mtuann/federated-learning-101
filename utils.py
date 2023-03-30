import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        seed = 0
        random.seed(seed)                          
        np.random.seed(seed)                       
        torch.manual_seed(seed)                    
        torch.cuda.manual_seed(seed)               
        torch.cuda.manual_seed_all(seed)           
        torch.backends.cudnn.deterministic = True
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        seed = 0
        torch.cuda.manual_seed(seed)               
        torch.cuda.manual_seed_all(seed)   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        # output = F.log_softmax(output, dim=1)
        
        return output
    

# import yaml
# import json
# import argparse

# from easydict import EasyDict


# def get_parser_args(json_file):
#     # get value from json file
#     with open(json_file, 'r') as f:
#         args = json.load(f)
#     # convert json key and sub key to argparse key
#     # for key in args.keys():
#     #     if isinstance(args[key], dict):
#     #         for sub_key in args[key].keys():
#     #             args[key + '_' + sub_key] = args[key][sub_key]
#     #         args.pop(key)
#     args_dict = EasyDict(args)
#     return args_dict

#     print(args_dict)
#     import IPython
#     IPython.embed()
    
#     args = argparse.Namespace(**args)
#     print(args)
#     print("--"*30)
#     print(args.fed_training)
#     fed_args = argparse.Namespace(**args.fed_training)
#     print(fed_args)
#     return args
# args = get_parser_args('configs/fedml_config_yaml.json')
# # print(args)
# # Open the YAML file and load its contents
# # with open('configs/fedml_config.yaml') as file:
# #     yaml_data = yaml.load(file, Loader=yaml.FullLoader)

# # # Convert the YAML data to a JSON string
# # # json_string = json.dumps(yaml_data)

# # # Print the JSON string
# # # print(json_string)
# # # json_dict = json.loads(json_string)

# # # write to json file configs/fedml_config.json
# # # with open('configs/fedml_config.json', 'w') as file:
# # #     json.dump(json_dict, file, indent=2)    

# # # yaml_data to json file
# # # with open('configs/fedml_config_yaml.json', 'w') as file:
# # #     json.dump(yaml_data, file, indent=2)
# # print(type(yaml_data), yaml_data.keys())
# # import argparse
# # args = argparse.Namespace(**yaml_data)
# # print(args)
# # import IPython
# # IPython.embed()

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