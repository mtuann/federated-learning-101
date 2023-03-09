import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
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
        #output = F.log_softmax(x, dim=1).
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