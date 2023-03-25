import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
bn = torch.nn.BatchNorm1d(1).cuda()
t = 100 * torch.ones([1,1,2]).cuda()
print(bn(t.cuda()))

# prints tensor([[[ 0.,  0.]]], device='cuda:0')

torch.backends.cudnn.enabled = True

bn = torch.nn.BatchNorm1d(1).cuda()
t = 100 * torch.ones([1,1,2]).cuda()
print(bn(t.cuda()))