from networks.network import DeepCFRModel
import torch


device = torch.device("cpu")
net = DeepCFRModel(f"ply_test", lr=1e-3).to(device)


input = (
    torch.randint(0,50,(12, 5)).to(device),
    torch.randint(0,50,(12, 2)).to(device),
    torch.randn((12, 3)).to(device),

    torch.randn((12, 3)).to(device),

    torch.randn((12, 3)).to(device),
    torch.randn((12, 3)).to(device),
    torch.randint(0,3,(12, 1)).to(device),
    torch.randint(0,2,(12, 1)).to(device),
)

net(input)