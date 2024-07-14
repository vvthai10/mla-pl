import torch 

path = "D:/workspace/thesis/sources/mvfa-ad/ckpts/zero-shot/v3/Brain.pth"
checkpoint = torch.load(path)
print(checkpoint["AUC"])
print(checkpoint["pAUC"])
