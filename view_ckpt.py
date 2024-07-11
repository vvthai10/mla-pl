import torch 

path = "/home/phu/workspace/thesis/sources/mvfa-ad/ckpts/few-shot/4-shot/test/Liver.pth"
checkpoint = torch.load(path)
print(checkpoint["AUC"])
print(checkpoint["pAUC"])
