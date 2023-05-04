import torch

a = torch.arange(6).reshape(2, 3)
print (a)

torch.einsum('ij->ji', [a])  #Transpuesta
print (a)
