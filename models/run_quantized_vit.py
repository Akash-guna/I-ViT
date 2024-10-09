from vit_quant import deit_base_patch16_224
import torch

model = deit_base_patch16_224(pretrained=False)
model = model.cuda()
random_number = torch.randn((1,3,224,224))
model(random_number.cuda())
print(random_number.shape)
