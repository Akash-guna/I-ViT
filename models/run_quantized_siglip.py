from siglip import SiglipVisionModel
import torch
model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
model = model.cuda()
random_number = torch.randn((1,3,224,224))
model(random_number.cuda())
print(random_number.shape)