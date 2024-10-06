from .siglip import SiglipVisionModel

model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
print(model)
