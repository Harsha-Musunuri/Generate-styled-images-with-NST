#misc
from PIL import Image
import matplotlib.pyplot as plt

#torch relted
import torch
import torchvision.transforms as transforms



#imageLoader
def image_loader(image_name,device,imsize):
    image = Image.open(image_name)
    transform = transforms.Compose([
        transforms.CenterCrop(imsize),
        # transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    # fake batch dimension required to fit network's input dimensions
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


#imageInteractiveVisualizer
def imshow(tensor, title=None):
    plt.ion()
    backTransform = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = backTransform(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated