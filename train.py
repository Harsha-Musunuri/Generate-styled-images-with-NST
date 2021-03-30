#misc
import argparse
from dataset import *
from model import *
import os
from tqdm import tqdm

#torchRelated
import torch
import torchvision.models as models
from torchvision import utils



def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img,device, epochs=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img,content_layers_default,style_layers_default,device)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    epoch = [0]
    while epoch[0] <= epochs:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print("epoch {}:".format(epoch))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NST")

    #args
    parser.add_argument("--name", type=str, help="name to save styled Img")
    parser.add_argument("--noise", action='store_true')
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train ?")
    parser.add_argument("--styleImg", type=str, default='path to styleImg')
    parser.add_argument("--contentImg", type=str, default='path to contentImg')
    parser.add_argument("--log_root", type=str, help="where to save training logs", default='logs')


    args = parser.parse_args()


    # data processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # desired size of the output image
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

    # load images
    style_img = image_loader(args.styleImg, device, imsize)
    content_img = image_loader(args.contentImg, device, imsize)

    # assert if the two images are of same size or not
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # load VGG19 model for feature extraction
    cnn = models.vgg19(pretrained=False)
    cnn.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))
    cnn = cnn.features.to(device).eval()

    '''
    VGG networks are trained on images with each channel normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]. 
    We will use them to normalize the image before sending it into the network.
    '''
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    if args.noise:
        input_img = torch.randn(content_img.data.size(), device=device)
    else:
        input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,device,epochs=args.epochs)

    utils.save_image(
                        output,
                        args.name+'.png',
                        # normalize=True,
                        range=(-1, 1),
                        )
