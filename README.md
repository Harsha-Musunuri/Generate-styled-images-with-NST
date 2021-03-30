# Neural Style Transfer
simply put: Use convolution's capability to find feature maps of a content image and style image and train an input image to get style from style image while retaining its content
# Architecture
- VGG19 for image's feature extraction
- Using conv4 for content layers and conv1 to conv5 for style layers
# Training
- Run train.py with necessary argumnets like location of style and content images, this will train the image and output is saved in the working directory
# Results
vincent van gogh: starry night             |  styled image
:-------------------------:|:-------------------------:
<img src="https://github.com/Harsha-Musunuri/Generate-styled-images-with-NST/blob/master/images/styleImages/vanGogh-theStarryNight.jpg" width="400" height="400">  | <img src="https://github.com/Harsha-Musunuri/Generate-styled-images-with-NST/blob/master/results/styled2.png" width="400" height="400">
Eugène Delacroix: Liberty Leading the People             |  styled image
<img src="https://github.com/Harsha-Musunuri/Generate-styled-images-with-NST/blob/master/images/styleImages/TheLiberty.jpeg" width="400" height="400">  | <img src="https://github.com/Harsha-Musunuri/Generate-styled-images-with-NST/blob/master/results/styled1.png" width="400" height="400">

# To-Do/Implement:
- control style at will - Controlling Perceptual Factors in Neural Style Transfer (https://arxiv.org/pdf/1611.07865.pdf)
- use adaptive instance normalization to do the style transfer (https://arxiv.org/pdf/1703.06868.pdf - Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization)
