![PytorchRevelio](/readme-images/feature-class-representation.jpg)
![PytorchRevelio](/readme-images/Saliency-map.jpg)

[![License](https://img.shields.io/github/license/farhad-dalirani/PytorchRevelio)](https://github.com/farhad-dalirani/PytorchRevelio/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4782563.svg)](https://doi.org/10.5281/zenodo.4782563)


# Introduction of PytorchRevelio 

PytorchRevelio is a collection of classes and functions that allow you to investigate MLP and convolutional networks written in Pytorch. These classes and functions enable you to visualize features that neurons and filters have learned or illustrate a saliency map for a given image. Even a Deep-Dream function is provided to have more fun. You can calculate visualization of learned features and saliency maps with several different methods. Although Some of these methods are better than others, we included weaker methods for educational purposes. For detail about different methods you can see this blog post [Reveling what neural networks see and learn: PytorchRevelio](https://dalirani-1373.medium.com/reveling-what-neural-networks-see-and-learn-pytorchrevelio-a218ef5fc61f)

**Keywords**: Pytorch, MLP Neural Networks, Convolutional Neural Networks, Deep Learning, Visualization, Saliency Map, Guided Gradient 


## Where can we use it?
With PytorchRevelio you can investigate MLP and Convolutional neural networks that are written in Pytorch. There is no matter that you have writtern the network or it is a Pytorch build-in neural network such as VGG-19. 


## How to use it in our code?
Download These Files and put them in your code directory:
  * PytorchRevelio.py
  * utilities_PytorchRevelio.py
  * imagenet_labels Folder

In the following, we introduce provided methods and show their use cases with different examples.  

## What should we pay attention to while using these methods?
Visualizing learned feature maps and illustrating saliency maps are highly sensitive to hyper-parameters. Therefore, make sure to choose a good set of hyper-parameters to obtain good results. 

## All methods of PytorchRevelio

Methods of PytorchRevelio | Reference (Paper, Webpage, etc)
--------|--------
activation_maximization(...) | [Visualizing higher-layer features of a deep network](https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network)
activation_maximization_with_gaussian_blurring(...) | [Visualizing GoogLeNet Classes](https://www.auduno.com/2015/07/29/visualizing-googlenet-classes/)
activation_maximization_with_bilateral_blurring(...) | [Class visualization with bilateral filters](https://mtyka.github.io/deepdream/2016/02/05/bilateral-class-vis.html)
saliency_map(...) | [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
saliency_map_guided(...) | [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
deep_dream(...) | [Inceptionism: Going Deeper into Neural Networks](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
grad_cam(...) | [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

For detail about different methods you can see this blog post [Reveling what neural networks see and learn: PytorchRevelio](https://dalirani-1373.medium.com/reveling-what-neural-networks-see-and-learn-pytorchrevelio-a218ef5fc61f)

## Which method does perform better? 
* The higher methods in the list perform better for feature visualization:
  1. activation_maximization_with_bilateral_blurring
  1. activation_maximization_with_gaussian_blurring
  1. activation_maximization
* The higher methods in the list perform better for saliency map:
  1. grad_cam
  1. saliency_map_guided
  1. saliency_map

## Citation
You can use the below text to cite PytorchRevelio.
```
Farhad Dalirani. (2021, May 22). farhad-dalirani/PytorchRevelio: PytorchRevelio-V1.0.0 (Version V1.0.0). Zenodo. http://doi.org/10.5281/zenodo.4780801
```

## Examples of using PytorchRevelio

* Visualizing features of Alexnet with activation_maximization:

```python
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from PytorchRevelio import PytorchRevelio
from utilities_PytorchRevelio import imagenet_labels


if __name__ == '__main__':

    # load pretrained Alexnet
    alexnet_net = torchvision.models.alexnet(pretrained=True)

    # choose GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # put network on device
    alexnet_net.to(device)

    # print name of modules
    for key, value in PytorchRevelio.layers_name_type(alexnet_net):
        print('+' * 10)
        print(key)
        print('-' * 10)
        print(value)

    # network transformer for input image
    img_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for different convolutional filter and neuron in fully connected layer
    # show representation
    first_layer_name = 'features.0'
    last_layer_name = 'classifier.6'
    for layer_name in alexnet_net.named_modules():

        layer_name = layer_name[0]

        # select convolutional and fully connected layers for visualization
        layer = PytorchRevelio.return_module_by_name(network=alexnet_net, module_name=layer_name)

        if isinstance(layer, nn.Conv2d):
            filter_neuron_num = layer.out_channels
            layer_type = 'Conv2d'
            num_iter = 150
            lr = 1
        elif isinstance(layer, nn.Linear):
            filter_neuron_num = layer.out_features
            layer_type = 'Linear'
            num_iter = 500
            lr = 1
        else:
            continue

        # from each layer select 8 filter our neurons
        filters_neuron_indexs = np.random.choice([i for i in range(filter_neuron_num)], size=8)

        # for each selected filter or neuron, calculate representation
        plt.figure()
        for i, filter_neuron_index in enumerate(filters_neuron_indexs):
            img = PytorchRevelio.activation_maximization(network=alexnet_net, img_transformer=img_transformer,
                                                         in_img_size=(224, 224, 3),
                                                         first_layer_name=first_layer_name, layer_name=layer_name,
                                                         filter_or_neuron_index=filter_neuron_index, num_iter=num_iter,
                                                         lr=lr, device=device)

            # to cpu and normalize for illustration purpose
            img = PytorchRevelio.tensor_outputs_to_image(img)

            # Illustrate
            ax = plt.subplot(2, 4, i+1)
            plt.imshow(img)
            if layer_name != last_layer_name:
                ax.set_title("{}".format(filter_neuron_index))
            else:
                ax.set_title("{}, {}".format(filter_neuron_index, imagenet_labels(class_number=filter_neuron_index)))

            plt.suptitle('Layer Name: {}, Type: {}'.format(layer_name, layer_type))
            ax.axis('off')
            print('Processing of layer {}, filter/neuron {} is done.'.format(layer_name, filter_neuron_index))

    plt.show()
```

Some of the outputs:
![PytorchRevelio](/readme-images/Figure_1_alexnet_AM.jpg)
![PytorchRevelio](/readme-images/Figure_3_alexnet_AM.jpg)




* Visualizing features of VGG-11 with activation_maximization_with_gaussian_blurring:

```python
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from PytorchRevelio import PytorchRevelio
from utilities_PytorchRevelio import imagenet_labels

if __name__ == '__main__':

    # load pretrained VGG11
    vgg11_net = torchvision.models.vgg11(pretrained=True)

    # choose GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # put network on device
    vgg11_net.to(device)

    # print name of modules
    for key, value in vgg11_net.named_modules():
        print('+' * 10)
        print(key)
        print('-' * 10)
        print(value)

    # network transformer for input image
    img_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for different convolutional filter and neuron in fully connected layer
    # show representation
    first_layer_name = 'features.0'
    last_layer_name = 'classifier.6'
    for layer_name in vgg11_net.named_modules():

        layer_name = layer_name[0]

        # select convolutional and fully connected layers for visualization
        layer = PytorchRevelio.return_module_by_name(network=vgg11_net, module_name=layer_name)
        if isinstance(layer, nn.Conv2d):
            filter_neuron_num = layer.out_channels
            layer_type = 'Conv2d'
            num_iter = 450
            lr = 0.09
            start_sigma = 2.5
            end_sigma = 0.5
        elif isinstance(layer, nn.Linear):
            filter_neuron_num = layer.out_features
            layer_type = 'Linear'
            num_iter = 450
            lr = 0.09
            start_sigma = 7.5,
            end_sigma = 2.5,
        else:
            continue

        # from each layer select 8 filter our neurons
        filters_neuron_indexs = np.random.choice([i for i in range(filter_neuron_num)], size=8)

        # for each selected filter or neuron, calculate representation
        plt.figure()
        for i, filter_neuron_index in enumerate(filters_neuron_indexs):
            img = PytorchRevelio.activation_maximization_with_gaussian_blurring(
                network=vgg11_net, img_transformer=img_transformer,
                in_img_size=(224, 224, 3),
                first_layer_name=first_layer_name,
                layer_name=layer_name,
                filter_or_neuron_index=filter_neuron_index,
                num_iter=num_iter,
                start_sigma=start_sigma,
                end_sigma=end_sigma,
                lr=lr,
                device=device)

            # to cpu and normalize for illustration purpose
            img = PytorchRevelio.tensor_outputs_to_image(img)

            # Illustrate
            ax = plt.subplot(2, 4, i+1)
            plt.imshow(img)
            if layer_name != last_layer_name:
                ax.set_title("{}".format(filter_neuron_index))
            else:
                ax.set_title("{}, {}".format(filter_neuron_index, imagenet_labels(class_number=filter_neuron_index)))

            plt.suptitle('Layer Name: {}, Type: {}'.format(layer_name, layer_type))
            ax.axis('off')
            print('Processing of layer {}, filter/neuron {} is done.'.format(layer_name, filter_neuron_index))

    plt.show()
```

Some of the outputs:
![PytorchRevelio](/readme-images/Figure_2_vgg11_am_gaussian.jpg)
![PytorchRevelio](/readme-images/Figure_4_vgg11_am_gaussian.jpg)
![PytorchRevelio](/readme-images/Figure_6_vgg11_am_gaussian.jpg)
![PytorchRevelio](/readme-images/Figure_7_vgg11_am_gaussian.jpg)
![PytorchRevelio](/readme-images/Figure_9_vgg11_am_gaussian.jpg)
![PytorchRevelio](/readme-images/Figure_11_vgg11_am_gaussian.jpg)



* Visualizing features of VGG-11 with activation_maximization_with_bilateral_blurring:

```python
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from PytorchRevelio import PytorchRevelio
from utilities_PytorchRevelio import imagenet_labels
from PIL import Image


if __name__ == '__main__':

    # load pretrained VGG11
    vgg11_net = torchvision.models.vgg11(pretrained=True)

    # choose GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # put network on device
    vgg11_net.to(device)

    # print name of modules
    for key, value in vgg11_net.named_modules():
        print('+' * 10)
        print(key)
        print('-' * 10)
        print(value)

    # network transformer for input image
    img_size = (224, 224, 3)
    img_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for different convolutional filter and neuron in fully connected layer
    # show representation
    first_layer_name = 'features.0'
    last_layer_name = 'classifier.6'
    for layer_name in vgg11_net.named_modules():

        layer_name = layer_name[0]

        # select convolutional and fully connected layers for visualization
        layer = PytorchRevelio.return_module_by_name(network=vgg11_net, module_name=layer_name)
        if isinstance(layer, nn.Conv2d):
            filter_neuron_num = layer.out_channels
            layer_type = 'Conv2d'
            num_iter = 300
            lr = 0.09
            start_sigma_color = 25
            end_sigma_color = 110
            start_sigma_space = 25
            end_sigma_space = 110
            kernel_size = 3
        elif isinstance(layer, nn.Linear):
            filter_neuron_num = layer.out_features
            layer_type = 'Linear'
            num_iter = 300
            lr = 0.09
            start_sigma_color = 25
            end_sigma_color = 110
            start_sigma_space = 25
            end_sigma_space = 110
            kernel_size = 3
        else:
            continue

        # from each layer select 8 filter our neurons
        filters_neuron_indexs = np.random.choice([i for i in range(filter_neuron_num)], size=8)

        # for each selected filter or neuron, calculate representation
        plt.figure()
        for i, filter_neuron_index in enumerate(filters_neuron_indexs):
            img = PytorchRevelio.activation_maximization_with_bilateral_blurring(
                network=vgg11_net,
                img_transformer=img_transformer,
                in_img_size=img_size,
                first_layer_name=first_layer_name,
                layer_name=layer_name,
                filter_or_neuron_index=filter_neuron_index,
                num_iter=num_iter,
                start_sigma_color=start_sigma_color,
                end_sigma_color=end_sigma_color,
                start_sigma_space=start_sigma_space,
                end_sigma_space=end_sigma_space,
                kernel_size=kernel_size,
                lr=lr,
                device=device)

            # to cpu and normalize for illustration purpose
            img = PytorchRevelio.tensor_outputs_to_image(img)

            # Illustrate
            ax = plt.subplot(2, 4, i+1)
            plt.imshow(img)
            if layer_name != last_layer_name:
                ax.set_title("{}".format(filter_neuron_index))
            else:
                ax.set_title("{}, {}".format(filter_neuron_index, imagenet_labels(class_number=filter_neuron_index)))

            plt.suptitle('Layer Name: {}, Type: {}'.format(layer_name, layer_type))
            ax.axis('off')
            print('Processing of layer {}, filter/neuron {} is done.'.format(layer_name, filter_neuron_index))

    plt.show()
```

Some of the outputs:
![PytorchRevelio](/readme-images/Figure_7_AM_bilateral.jpg)
![PytorchRevelio](/readme-images/Figure_11_AM_bilateral.jpg)




* Saliency map of ResNet-18 with saliency_map:

```python
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from PytorchRevelio import PytorchRevelio

if __name__ == '__main__':

    # load pretrained resnet18
    resnet18_net = torchvision.models.resnet18(pretrained=True)

    # choose GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))

    # put network on device
    resnet18_net.to(device)

    # print name of modules
    for key, value in resnet18_net.named_modules():
        print('+' * 10)
        print(key)
        print('-' * 10)
        print(value)

    # network transformer for input image
    img_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for different convolutional filter and neuron in fully connected layer
    # show representation
    first_layer_name = 'conv1'
    last_layer_name = 'fc'

    for input_image_name, class_number in [("test_images/kit_fox_278_imagenet.jpg", 278),
                                           ("test_images/bald_eagle_22_imagenet.jpg", 22),
                                           ("test_images/peacock_imagenet_84.jpg", 84),
                                           ("test_images/tiger_imagenet_292.jpg", 292),
                                           ("test_images/toucan_imagenet_96.jpg", 96),
                                           ("test_images/cello_imagenet_486.jpg", 486)]:

        # read input image
        input_image = Image.open(input_image_name).convert('RGB')
        network_input_shape = (input_image.size[0], input_image.size[1], 3)

        gradients = PytorchRevelio.saliency_map(network=resnet18_net,
                                                input_image=input_image,
                                                class_number=class_number,
                                                img_transformer=img_transformer,
                                                first_layer_name=first_layer_name,
                                                last_layer_name=last_layer_name,
                                                device=device)

        gradients = PytorchRevelio.tensor_outputs_to_image(gradients)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(gradients)
        plt.subplot(1, 2, 2)
        plt.imshow(input_image.resize(size=(224, 224)))

    plt.show()
```

Some of the outputs:
![PytorchRevelio](/readme-images/Figure_2_sm.jpg)





* Saliency map of ResNet-50 with saliency_map_guided:

```python
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from PytorchRevelio import PytorchRevelio

if __name__ == '__main__':

    # load pretrained resnet50
    resnet18_net = torchvision.models.resnet50(pretrained=True)

    # choose GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print('Device: {}'.format(device))

    # put network on device
    resnet18_net.to(device)

    # print name of modules
    for key, value in resnet18_net.named_modules():
        print('+' * 10)
        print(key)
        print('-' * 10)
        print(value)

    # network transformer for input image
    img_transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    network_input_shape = (224, 224, 3)

    # for different convolutional filter and neuron in fully connected layer
    # show representation
    first_layer_name = 'conv1'
    last_layer_name = 'fc'

    for input_image_name, class_number in [("test_images/kit_fox_278_imagenet.jpg", 278),
                                           ("test_images/bald_eagle_22_imagenet.jpg", 22),
                                           ("test_images/peacock_imagenet_84.jpg", 84),
                                           ("test_images/tiger_imagenet_292.jpg", 292),
                                           ("test_images/toucan_imagenet_96.jpg", 96),
                                           ("test_images/cello_imagenet_486.jpg", 486)]:

        # read input image
        input_image = Image.open(input_image_name).convert('RGB')

        gradients = PytorchRevelio.saliency_map_guided(network=resnet18_net,
                                                       input_image=input_image,
                                                       class_number=class_number,
                                                       img_transformer=img_transformer,
                                                       first_layer_name=first_layer_name,
                                                       device=device)

        gradients = PytorchRevelio.tensor_outputs_to_image(gradients)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(gradients)
        plt.subplot(1, 2, 2)
        plt.imshow(input_image.resize(size=(224, 224)))

    plt.show()
```

Some of the outputs:
![PytorchRevelio](/readme-images/Figure_2_sm_guided.jpg)
![PytorchRevelio](/readme-images/Figure_3_sm_guided.jpg)
![PytorchRevelio](/readme-images/Figure_5_sm_guided.jpg)



* Saliency map of ResNet-50 with grad_cam:

```python
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from PytorchRevelio import PytorchRevelio

if __name__ == '__main__':

    # load pretrained resnet50
    vgg11_net = torchvision.models.resnet50(pretrained=True)

    # choose GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    print('Device: {}'.format(device))

    # put network on device
    vgg11_net.to(device)

    # print name of modules
    for key, value in vgg11_net.named_modules():
        print('+' * 10)
        print(key)
        print('-' * 10)
        print(value)

    # network transformer for input image
    img_transformer = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for different convolutional filter and neuron in fully connected layer
    # show representation
    first_layer_name = 'conv1'
    last_layer_name = 'fc'

    for input_image_name, class_number in [("test_images/cheta-zebra-293-340.jpg", 293),
                                           ("test_images/cheta-zebra-293-340.jpg", 340),
                                           ("test_images/vulture-lion-23-291.jpg", 23),
                                           ("test_images/vulture-lion-23-291.jpg", 291),
                                           ("test_images/Gorilla-Lion-Tiger-366-291-292.jpg", 366),
                                           ("test_images/Gorilla-Lion-Tiger-366-291-292.jpg", 291),
                                           ("test_images/Gorilla-Lion-Tiger-366-291-292.jpg", 292),
                                           ("test_images/bull_mastiff_tabbycat_243_281.png", 243),
                                           ("test_images/bull_mastiff_tabbycat_243_281.png", 281),
                                           ("test_images/hen_imagenet_8.jpg", 8),
                                           ]:

        # read input image
        input_image = Image.open(input_image_name).convert('RGB')

        input_image_size = input_image.size

        guided_saliency, sum_last_conv_features_map, grad_cam_gradients = PytorchRevelio.grad_cam(
                             network=vgg11_net,
                             input_image=input_image,
                             input_image_size=input_image_size,
                             class_number=class_number,
                             img_transformer=img_transformer,
                             first_layer_name=first_layer_name,
                             selected_conv_layer_name="layer4.2.conv3",
                             device=device)

        #
        grad_cam_gradients = PytorchRevelio.tensor_outputs_to_image(grad_cam_gradients)

        #
        sum_last_conv_features_map = PytorchRevelio.tensor_outputs_to_image(sum_last_conv_features_map)

        #
        guided_saliency = PytorchRevelio.tensor_outputs_to_image(guided_saliency)

        plt.figure()
        ax = plt.subplot(2, 2, 1)
        plt.imshow(input_image)
        ax.set_title("Input Image")
        ax = plt.subplot(2, 2, 2)
        plt.imshow(guided_saliency)
        ax.set_title("Guided Saliency")
        ax = plt.subplot(2, 2, 3)
        plt.imshow(sum_last_conv_features_map)
        ax.set_title("Grad Cam")
        ax = plt.subplot(2, 2, 4)
        plt.imshow(grad_cam_gradients)
        ax.set_title("Guided Grad Cam")

    plt.show()
```

Some of the outputs:
![PytorchRevelio](/readme-images/Figure_3_sm_ggc.jpg)
![PytorchRevelio](/readme-images/Figure_4_sm_ggc.jpg)
![PytorchRevelio](/readme-images/Figure_5_sm_ggc.jpg)
![PytorchRevelio](/readme-images/Figure_6_sm_ggc.jpg)
![PytorchRevelio](/readme-images/Figure_8_sm_ggc.jpg)
![PytorchRevelio](/readme-images/Figure_9_sm_ggc.jpg)
![PytorchRevelio](/readme-images/Figure_10_sm_ggc.jpg)

## What is the next step?
There are a lot of ways to undestand what a neural network sees in images. Therefore, we will add more methods to this repositories in future.

## What about contributing?
There are a lot of different methods for opening the black box of deep neural networks. Please add new methods to the repository or fix the existing mistakes. 
