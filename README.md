# PytorchRevelio Introduction

PytorchRevelio is a collection of classes and functions that allow you to investigate MLP and convolutional networks written in Pytorch. These classes and functions enable you to visualize features that neurons and filters have learned or illustrate a saliency map for a given image. Even a Deep-Dream function is provided to have more fun. You can calculate feature visualization and saliency map with several different methods. Although Some of these methods are better than others, we included weaker methods for educational purposes.

## Where can we use it?
With PytorchRevelio you can investigate MLP and Convolutional neural networks that are written in Pytorch. There is no matter that you have writtern the network or it is a Pytorch build-in neural network such as VGG-19. 


## How to use it in our code?
Download These Files and put them in your code directory:
  * PytorchRevelio.py
  * utilities_PytorchRevelio.py
  * imagenet_labels Folder

In the following we introduce provided methods and show usecase of them with different examples.  

## What sould we pay attention to during using these methods?
Visualizing learned feature maps and illustrating saliency maps are highly sensitive to hyper-parameters. Therefore, make sure to choose a good set of hyper-parameters to obtain good results. 
