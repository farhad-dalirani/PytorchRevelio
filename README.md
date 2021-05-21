![PytorchRevelio](/readme-images/PytorchRevelio.jpg)


# Introduction of PytorchRevelio 

PytorchRevelio is a collection of classes and functions that allow you to investigate MLP and convolutional networks written in Pytorch. These classes and functions enable you to visualize features that neurons and filters have learned or illustrate a saliency map for a given image. Even a Deep-Dream function is provided to have more fun. You can calculate feature visualization and saliency map with several different methods. Although Some of these methods are better than others, we included weaker methods for educational purposes.

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




