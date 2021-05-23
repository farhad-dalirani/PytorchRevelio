import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2 as cv
from utilities_PytorchRevelio import *


class PytorchRevelio:

    @staticmethod
    def layers_name_type(network):
        """
        it receives a network and outputs submodules and layers' name and type.
        :param network:
        :return: [(name, value), (name, value), ...]
        """
        name_type_pair = [(key, value) for key, value in network.named_modules()]
        return name_type_pair

    @staticmethod
    def return_module_by_name(network, module_name):
        """
        It gets a network and name of a layer. Then, it returns layers.
        :param network: a network
        :param module_name: name of a layer
        :return: layer
        """
        if module_name == '' or module_name == " ":
            return None

        modules_name = module_name.split('.')

        module = network._modules[modules_name[0]]
        for i in range(1, len(modules_name)):
            module = module._modules[modules_name[i]]

        return module

    @staticmethod
    def tensor_outputs_to_image(input_tensor):
        """
        Outputs of this class' methods usually are gradients, saliency maps, etc.
        This method gets tensor output of one of these methods, and converts it to an image.
        :param input_tensor: input tensor
        :return: PIL Image
        """

        input_tensor = input_tensor.squeeze(0).detach().cpu()
        input_tensor = normalize_for_display(img=input_tensor)
        output_image = transforms.ToPILImage()(input_tensor).convert("RGB")
        return output_image

    @staticmethod
    def activation_maximization(network, img_transformer, in_img_size,
                                first_layer_name, layer_name, filter_or_neuron_index,
                                num_iter, lr, device):
        """
        This method finds a representation for a given filter/neuron by using activation maximization method
        that can be find in:

        @article{erhan2009visualizing,
          title={Visualizing higher-layer features of a deep network},
          author={Erhan, Dumitru and Bengio, Yoshua and Courville, Aaron and Vincent, Pascal},
          journal={University of Montreal},
          volume={1341},
          number={3},
          pages={1},
          year={2009}
          }

        :param network: input network
        :param img_transformer: pytorch input transformer for the network
        :param in_img_size: size of input image to the network.
        :param first_layer_name: name of networks' first layer name. If you don't know the name you can use
                                 "layers_name_type" function to find it
        :param layer_name: the name of the layer that contains the filter/neuron that you want find a representation
                           for it
        :param filter_or_neuron_index: index of the filter/neuron among all of filters/neurons in
                                       the layer that you want to obtain a representation for it. it starts from zero
        :param num_iter: number of times that gradient ascent is perfomed to obtain a representation for given
                         filter/neuron.
        :param lr:  Learning Rate
        :param device: name of device. For example it can be 'cpu', 'cuda:0'
        :return: representation for selected filter/neuron. Use tensor_outputs_to_image function for visualization
        """
        # put network in train mode
        network.train()

        # forward hook
        f_hook = LayerForwardHook()

        # backward hook
        b_hook = LayerBackwardHook()

        # layer
        layer = PytorchRevelio.return_module_by_name(network=network, module_name=layer_name)

        if not(isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
            raise ValueError('Layer should be convolutional of fc.')

        # add forward hooks
        layer.register_forward_hook(f_hook)

        first_layer = PytorchRevelio.return_module_by_name(network=network, module_name=first_layer_name)

        # add backward hooks
        first_layer.register_backward_hook(b_hook)

        # initialize input image with uniform noise
        # for visualization, input image is not needed and should be None
        input_img = np.uint8(np.random.uniform(140, 170, in_img_size))

        # to pil image
        input_img = F.to_pil_image(input_img)

        # transform input image
        input_img = img_transformer(input_img)

        # in batch form
        input_img = input_img.unsqueeze(0)

        # to device
        input_img = input_img.to(device)

        # enable grad
        input_img.requires_grad = True

        # for number of iterations
        for i_iter in range(0, num_iter):

            # zero gradients
            network.zero_grad()

            # feedforward
            network(input_img)

            if isinstance(layer, nn.Conv2d):
                # calculate gradients with respect of output of a specific filter
                torch.mean(f_hook.activations[:, filter_or_neuron_index, :, :]).backward()
            elif isinstance(layer, nn.Linear):
                # calculate gradients with respect of output of a specific neuron
                f_hook.activations[:, filter_or_neuron_index].backward()

            if isinstance(first_layer, nn.Conv2d):

                # normalize gradients
                b_hook.gradients_in[0] /= torch.sqrt(torch.mean(
                    torch.mul(b_hook.gradients_in[0], b_hook.gradients_in[0]))) + 0.00001

                # update image
                input_img = input_img + b_hook.gradients_in[0] * lr
            elif isinstance(first_layer, nn.Linear):

                # normalize gradients
                b_hook.gradients_in[1] /= torch.sqrt(torch.mean(
                    torch.mul(b_hook.gradients_in[1], b_hook.gradients_in[1]))) + 0.00001

                # update image
                input_img = input_img + torch.reshape(b_hook.gradients_in[1],
                                                      shape=(1, in_img_size[2], in_img_size[1], in_img_size[0])) * lr

        return input_img

    @staticmethod
    def activation_maximization_with_gaussian_blurring(network, img_transformer, in_img_size,
                                                       first_layer_name, layer_name,
                                                       filter_or_neuron_index,
                                                       start_sigma, end_sigma,
                                                       num_iter, lr, device):
        """
            This method finds a representation for a given filter/neuron by using activation maximization with
            gaussian blurring method that can be find in:

                https://www.auduno.com/2015/07/29/visualizing-googlenet-classes/

            :param network: input network
            :param img_transformer: pytorch input transformer for the network
            :param in_img_size: size of input image to the network.
            :param first_layer_name: name of networks' first layer name. If you don't know the name you can use
                                         "layers_name_type" function to find it
            :param layer_name: the name of the layer that contains the filter/neuron that you want find a representation
                                   for it
            :param filter_or_neuron_index: index of the filter/neuron among all of filters/neurons in
                                               the layer that you want to obtain a representation for it. it starts from zero
            :param start_sigma: standard deviation of gaussian filter at the first iteration
            :param end_sigma: standard deviation of gaussian filter at the last iteration
            :param num_iter: number of times that gradient ascent is perfomed to obtain a representation for given
                                 filter/neuron.
            :param lr:  Learning Rate
            :param device: name of device. For example it can be 'cpu', 'cuda:0'
            :return: representation for selected filter/neuron. Use tensor_outputs_to_image function for visualization
        """

        # put network in your train mode
        network.train()

        # forward hook
        f_hook = LayerForwardHook()

        # backward hook
        b_hook = LayerBackwardHook()

        # layer
        layer = PytorchRevelio.return_module_by_name(network=network, module_name=layer_name)

        if not(isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
            raise ValueError('Layer should be convolutional of fc.')

        # add forward hooks
        layer.register_forward_hook(f_hook)

        first_layer = PytorchRevelio.return_module_by_name(network=network, module_name=first_layer_name)

        # add backward hooks
        first_layer.register_backward_hook(b_hook)

        # initialize input image with uniform noise
        # for visualization, input image is not needed and should be None
        input_img = np.uint8(np.random.uniform(140, 170, in_img_size))

        # to pil image
        input_img = F.to_pil_image(input_img)

        # transform input image
        input_img = img_transformer(input_img)

        # in batch form
        input_img = input_img.unsqueeze(0)

        # to device
        input_img = input_img.to(device)

        # enable grad
        input_img.requires_grad = True

        # sigma for each iteration
        sigmas = np.linspace(start=start_sigma, stop=end_sigma, num=num_iter)

        # for number of iterations
        for i_iter in range(0, num_iter):

            # zero gradients
            network.zero_grad()

            # feedforward
            network(input_img)

            if isinstance(layer, nn.Conv2d):
                # calculate gradients with respect of output of a specific filter
                torch.mean(f_hook.activations[:, filter_or_neuron_index, :, :]).backward()
            elif isinstance(layer, nn.Linear):
                # calculate gradients with respect of output of a specific neuron
                f_hook.activations[:, filter_or_neuron_index].backward()

            if isinstance(first_layer, nn.Conv2d):

                # normalize gradients
                b_hook.gradients_in[0] /= torch.sqrt(torch.mean(
                    torch.mul(b_hook.gradients_in[0], b_hook.gradients_in[0]))) + 0.00001

                # update image
                input_img = input_img + b_hook.gradients_in[0] * lr
            elif isinstance(first_layer, nn.Linear):

                # normalize gradients
                b_hook.gradients_in[1] /= torch.sqrt(torch.mean(
                    torch.mul(b_hook.gradients_in[1], b_hook.gradients_in[1]))) + 0.00001

                # update image
                input_img = input_img + torch.reshape(b_hook.gradients_in[1],
                                                      shape=(1, in_img_size[2], in_img_size[1], in_img_size[0])) * lr

            # sigma for i'th iteration
            sigma = sigmas[i_iter]

            # gaussian blurring image
            smoothing = GaussianSmoothing(channels=input_img.shape[1], kernel_size=3, sigma=sigma, dim=2, device=device)
            padder = nn.ZeroPad2d(1)
            input_img = padder(input_img)
            input_img = smoothing(input_img.detach().clone())
            input_img.requires_grad = True

        return input_img

    @staticmethod
    def activation_maximization_with_bilateral_blurring(network, img_transformer, in_img_size,
                                                        first_layer_name, layer_name,
                                                        filter_or_neuron_index,
                                                        start_sigma_color,
                                                        end_sigma_color,
                                                        start_sigma_space,
                                                        end_sigma_space,
                                                        kernel_size,
                                                        num_iter,
                                                        lr,
                                                        device):
        """
            This method finds a representation for a given filter/neuron by using activation maximization with
            bilateral blurring method that can be find in:

                https://mtyka.github.io/deepdream/2016/02/05/bilateral-class-vis.html

            :param network: input network
            :param img_transformer: pytorch input transformer for the network
            :param in_img_size: size of input image to the network.
            :param first_layer_name: name of networks' first layer name. If you don't know the name you can use
                                         "layers_name_type" function to find it
            :param layer_name: the name of the layer that contains the filter/neuron that you want find a representation
                                   for it
            :param filter_or_neuron_index: index of the filter/neuron among all of filters/neurons in
                                               the layer that you want to obtain a representation for it. it starts from zero
            :param start_sigma_color: color standard deviation of bilateral filter at the first iteration
            :param end_sigma_color: color standard deviation of bilateral filter at the last iteration
            :param start_sigma_space: space standard deviation of bilateral filter at the first iteration
            :param end_sigma_space: space standard deviation of bilateral filter at the last iteration
            :param kernel_size: kernel size of bilateral filter
            :param num_iter: number of times that gradient ascent is performed to obtain a representation for given
                                 filter/neuron.
            :param lr:  Learning Rate
            :param device: name of device. For example it can be 'cpu', 'cuda:0'
            :return: representation for selected filter/neuron. Use tensor_outputs_to_image function for visualization
        """

        # put model in train mode
        network.train()

        # forward hook
        f_hook = LayerForwardHook()

        # backward hook
        b_hook = LayerBackwardHook()

        # layer
        layer = PytorchRevelio.return_module_by_name(network=network, module_name=layer_name)

        if not(isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
            raise ValueError('Layer should be convolutional of fc.')

        # add forward hooks
        layer.register_forward_hook(f_hook)

        first_layer = PytorchRevelio.return_module_by_name(network=network, module_name=first_layer_name)

        # add backward hooks
        first_layer.register_backward_hook(b_hook)

        # initialize input image with uniform noise
        # for visualization, input image is not needed and should be None
        input_img = np.uint8(np.random.uniform(140, 170, in_img_size))

        # to pil image
        input_img = F.to_pil_image(input_img)

        # transform input image
        input_img = img_transformer(input_img)

        # in batch form
        input_img = input_img.unsqueeze(0)

        # to device
        input_img = input_img.to(device)

        # enable grad
        input_img.requires_grad = True

        # sigma for each iteration
        sigmas_color = np.linspace(start=start_sigma_color, stop=end_sigma_color, num=num_iter)
        sigmas_space = np.linspace(start=start_sigma_space, stop=end_sigma_space, num=num_iter)

        # for number of iterations
        for i_iter in range(0, num_iter):

            # zero gradients
            network.zero_grad()

            # feedforward
            network(input_img)

            if isinstance(layer, nn.Conv2d):
                # calculate gradients with respect of output of a specific filter
                torch.mean(f_hook.activations[:, filter_or_neuron_index, :, :]).backward()
            elif isinstance(layer, nn.Linear):
                # calculate gradients with respect of output of a specific neuron
                f_hook.activations[:, filter_or_neuron_index].backward()

            if isinstance(first_layer, nn.Conv2d):

                # normalize gradients
                b_hook.gradients_in[0] /= torch.sqrt(torch.mean(
                    torch.mul(b_hook.gradients_in[0], b_hook.gradients_in[0]))) + 0.00001

                # update image
                input_img = input_img + b_hook.gradients_in[0] * lr
            elif isinstance(first_layer, nn.Linear):

                # normalize gradients
                b_hook.gradients_in[1] /= torch.sqrt(torch.mean(
                    torch.mul(b_hook.gradients_in[1], b_hook.gradients_in[1]))) + 0.00001

                # update image
                input_img = input_img + torch.reshape(b_hook.gradients_in[1],
                                                      shape=(1, in_img_size[2], in_img_size[1], in_img_size[0])) * lr

            # sigma for i' th iteration
            sigma_color = sigmas_color[i_iter]
            sigma_space = sigmas_space[i_iter]

            # to numpy array
            input_img = input_img.squeeze(0)
            input_img = input_img.detach().cpu().numpy()

            # convert from C*W*H to W*H*C
            input_img = np.transpose(input_img, (1, 2, 0))

            # range of pixels to [0, 1]
            input_img_min = np.min(input_img)
            input_img_max = np.max(input_img)
            input_img = (input_img - input_img_min) / (input_img_max - input_img_min)

            # bilateral smoothing
            input_img = cv.bilateralFilter(input_img * 255.0, kernel_size, sigma_color, sigma_space)

            # convert input to its original range
            input_img /= 255.0
            input_img = (input_img * (input_img_max - input_img_min)) + input_img_min

            # convert from W*H*C to C*W*H
            input_img = np.transpose(input_img, (2, 0, 1))

            # numpy array to tensor
            input_img = torch.tensor(input_img, dtype=torch.float32)
            input_img = input_img.unsqueeze(0)
            input_img = input_img.to(device)

            input_img.requires_grad = True

        return input_img

    @staticmethod
    def saliency_map(network, input_image, class_number, img_transformer,
                     first_layer_name, last_layer_name, device):
        """
        This function calculates saliency map of a given image with respect to a given class.
        It is obtained according to this paper.

        Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
        https://arxiv.org/abs/1312.6034

        :param network: a pytorch network
        :param input_image: a PIL image
        :param class_number: INT, target class that we want to calculate saliency map with respect to it
        :param img_transformer: pytorch input transformer for the network
        :param first_layer_name: name of networks' first layer name. If you don't know the name you can use
                                 "layers_name_type" function to find it
        :param last_layer_name: the name of the layer that contains the filter/neuron that you want find a
                                representation for it
        :param device: name of device. For example it can be 'cpu', 'cuda:0'
        :return: tensor, saliency map
        """

        # put model in evaluation mode
        network.eval()

        # forward hook
        f_hook_last_layer = LayerForwardHook()

        # backward hook
        b_hook_first_layer = LayerBackwardHook()

        # last layer
        last_layer = PytorchRevelio.return_module_by_name(network=network, module_name=last_layer_name)

        if not(isinstance(last_layer, nn.Conv2d) or isinstance(last_layer, nn.Linear)):
            raise ValueError('Last layer should be convolutional of fc.')

        # add forward hooks for last layer
        last_layer.register_forward_hook(f_hook_last_layer)

        # first layer
        first_layer = PytorchRevelio.return_module_by_name(network=network, module_name=first_layer_name)

        # add backward hooks
        first_layer.register_backward_hook(b_hook_first_layer)

        # transform input image
        input_img = img_transformer(input_image.copy())

        # in batch form
        input_img = input_img.unsqueeze(0)

        # to device
        input_img = input_img.to(device)

        # enable grad
        input_img.requires_grad = True

        # zero gradients
        network.zero_grad()

        # feedforward
        output = network(input_img)

        # 1-top prediction of network
        top_predicted_class = output.topk(1, dim=1)
        top_predicted_class = list(top_predicted_class)[1].item()

        if top_predicted_class != class_number:
            print('Warning! The class that you gave,{}, is not same with the prediction,{}, of the network.'.format(
                top_predicted_class, class_number))

        #
        target = torch.zeros(size=output.shape, dtype=torch.float32)
        target = target.to(device)
        target[0][class_number] = 1
        #

        output.backward(gradient=target)

        gradients = None
        if isinstance(first_layer, nn.Conv2d):
            gradients = b_hook_first_layer.gradients_in[0].detach().cpu()
        elif isinstance(first_layer, nn.Linear):
            gradients = b_hook_first_layer.gradients_in[1].detach().cpu()

        if input_image.mode == "RGB":
            # for a pixel just keep biggest value among all channels
            gradients = gradients.max(dim=1, keepdim=True)[0]
            gradients = torch.cat((gradients, gradients * 0, gradients * 0), dim=1)

        return gradients

    @staticmethod
    def saliency_map_guided(network, input_image, class_number, img_transformer, first_layer_name,
                            device):
        """
        This function calculates "guided saliency map" of a given image with respect to a given class.
        It is obtained according to this paper.

            Striving for Simplicity: The All Convolutional Net
            https://arxiv.org/abs/1412.6806

        :param network: a pytorch network
        :param input_image: a PIL image
        :param class_number: INT, target class that we want to calculate saliency map with respect to it
        :param img_transformer: pytorch input transformer for the network
        :param first_layer_name: name of networks' first layer name. If you don't know the name you can use
                                         "layers_name_type" function to find it
        :param device: name of device. For example it can be 'cpu', 'cuda:0'
        :return: tensor, Guided Saliency Map
        """

        # put model in evaluation mode
        network.eval()

        # forward hook
        f_hook = RELUsForwardHook()

        # backward hook
        b_hook = RELUsBackwardHook(network_forward_hook=f_hook)

        # backward hook
        b_hook_first_layer = LayerBackwardHook()

        # first layer
        first_layer = PytorchRevelio.return_module_by_name(network=network, module_name=first_layer_name)

        # add backward hooks
        first_layer.register_backward_hook(b_hook_first_layer)

        for _, module in network.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(f_hook)
                module.register_backward_hook(b_hook)

        # transform input image
        input_img = img_transformer(input_image.copy())

        # in batch form
        input_img = input_img.unsqueeze(0)

        # to device
        input_img = input_img.to(device)

        # enable grad
        input_img.requires_grad = True

        # zero gradients
        network.zero_grad()

        # feedforward
        output = network(input_img)

        # 1-top prediction of network
        top_predicted_class = output.topk(1, dim=1)
        if list(top_predicted_class)[1].item() != class_number:
            print('Warning! The class that you gave,{}, is not same with the prediction,{}, of the network.'.format(
                top_predicted_class, class_number))

        target = torch.zeros(size=output.shape, dtype=torch.float32)
        target = target.to(device)
        target[0][class_number] = 1

        output.backward(gradient=target)

        gradients = None
        if isinstance(first_layer, nn.Conv2d):
            gradients = b_hook_first_layer.gradients_in[0].detach().cpu()
        elif isinstance(first_layer, nn.Linear):
            gradients = b_hook_first_layer.gradients_in[1].detach().cpu()

        if input_image.mode == "RGB":
            # for a pixel just keep biggest value among all channels
            gradients = gradients.max(dim=1, keepdim=True)[0]
            gradients = torch.cat((gradients, gradients * 0, gradients * 0), dim=1)

        return gradients

    @staticmethod
    def deep_dream(network, img_transformer, in_img_size,
                   first_layer_name, layer_name,
                   num_iter, lr, device, input_img):
        """
        Deep Dream

        https://en.wikipedia.org/wiki/DeepDream
        https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

        :param network: a pytorch network
        :param img_transformer: pytorch input transformer for the network
        :param in_img_size: size of input image: (W, H, C)
        :param first_layer_name: name of networks' first layer name. If you don't know the name you can use
                                         "layers_name_type" function to find it
        :param layer_name: target layer in network that we want maximize its feature in input image
        :param num_iter: number of iteration
        :param lr: learning rate
        :param device: name of device. For example it can be 'cpu', 'cuda:0'
        :param input_img: input image, PIL format
        :return:
        """
        # put model in evaluation mode
        network.train()

        # forward hook
        f_hook = LayerForwardHook()

        # backward hook
        b_hook = LayerBackwardHook()

        # layer
        layer = PytorchRevelio.return_module_by_name(network=network, module_name=layer_name)

        if not(isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
            raise ValueError('Layer should be convolutional of fc.')

        # add forward hooks
        layer.register_forward_hook(f_hook)

        first_layer = PytorchRevelio.return_module_by_name(network=network, module_name=first_layer_name)

        # add backward hooks
        first_layer.register_backward_hook(b_hook)

        if input_img is None:
            # initialize input image with uniform noise
            # for visualization, input image is not needed and should be None
            input_img = np.uint8(np.random.uniform(140, 170, in_img_size))
        else:
            # for deep dream input image is required
            pass

        # to pil image
        input_img = F.to_pil_image(input_img)

        # transform input image
        input_img = img_transformer(input_img)

        # in batch form
        input_img = input_img.unsqueeze(0)

        # to device
        input_img = input_img.to(device)

        # enable grad
        input_img.requires_grad = True

        # for number of iterations
        for i_iter in range(0, num_iter):

            # zero gradients
            network.zero_grad()

            # feedforward
            network(input_img)

            if isinstance(layer, nn.Conv2d):
                # calculate gradients with respect of output of a specific filter
                torch.mean(f_hook.activations).backward()
            elif isinstance(layer, nn.Linear):
                # calculate gradients with respect of output of a specific neuron
                torch.mean(f_hook.activations).backward()

            if isinstance(first_layer, nn.Conv2d):

                # normalize gradients
                b_hook.gradients_in[0] /= torch.sqrt(torch.mean(
                    torch.mul(b_hook.gradients_in[0], b_hook.gradients_in[0]))) + 0.00001

                # update image
                input_img = input_img + b_hook.gradients_in[0] * lr
            elif isinstance(first_layer, nn.Linear):

                # normalize gradients
                b_hook.gradients_in[1] /= torch.sqrt(torch.mean(
                    torch.mul(b_hook.gradients_in[1], b_hook.gradients_in[1]))) + 0.00001

                # update image
                input_img = input_img + torch.reshape(b_hook.gradients_in[1],
                                                      shape=(1, in_img_size[2], in_img_size[1], in_img_size[0])) * lr

        return input_img

    @staticmethod
    def grad_cam(network, input_image, input_image_size, class_number, img_transformer,
                 first_layer_name, selected_conv_layer_name, device):

        """
                This function calculates saliency map with Grad-Cam Method of a given image with respect to a
                given class.
                It is obtained according to this paper.

                Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
                    https://arxiv.org/abs/1610.02391

                :param network: a pytorch network
                :param input_image: a PIL image
                :param input_image_size: (W, H)
                :param class_number: INT, target class that we want to calculate saliency map with respect to it
                :param img_transformer: pytorch input transformer for the network
                :param first_layer_name: name of networks' first layer name. If you don't know the name you can use
                                                 "layers_name_type" function to find it
                :param selected_conv_layer_name: name of a conv layer that should be selected according to the paper
                                                    for the best result, usually last conv layer works best
                :param device: name of device. For example it can be 'cpu', 'cuda:0'
                :return: tensor, Guided Saliency Map
        """

        # calculate guided saliency map
        guided_saliency = PytorchRevelio.saliency_map_guided(network=network,
                                                             input_image=input_image,
                                                             class_number=class_number,
                                                             img_transformer=img_transformer,
                                                             first_layer_name=first_layer_name,
                                                             device=device)

        guided_saliency = guided_saliency.squeeze(0).detach().cpu()

        # last convolutional layer
        last_conv_layer = PytorchRevelio.return_module_by_name(network=network, module_name=selected_conv_layer_name)

        network.zero_grad()

        # put network in eval mode
        network.eval()

        # forward hook
        f_hook = LayerForwardHook()

        # add forward hook to last convolutional layer
        last_conv_layer.register_forward_hook(f_hook)

        # backward hook
        b_hook = LayerBackwardHook()

        # add backward hook to last convolutional layer
        last_conv_layer.register_backward_hook(b_hook)

        # transform input image
        input_img = img_transformer(input_image.copy())

        # in batch form
        input_img = input_img.unsqueeze(0)

        # to device
        input_img = input_img.to(device)

        # feedforward
        output = network(input_img)

        target = torch.zeros(size=output.shape, dtype=torch.float32)
        target = target.to(device)
        target[0][class_number] = 1

        output.backward(gradient=target)

        # derivative of the class with respect ro activation of last conv layer
        gradients = b_hook.gradients_out

        # output feature map of last convolutional layer
        activations = f_hook.activations

        sum_last_conv_features_map = torch.zeros(size=(activations.shape[2], activations.shape[3]), dtype=torch.float32)
        sum_last_conv_features_map = sum_last_conv_features_map.to(device)

        # calculate weight of each filter in last conv layer
        for kernel_i in range(0, gradients.shape[1]):
            weight = torch.mean(gradients[0, kernel_i, :, :]).item()
            sum_last_conv_features_map += weight * activations[0, kernel_i, :, :]

        sum_last_conv_features_map[sum_last_conv_features_map < 0] = 0

        min_sum = torch.min(sum_last_conv_features_map)
        max_sum = torch.max(sum_last_conv_features_map)
        sum_last_conv_features_map = (sum_last_conv_features_map-min_sum)/(max_sum-min_sum)

        sum_last_conv_features_map = torch.stack((sum_last_conv_features_map,
                                                  sum_last_conv_features_map * 0,
                                                  sum_last_conv_features_map * 0),
                                                 dim=0)

        # The resize operation on tensor.
        sum_last_conv_features_map = FF.interpolate(sum_last_conv_features_map, size=input_image_size[0])
        sum_last_conv_features_map = sum_last_conv_features_map.permute(0, 2, 1)
        sum_last_conv_features_map = FF.interpolate(sum_last_conv_features_map, size=input_image_size[1])
        sum_last_conv_features_map = sum_last_conv_features_map.permute(0, 2, 1)
        sum_last_conv_features_map = sum_last_conv_features_map.squeeze(0).detach().cpu()

        # element-wise multiply
        grad_cam_gradients = torch.mul(guided_saliency, sum_last_conv_features_map)

        return guided_saliency, sum_last_conv_features_map, grad_cam_gradients

