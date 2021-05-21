import torch
import torch.nn as nn
from torch.nn import functional as FF
import numbers
import math


def normalize_for_display(img, saturation=0.15, brightness=0.5):

    mean, std = img.mean(), img.std()

    if std == 0:
        std += 1e-6

    zero_mean_std_one = img.sub(mean).div(std)
    normalized = zero_mean_std_one.mul(saturation)
    output_img = normalized.add(brightness).clamp(0.0, 1.0)

    return output_img


def imagenet_labels(class_number, length=20):

    file = open("imagenet_labels/imagenet1000_clsidx_to_labels.json", "r")
    lines = file.readlines()

    string = lines[class_number].split('\'')[1]
    if len(string) > length:
        string = string[0:length]

    return string


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing. Filtering is performed seperately for each channel.

    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

        source of this functions:
        discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
    """
    def __init__(self, channels, kernel_size, sigma, dim, device):
        self.device = device

        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = FF.conv1d
        elif dim == 2:
            self.conv = FF.conv2d
        elif dim == 3:
            self.conv = FF.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = input.to(self.device)
        self.weight = self.weight.to(self.device)
        output = self.conv(input, weight=self.weight, groups=self.groups)
        return output


class LayerForwardHook:
    def __init__(self):
        self.activations = None

    def __call__(self, module, module_input, module_output):
        self.activations = module_output

    def remove(self):
        self.activations = None


class LayerBackwardHook:
    def __init__(self):
        self.gradients_in = None
        self.gradients_out = None

    def __call__(self, module, module_gradient_input, module_gradient_output):
        self.gradients_in = list(module_gradient_input)
        self.gradients_out = module_gradient_output[0]

    def remove(self):
        self.gradients_in = None
        self.gradients_out = None


class RELUsForwardHook:
    def __init__(self):
        self.activations = []

    def __call__(self, module, module_input, module_output):
        self.activations.append(module_output)

    def remove(self):
        self.activations = []


class RELUsBackwardHook:
    def __init__(self, network_forward_hook):
        self.gradients = None
        self.network_forward_hook = network_forward_hook

    def __call__(self, module, module_gradient_input, module_gradient_output):
        self.gradients = list(module_gradient_input)

        self.gradients = self.gradients[0]

        self.gradients[self.gradients < 0] = 0

        last_activation = self.network_forward_hook.activations.pop()

        self.gradients = self.gradients.mul(last_activation)

    def remove(self):
        self.gradients = None
