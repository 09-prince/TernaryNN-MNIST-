import torch
import torch.nn as nn
import torch.nn.functional as F

class TernaryLinear(nn.Module):
    """
    A custom linear (fully connected) layer with ternary weight quantization.

    The weights are quantized to one of {-1, 0, +1} during the forward pass
    using a thresholding approach based on a user-defined ratio. After quantization,
    the ternary weights are scaled by the average absolute weight value to preserve
    magnitude information.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Default is True.
        ratio (float): Threshold ratio to determine ternary weights. 
                       Smaller ratio keeps more weights at 0. Default is 0.05.

    Forward:
        Applies ternary quantization to weights and computes a linear transformation.
    """
    def __init__(self, in_features, out_features, bias=True, ratio=0.05):
        super().__init__()
        self.ratio = ratio

        # Initialize real-valued weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')

        # Optional bias term
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # Compute scaling factor (mean absolute weight value)
        scale = self.weight.abs().mean()

        # Compute ternary threshold
        thr = scale * self.ratio

        # Ternarize weights using threshold: {-1, 0, +1}
        w_tern = torch.where(self.weight >  thr,  1.0,
                  torch.where(self.weight < -thr, -1.0, 0.0)) * scale

        # Apply linear transformation with ternary weights
        return F.linear(x, w_tern, self.bias)



class TernaryConv2d(nn.Module):
    """
    A 2D convolutional layer with ternary weight quantization.

    The weights are quantized to one of {-1, 0, +1} during the forward pass
    using a threshold-based ternarization method. This ternary representation
    helps reduce memory usage and computation while retaining performance.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel (assumes square).
        stride (int): Stride of the convolution. Default is 1.
        padding (int): Zero-padding added to both sides of the input. Default is 0.
        bias (bool): Whether to include a learnable bias. Default is True.
        ratio (float): Thresholding ratio for ternarization. Default is 0.05.

    Forward:
        Applies convolution using ternarized weights and optional bias.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, ratio=0.05):
        super().__init__()
        self.ratio = ratio

        # Initialize real-valued convolution weights
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity='conv2d')

        # Optional learnable bias
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

        # Store convolution stride and padding
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Compute scale as mean absolute value of weights
        scale = self.weight.abs().mean()

        # Compute ternary threshold
        thr = scale * self.ratio

        # Ternarize weights to {-1, 0, +1}, then scale them
        w_tern = torch.where(self.weight >  thr,  1.0,
                  torch.where(self.weight < -thr, -1.0, 0.0)) * scale

        # Apply 2D convolution with ternarized weights
        return F.conv2d(x, w_tern, self.bias, stride=self.stride, padding=self.padding)
