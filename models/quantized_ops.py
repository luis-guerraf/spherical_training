import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter


class Binarize_W(Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Binarize_project_A(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # import pdb; pdb.set_trace()

        # Projection onto tangent space
        x, = ctx.saved_tensors
        batch_size = x.size(0)

        x = x.data.sign().view(batch_size, -1).unsqueeze(-1)
        grad = grad_output.data.view(batch_size, -1).unsqueeze(-1)
        x = x.permute(0, 2, 1)

        temp = torch.bmm(x, grad).squeeze(-1)
        x = x.squeeze(1)
        grad = grad.squeeze(-1)

        P_grad = grad - x * temp
        P_grad = P_grad.view(grad_output.shape)

        P_grad[input > 1] = 0
        P_grad[input < -1] = 0

        return P_grad


class Binarize_A(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0

        return grad_output


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        # weight = Binarize_W.apply(self.weight)
        # input = Binarize_project_A.apply(input)
        output = F.conv2d(input, self.weight, self.bias,
                          self.stride, self.padding, self.dilation, self.groups)
        return output

    def init_scalings(self):
        tensor = self.weight
        n = tensor[0, 0].nelement()
        m = tensor.norm(1, 3).sum(2).div(n)
        self.scale.data = m


class QuantizedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True):
        super(QuantizedLinear, self).__init__(in_channels, out_channels, bias=bias)

    def forward(self, input):
        weight = Binarize_W.apply(self.weight)

        output = F.linear(input, weight, self.bias)
        return output
