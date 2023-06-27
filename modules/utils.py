import torch.nn as tnn


class Sequential(tnn.Sequential):
    def forward(self, *x, **kwargs):
        for i, module in enumerate(self):
            if type(x) == tuple:
                if i == 0:
                    x = module(*x, **kwargs)
                else:
                    x = module(*x)
            else:
                x = module(x)
        return x


class Residual(tnn.Module):
    def __init__(self, module: tnn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) + args[0]


def init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, tnn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, tnn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)
