import torch
import torch.nn as nn
import numpy as np
import math
from torch.ao.quantization import QuantStub, DeQuantStub

class SDFNetworkQAT(nn.Module):
    def __init__(self, d_in=3, d_out=1, d_hidden=512, n_layers=8, skip_in=(4,),
                 geometric_init=True, radius_init=1.0, beta=100):
        super().__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.skip_in = skip_in
        dims = [d_in] + [d_hidden] * n_layers + [d_out]
        self.num_layers = len(dims)
        self.layers = nn.ModuleList()

        for i in range(0, self.num_layers - 1):
            in_dim = dims[i]
            if i + 1 in skip_in:
                out_dim = dims[i + 1] - d_in
            else:
                out_dim = dims[i + 1]

            lin = nn.Linear(in_dim, out_dim)

            if geometric_init:
                if i == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            self.layers.append(lin)

        self.activation = nn.Softplus(beta=beta)

    def forward(self, input_coords):
        x = self.quant(input_coords)

        for i, layer in enumerate(self.layers):
            if i in self.skip_in:
                x = torch.cat([x, self.quant(input_coords)], dim=-1) / math.sqrt(2)

            x = layer(x)

            if i < len(self.layers) - 1:
                x = self.activation(x)

        x = self.dequant(x)
        return x
