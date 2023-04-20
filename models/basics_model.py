import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##########################################
# Fully connected Layer
##########################################
class FCLayer(nn.Module):
    """Fully connected layer """
    def __init__(self, in_feature, out_feature, 
                        activation = "gelu",
                        is_normalized = True): 
        super().__init__()
        if is_normalized:
            self.LinearBlock = nn.Sequential(
                            nn.Linear(in_feature,out_feature),
                            LayerNorm(out_feature),
                            )
                                        
        else:
            self.LinearBlock = nn.Linear(in_feature,out_feature)
        self.act = set_activ(activation)
    def forward(self, x):
        return self.act(self.LinearBlock(x))
##########################################
# Fully connected Block
##########################################
class FC_nn(nn.Module):
    r"""Simple MLP to code lifting and projection"""
    def __init__(self, sizes = [2, 128, 128, 1], 
                        activation = 'relu',
                        outermost_linear = True, 
                        outermost_norm = True,  
                        drop = 0.):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.net = nn.ModuleList([FCLayer(in_feature= m, out_feature= n, 
                                            activation=activation, 
                                            is_normalized = False)   
                                for m, n in zip(sizes[:-2], sizes[1:-1])
                                ])
        if outermost_linear == True: 
            self.net.append(FCLayer(sizes[-2],sizes[-1], activation = None, 
                                    is_normalized = outermost_norm))
        else: 
            self.net.append(FCLayer(in_feature= sizes[-2], out_feature= sizes[-1], 
                                    activation=activation,
                                    is_normalized = outermost_norm))

    def forward(self,x):
        for module in self.net:
            x = module(x)
            x = self.dropout(x)
        return x

  
###### Inverse Bottleneck ########
class MLP_inv_bottleneck(nn.Module):
  """Inverse Bottleneck MLP"""
  def __init__(self, dim, activation = 'gelu'):
    super().__init__()
    self.nonlinear = set_activ(activation)
    self.L1 = nn.Linear(dim, 4*dim)
    self.L2 = nn.Linear(4*dim, dim)
  def forward(self,x):
    x = self.L1(x)
    x = self.nonlinear(x)
    x = self.L2(x)
    return x

########## Simple MLP ##############
class MLP_join(nn.Module):
  """Simple MLP to code lifting and projection"""
  def __init__(self, sizes = [1, 128, 128, 1], activation = 'gelu', drop = 0.):
    super(MLP_join, self).__init__()
    self.hidden_layer = sizes
    self.nonlinear = set_activ(activation)
    self.dropout = nn.Dropout(drop)
    self.net = nn.ModuleList([nn.Linear(m, n)   
                              for m, n in zip(sizes[:-1], sizes[1:])
                              ])
  def forward(self,x):
    for module in self.net[:-1]:
      x = module(x)
      x = self.nonlinear(x)
      x = self.dropout(x)
    return self.net[-1](x)

########## Layer Normalization ##############
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
###################################################################################### 
# new additions over the main code            
class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

######################################################################################
# Miscellaneous functions
######################################################################################

########## Getting the 2D grid using the batch
def get_grid2D(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


########## Set automatically the activation function for the NN
def set_activ(activation):
    if activation is not None: 
        activation = activation.lower()
    if activation == 'relu':
        nonlinear = F.relu
    elif activation == "leaky_relu": 
        nonlinear = F.leaky_relu
    elif activation == 'tanh':
        nonlinear = F.tanh
    elif activation == 'sine':
        nonlinear= torch.sin
    elif activation == 'gelu':
        nonlinear= F.gelu
    elif activation == 'elu':
        nonlinear = F.elu_
    elif activation == None:
        nonlinear = nn.Identity()
    else:
        raise Exception('The activation is not recognized from the list')
    return nonlinear

    