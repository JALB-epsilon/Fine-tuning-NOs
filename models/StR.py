import torch
import pytorch_lightning as pl
from utilities import LpLoss
from torch import optim, nn
from .basics_model import set_activ, LayerNorm
from collections import OrderedDict
from timm.models.layers import trunc_normal_
from .sFNO_epsilon_v2_updated import sFNO_epsilon_v2_updated

class FCNN(nn.Module):
    """
    Implementation of FCNN
    """
    def __init__(self, in_features, 
                hidden_features=None, 
                out_features=None,  
                activation = "leaky_relu"):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = set_activ(activation)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


###################################
#Functional Batch Linear Layer
##################################
class FCBatchLinear(nn.Module):
    '''A linear meta-BatchLayer that can deal with 
    batched weight matrices and biases.'''
    def __init__(self, in_features = None, 
                        out_features = None):
        super().__init__()

    def forward(self, x, weight, bias):
        assert weight.shape[0] == bias.shape[0]
        assert weight.shape[-1] == x.shape[-1]
        assert weight.shape[-2] == bias.shape[-1]
        output = torch.einsum('bcoi, bcsi->bcso',  [weight, x])
        output += bias.unsqueeze(2)
        return output
  

##################################
#Linear Layer
##################################
class MetaFCLayer(nn.Module):
    '''A linear meta-BatchLayer.'''
    def __init__(self, in_features=None, 
                        out_features=None,
                        activation=None, 
                        is_last = False, 
                        is_conv = False,
                        drop=0.):
        super().__init__()
        self.nonlinear = set_activ(activation) if not is_last else nn.Identity()
        self.batch_linear = FCBatchLinear(in_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()
        self.is_conv = is_conv
        if is_conv:
            self.conv_mix = nn.Conv2d(2, 2,kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, weight, bias):
        x = self.batch_linear(x, weight, bias)
        x =self.nonlinear(x)
        if self.is_conv:
            x = self.conv_mix(x)
        x = self.drop(x)
        return x
    
##################################
#MetaSrcRcv
##################################
class MetaSrcRcv(nn.Module):
    def __init__(self, 
                dim_src_position =1, 
                numb_rcv=128,
                Lifting=None, 
                activation = "leaky_relu",
                mlp_ratio=4,
                use_layer_scale=True,
                layer_scale_init_value=1e-5, 
                norm_layer=nn.LayerNorm, 
                drop= 0.
                  ):
      super().__init__()
      self.norm1 = norm_layer(numb_rcv)
      self.norm2 = norm_layer(numb_rcv)
      self.dim_position = dim_src_position
      self.numb_rcv = numb_rcv
      if Lifting is None:
        #the FCNN generates the correct feature space (numb_rcv) we can assume is given and fixed.
        #while the conv layer is used to generate the correct spatial dimension for the 
        #real an imaginary part of the field seeing as channels.
        mlp_hidden_features = int(numb_rcv * mlp_ratio)
        self.Lifting = nn.Sequential(
                           FCNN(in_features=dim_src_position, 
                           hidden_features=mlp_hidden_features,
                           out_features=numb_rcv,
                           activation = activation), 
                           norm_layer(numb_rcv),
                           nn.Conv2d(1, 2, kernel_size=1, stride=1,  
                                     padding=0, bias=True)) 

      else:
        self.Lifting = Lifting


      self.metanet1 = MetaFCLayer(numb_rcv,
                                    numb_rcv, 
                                    activation = activation, 
                                    is_last = False, 
                                    is_conv = False,
                                    drop=drop)
      
                              
      self.metanet2 = MetaFCLayer(numb_rcv,
                                    numb_rcv, 
                                    activation = activation, 
                                    is_last = False,
                                    is_conv = False,
                                    drop=drop)
      #added 
      self.last_layer = nn.Linear(numb_rcv, numb_rcv)
      
      
      self.use_layer_scale = use_layer_scale
      if use_layer_scale: 
            self.layer_scale1 = nn.Parameter(torch.ones((2))*layer_scale_init_value, requires_grad=True)
            self.layer_scale2 = nn.Parameter(torch.ones((2))*layer_scale_init_value, requires_grad=True)

      self.params = self.set_params()

    def set_params(self):
      """Set the parameters of the MetaSrcRcv layer."""
      layers_shapes = OrderedDict()
      for l in range(2): 
        layers_shapes[f"{l}.weight"] = torch.Size([2, self.numb_rcv, self.numb_rcv])
        layers_shapes[f"{l}.bias"] = torch.Size([2, self.numb_rcv])
      return layers_shapes
    
    def forward(self, x, params):
      #assert the tensor is four dimensional
        assert len(x.shape) == 4, f"The input tensor must be four dimensional (1,1,src,1) is {x.shape}"
        x = self.Lifting(x)
        if self.use_layer_scale: 
            x =x + self.layer_scale1.unsqueeze(-1).unsqueeze(-1)*(self.metanet1(self.norm1(x), params['0.weight'], params['0.bias']))
            x =x + self.layer_scale2.unsqueeze(-1).unsqueeze(-1)*(self.metanet2(self.norm2(x), params['1.weight'], params['1.bias']))
        else: 
            x =x + self.metanet1(self.norm1(x), params['0.weight'], params['0.bias'])
            x =x + self.metanet2(self.norm2(x), params['1.weight'], params['1.bias'])
        self.last_layer(x)
        return x
    
######################################################################################################

class sFNOc2params(nn.Module):
    def __init__(self,
            stage_list=[3, 3,3],
            wavenumber_stage_list=[12, 12, 12], 
            features_stage_list=[36, 36, 36], 
            activation="leaky_relu",
            dim_input=1,
            dim_output=2, 
            drop_path_rate=0.3, 
            bias=False):
            super().__init__()
            self.bias = bias
            if bias == False:
                self.network = sFNO_epsilon_v2_updated(
                                    stage_list=stage_list,
                                    wavenumber_stage_list=wavenumber_stage_list, 
                                    features_stage_list=features_stage_list, 
                                    activation=activation,
                                    dim_input=dim_input,
                                    dim_output=dim_output, 
                                    drop_path_rate=drop_path_rate
                                                    )
                
                self.encoder1 = nn.ConvTranspose2d(2, 2, kernel_size=2, 
                                            stride=2, padding=0, 
                                            bias=False)
            else:
                self.network = nn.Conv1d(64, 2, kernel_size=1)
                self.encoder1 = nn.ConvTranspose1d(2, 2, kernel_size=2, 
                                    stride=2, padding=0, 
                                    bias=False)
  
    def forward(self, x):
        if self.bias == False:
            x = self.network(x)
            x = self.encoder1(x.permute(0, 3, 1, 2))
        if self.bias == True:
            x = self.network(x.squeeze(-1).permute(0,2,1))
            x = self.encoder1(x)
        return x
    
######################################################################################################
#Hyper_sFNO controling the the parameters of the 
#src to rcv
######################################################################################################
class Hyper_sFNO(nn.Module):
      def __init__(self,
                params,
                stage_list=[2, 2, 2],
                wavenumber_stage_list=[12, 12, 12], 
                features_stage_list=[36, 36, 36], 
                activation="leaky_relu",
                dim_input=1,
                dim_output=2, 
                drop_path_rate=0.3, 
                  ):
        super().__init__()
        self.MetaParams = params
        self.nets = nn.ModuleList()
        for name in params.keys():
          if name.endswith(".weight"):     
            self.nets.append(sFNOc2params(stage_list=stage_list,
                    wavenumber_stage_list=wavenumber_stage_list, 
                    features_stage_list=features_stage_list, 
                    activation=activation,
                    dim_input=dim_input,
                    dim_output=dim_output, 
                    drop_path_rate=drop_path_rate, 
                    bias=False))
          else:   
            self.nets.append(sFNOc2params(stage_list=stage_list,
                    wavenumber_stage_list=wavenumber_stage_list, 
                    features_stage_list=features_stage_list, 
                    activation=activation,
                    dim_input=dim_input,
                    dim_output=dim_output, 
                    drop_path_rate=drop_path_rate,
                    bias=True))
            
      def forward(self, x):
        params = OrderedDict()
        for name, net in zip(self.MetaParams, self.nets):
            batch_param_shape = (-1,) + self.MetaParams[name]
            params[name] = net(x).reshape(batch_param_shape)
        return params
      

######################################################################################################
#Forward map
######################################################################################################
class Forward_Operator(pl.LightningModule):
    r"""The forward map, going from wavespeed to the field generated at the receiver. 
    Neural Operators map wavspeed into a parameter that describes a network. Here
    we use a meta-network to capture the StR map. That is,
    wavespeed is mapped to a parameter whose described a network named 
    metaSrcRcv, whose send src position along a line into pressure field induced at 
    rcvs position.
    """
    def __init__(self, 
                dim_src_position =1, 
                numb_rcv=128,
                Lifting=None, 
                activation = "leaky_relu",
                mlp_ratio=4,
                use_layer_scale=True,
                layer_scale_init_value=1e-5, 
                stage_list=[2, 2, 2],
                wavenumber_stage_list=[12, 12, 12], 
                features_stage_list=[36, 36, 36], 
                dim_input=1,
                dim_output=2, 
                drop_path_rate=0.3,
                drop=0.1,
                loss = "rel_l2",
                learning_rate = 1e-3, 
                step_size= 70,
                gamma= 0.5,
                weight_decay= 1e-5, 
                ):
        super().__init__()
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()  
        elif loss == "rel_l2":
            self.criterion =LpLoss()
        self.metaSrC = MetaSrcRcv(
                                dim_src_position =dim_src_position, 
                                numb_rcv=numb_rcv,
                                Lifting=Lifting,
                                activation = activation,
                                mlp_ratio=mlp_ratio,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                drop=drop
                                )
        self.Wave2Src2Rcv = Hyper_sFNO(
                                stage_list=stage_list,  
                                wavenumber_stage_list=wavenumber_stage_list,
                                features_stage_list=features_stage_list,
                                activation=activation,
                                dim_input=dim_input,
                                dim_output=dim_output,
                                drop_path_rate=drop_path_rate,
                                params=self.metaSrC.params
                                )
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        
    def forward(self, wavespeed, src_position):
        params = self.Wave2Src2Rcv(wavespeed)
        return self.metaSrC(src_position, params)
    
    def training_step(self, batch: torch.Tensor, batch_idx):
      # training_step defines the train loop.
      # it is independent of forward        
      x, f, y = batch
      batch_size = x.shape[0]
      out= self(x,f)
      loss = self.criterion(out.view(batch_size,-1), y.view(batch_size,-1))   
      self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
      return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        x, f, y = val_batch
        batch_size = x.shape[0]
        out= self(x,f)
        val_loss = self.criterion(out.view(batch_size,-1), y.view(batch_size,-1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)     
        return val_loss

    def test_step(self, test_batch: torch.Tensor, batch_idx):
        x, f, y = test_batch
        batch_size = x.shape[0]
        out= self(x,f)
        test_loss = self.criterion(out.view(batch_size,-1), y.view(batch_size,-1))
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True)     
        return test_loss


    def configure_optimizers(self, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if  scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }