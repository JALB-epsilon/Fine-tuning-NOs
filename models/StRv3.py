import torch
import pytorch_lightning as pl
from utilities import LpLoss
from torch import optim, nn
from .basics_model import set_activ, LayerNorm
from collections import OrderedDict

from .sFNO_epsilon_v2_updated import sFNO_epsilon_v2_updated, Mlp
from .StR import FCNN, sFNOc2params
from .StRv2 import Complex_MetaFCLayer
from functools import partial

###################################    
class ComplexMetaSrc2Rcv_layer(nn.Module):
    def __init__(self, activation = "leaky_relu",
                numb_rcv=128,
                norm_layer=nn.LayerNorm, 
                drop= 0.,
                with_bias = False,
                layer_number = 0,
                is_last = False):
      super().__init__()
      self.numb_rcv = numb_rcv
      self.layer_number = layer_number
      self.with_bias = with_bias
      self.norm1 = norm_layer(128)
      self.metanet1 = Complex_MetaFCLayer(numb_rcv,
                              numb_rcv, 
                              activation = activation, 
                              is_last = is_last,
                              drop=drop)

      self.params = self.set_params()

    def set_params(self):
      """Set the parameters of the MetaSrcRcv layer."""
      layers_shapes = OrderedDict()
      layers_shapes[f"{self.layer_number}.weight"] = torch.Size([2, self.numb_rcv, self.numb_rcv])
      if self.with_bias:
            layers_shapes[f"{self.layer_number}.bias"] = torch.Size([2, self.numb_rcv])
      else:
            layers_shapes[f"{self.layer_number}.bias"] = None
      return layers_shapes
    
    
    def forward(self, x, params):
      #assert the tensor is four dimensional
      assert len(x.shape) == 4, f"The input tensor must be four dimensional (b,1,src,1) is {x.shape}"
      x = self.metanet1(self.norm1(x), params[f'{self.layer_number}.weight'], params[f"{self.layer_number}.bias"])
      return x
    
   
######################################################################################################
#Hyper_sFNO controling the the parameters of the 
#src to rcv
######################################################################################################
class Hyper_sFNO_v2(nn.Module):
    def __init__(self,
            params,
            stage_list=[2, 2, 2],
            wavenumber_stage_list=[12, 12, 12], 
            features_stage_list=[36, 36, 36], 
            activation="leaky_relu",
            dim_input=1,
            dim_output=2, 
            drop_path_rate=0.3
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
            elif name.endswith(".bias") and params[name] is not None:
                self.nets.append(sFNOc2params(stage_list=stage_list,
                        wavenumber_stage_list=wavenumber_stage_list, 
                        features_stage_list=features_stage_list, 
                        activation=activation,
                        dim_input=dim_input,
                        dim_output=dim_output, 
                        drop_path_rate=drop_path_rate,
                        bias=True))
            else: self.nets.append(nn.Identity())  
        
        if list(params.keys())[0].startswith ("1."):
            self.conv = nn.Sequential(
                        nn.Conv2d(2, 16, kernel_size= (1,2), stride=(1,2)),
                        nn.LeakyReLU())

    def forward(self, c, pressure_approx = None):
        params = OrderedDict()
        for name, net in zip(self.MetaParams, self.nets):
            if name.startswith ("0."):
                if self.MetaParams[name] is not None:
                  batch_param_shape = (-1,) + self.MetaParams[name]
                  params[name] = net(c).reshape(batch_param_shape)
                else: params[name] = None
            elif name.startswith ("1."):
                if name == "1.weight":
                  pressure_approx = (self.conv(pressure_approx)).permute(0,2,3, 1)
                  c_new = torch.cat((c, pressure_approx), dim=-1)
                  del pressure_approx
                  batch_param_shape = (-1,) + self.MetaParams[name]
                  params[name] = net(c_new).reshape(batch_param_shape)
                  del c_new
                elif name == "1.bias" and  self.MetaParams[name] is not None:
                    batch_param_shape = (-1,) + self.MetaParams[name]
                    params[name] = net(c).reshape(batch_param_shape)
                else: params[name] = None
        return params

class Forward_Operator_v3(pl.LightningModule):
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
                with_bias = False, 
                activation = "leaky_relu",
                mlp_ratio=4,
                stage_list=[2, 2, 2],
                wavenumber_stage_list=[12, 12, 12], 
                features_stage_list=[36, 36, 36], 
                dim_input=1,
                dim_output=2, 
                use_layer_scale = True,
                layer_scale_init_value =1e-3,
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

        self.save_hyperparameters() 
        mlp_hidden_features = int(numb_rcv * mlp_ratio)
        self.Lifting = nn.Sequential(
                           FCNN(in_features=dim_src_position, 
                           hidden_features=mlp_hidden_features,
                           out_features=numb_rcv,
                           activation = activation),   
                           nn.LayerNorm(numb_rcv),  
                           Mlp(in_features=1, hidden_features=8, out_features=2)              
                         ) 
        self.Layer0 =  ComplexMetaSrc2Rcv_layer(activation=activation,
                                                numb_rcv=numb_rcv,
                                                norm_layer=nn.LayerNorm,
                                                drop=drop, 
                                                with_bias=with_bias,
                                                layer_number=0,
                                                is_last=False)
        self.norm0 = nn.LayerNorm(numb_rcv)
        self.hyper0 = Hyper_sFNO_v2(params=self.Layer0.params,
                                    stage_list=stage_list,
                                    wavenumber_stage_list=wavenumber_stage_list, 
                                    features_stage_list=features_stage_list, 
                                    activation=activation,
                                    dim_input=dim_input,
                                    dim_output=dim_output, 
                                    drop_path_rate=drop_path_rate)
        
        self.Layer1 =  ComplexMetaSrc2Rcv_layer(activation=activation,
                                                numb_rcv=numb_rcv,
                                                norm_layer=nn.LayerNorm,
                                                drop=drop, 
                                                with_bias=with_bias,
                                                layer_number=1,
                                                is_last=False)
        
        self.hyper1 = Hyper_sFNO_v2(params=self.Layer1.params,
                                    stage_list=stage_list,
                                    wavenumber_stage_list=wavenumber_stage_list, 
                                    features_stage_list=features_stage_list, 
                                    activation=activation,
                                    dim_input=dim_input+16,
                                    dim_output=dim_output, 
                                    drop_path_rate=drop_path_rate)
        self.norm1 = nn.LayerNorm(numb_rcv)
        
        self.last_layer =FCNN(numb_rcv, numb_rcv*4, numb_rcv)
        if use_layer_scale: 
            self.layer_scale0 = nn.Parameter(torch.ones((2))*layer_scale_init_value, requires_grad=True)
            self.layer_scale1 = nn.Parameter(torch.ones((2))*layer_scale_init_value, requires_grad=True)
        
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        
    def forward(self, wavespeed, src_position):
        p_sigma = self.Lifting(src_position)
        del src_position
        params = self.hyper0(wavespeed)
        p_sigma = p_sigma+  self.layer_scale0.unsqueeze(-1).unsqueeze(-1)*(self.Layer0(self.norm0(p_sigma), params))
        del params
        params = self.hyper1(wavespeed, p_sigma)
        p_sigma = p_sigma+ self.layer_scale1.unsqueeze(-1).unsqueeze(-1)*(self.Layer1(self.norm1(p_sigma), params))
        del params
        p_sigma = p_sigma+self.last_layer(p_sigma)
        return p_sigma
    
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