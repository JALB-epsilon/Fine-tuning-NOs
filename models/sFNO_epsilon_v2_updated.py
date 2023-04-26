import pytorch_lightning as pl
import torch
from torch import optim, nn
from .FNO import fourier_conv_2d
from .basics_model import LayerNorm, get_grid2D, set_activ, GroupNorm
import torch.nn.functional as F
from utilities import LpLoss
from timm.models.layers import DropPath, trunc_normal_
import os
from .sFNO_epsilon_v1 import IO_layer 

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, 
                hidden_features=None, 
                out_features=None,  
                activation = "leaky_relu"):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = set_activ(activation)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
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

#######################################
# Transformer look-alike block with Neural Operators
#######################################
class NOFormerBlock(nn.Module):
    def __init__(self, features_, 
                        wavenumber,
                        drop = 0., 
                        drop_path= 0., 
                        activation = "leaky_relu", 
                        use_layer_scale=True, 
                        layer_scale_init_value=1e-5, 
                        norm_layer=GroupNorm,
                        mlp_ratio=4):
        super().__init__()
        self.IO = IO_layer(features_=features_,
                            wavenumber=wavenumber, 
                            drop= drop, 
                            activation = activation)
        self.norm1 = norm_layer(features_)
        self.norm2 = norm_layer(features_)
        self.act = set_activ(activation) if activation is not None else set_activ("gelu")
        mlp_hidden_features = int(features_ * mlp_ratio)
        self.mlp = Mlp(in_features=features_, 
                        hidden_features=mlp_hidden_features, 
                        activation=activation)
        self.drop_path= DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale 
        if use_layer_scale: 
            self.layer_scale_1= nn.Parameter(torch.ones((features_))*layer_scale_init_value, requires_grad=True)
            self.layer_scale_2= nn.Parameter(torch.ones((features_))*layer_scale_init_value, requires_grad=True)
        
   
    def forward(self, x):
        if self.use_layer_scale:
            x = x+ self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)*self.IO(self.norm1(x)))
            x = x+ self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)*self.mlp(self.norm2(x)))
        else:
            x = x+ self.drop_path(self.IO(self.norm1(x)))
            x = x+ self.drop_path(self.mlp(self.norm2(x)))
        return x

#######################################
class sFNO_epsilon_v2_updated(pl.LightningModule):
    def __init__(self, 
                stage_list,
                features_stage_list, 
                wavenumber_stage_list, 
                dim_input = None, 
                dim_output = None, 
                proj= None,
                lifting=None,
                activation="leaky_relu",
                norm_layer=GroupNorm,
                drop_rate= 0., 
                drop_path_rate= 0.,
                use_layer_scale=True, 
                layer_scale_init_value=1e-5,                  
                with_grid=True,
                padding=9,
                loss = "rel_l2",
                learning_rate = 1e-3, 
                step_size= 70,
                gamma= 0.5,
                weight_decay= 1e-5, 
                mlp_ratio=4):
        super().__init__()
        self.save_hyperparameters()
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()  
        elif loss == "rel_l2":
            self.criterion =LpLoss()
        self.padding = padding
        self.with_grid = with_grid
        self.padding = padding
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        
        if with_grid == True: 
            dim_input+=2 
        if lifting is None:
            self.lifting = Mlp(in_features=dim_input, 
                                out_features=features_stage_list[0],
                                hidden_features=features_stage_list[0], 
                                activation=activation)
            
        else: 
            self.lifting = lifting
            
        if  proj is None: 
            self.proj =  Mlp(in_features=features_stage_list[-1], 
                                out_features=dim_output,
                                hidden_features=features_stage_list[-1], 
                                activation=activation)
            
        else: 
            self.proj = proj   

        assert len(features_stage_list) == len(wavenumber_stage_list) == len(stage_list)
        network = []
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(stage_list))] 
        cur = 0
        for i in range(len(stage_list)):
            stage = self.Ensemble_stage(features_=features_stage_list[i],
                                        index= i,
                                        layers=stage_list,
                                        wavenumber_stage=wavenumber_stage_list[i], 
                                        mlp_ratio=mlp_ratio,
                                        activation=activation,
                                        norm_layer=norm_layer,
                                        drop_rate=drop_rate, 
                                        drop_path_rate=dp_rates[cur:cur+stage_list[i]],
                                        use_layer_scale=use_layer_scale, 
                                        layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            cur += stage_list[i]

        self.network = nn.ModuleList(network)
       
        #######################################
    def Ensemble_stage(self, features_, 
                            index, 
                            layers, 
                            wavenumber_stage, 
                            mlp_ratio,
                            activation,
                            norm_layer,
                            drop_rate, 
                            drop_path_rate,
                            use_layer_scale, 
                            layer_scale_init_value, 
                            ):
        """
        generate the ensemble of blocks
        return: NOFormerBlock
        """
        blocks = []
        for j in range(layers[index]):
            blocks.append(NOFormerBlock(features_= features_,
                                        wavenumber= [wavenumber_stage]*2,
                                        norm_layer=norm_layer,
                                        drop= drop_rate,
                                        drop_path= drop_path_rate[j],
                                        use_layer_scale=use_layer_scale,
                                        layer_scale_init_value=layer_scale_init_value, 
                                        mlp_ratio=mlp_ratio,
                                        activation=activation))
        blocks = nn.Sequential(*blocks)
        return blocks

    def forward_NOFormer(self, x):
        """
        forward the NOFormer
        """
        for stage in self.network:
            x = stage(x)
        return x
    
    def add_grid(self, x):
        """
        add grid to the input
        """
        grid = get_grid2D(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) 
        del grid
        return x
        

    def forward(self, x):
        if self.with_grid == True:
            x = self.add_grid(x)  
        x = self.lifting(x.permute(0, 3, 1, 2))
        x = F.pad(x, [0,self.padding, 0,self.padding]) 
        x = self.forward_NOFormer(x)
        x =x[..., :-self.padding, :-self.padding] 
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)
    
    def training_step(self, batch: torch.Tensor, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward        
        x, y = batch
        batch_size = x.shape[0]
        out= self(x)
        loss = self.criterion(out.view(batch_size,-1), y.view(batch_size,-1))   
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        x, y = val_batch
        batch_size = x.shape[0]
        out= self(x)
        val_loss = self.criterion(out.view(batch_size,-1), y.view(batch_size,-1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)     
        return val_loss

    def test_step(self, test_batch: torch.Tensor, batch_idx):
        x, y = test_batch
        batch_size = x.shape[0]
        out= self(x)
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



class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x