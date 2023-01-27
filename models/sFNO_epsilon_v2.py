import pytorch_lightning as pl
import torch
from torch import optim, nn
from .FNO import fourier_conv_2d
from .basics_model import LayerNorm, get_grid2D, FC_nn
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from utilities import LpLoss
from .sFNO import IO_layer

###################################################
# Integral Operator Layer Block with skip connection
###################################################
class IO_ResNetblock(nn.Module):
    def __init__(self,  features_, 
                      wavenumber, 
                      drop_path = 0., 
                      drop = 0.):
        super().__init__()
        self.IO = IO_layer(features_, wavenumber, drop)
        self.pwconv1 = nn.Conv2d(features_, 4* features_, 1) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 =nn.Conv2d(4 * features_, features_,1)
        self.norm1 = LayerNorm(features_, eps=1e-5,  data_format = "channels_first")
        self.norm2 = LayerNorm(features_, eps=1e-5,  data_format = "channels_first")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

 
    def forward(self, x):
        skip = x
        x = self.norm1(x)
        x =self.IO(x)
        x =skip+self.drop_path(x) #NonLocal Layers
        skip = x 
        x = self.norm2(x)
        #local
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = skip + self.drop_path(x)
        return x  



#######################################
#sFNO_epsilon_v2
#######################################
class sFNO_epsilon_v2(pl.LightningModule): 
    def __init__(self, 
                in_chans = 3, 
                out_chans = 1, 
                modes = [12, 12, 12, 12],
                depths = [3,3,9,3], 
                dims = [36, 36, 32, 34],
                drop_path_rate = 0., 
                drop = 0.,
                head_init_scale=1.,
                padding=9,
                with_grid = True, 
                loss = "rel_l2",
                learning_rate = 1e-3, 
                step_size= 100,
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

        self.with_grid = with_grid
        self.padding = padding
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.lifting_layers = nn.ModuleList()
        steam =  nn.Conv2d(in_chans, dims[0], 1,1)
        self.lifting_layers.append(steam)
        for i in range(3):
            lifting_layers = nn.Sequential(
                                            LayerNorm(dims[i], eps=1e-6, data_format= "channels_first"),
                                            nn.Conv2d(dims[i], dims[i+1], kernel_size = 1, stride = 1)
                                            
                                        )
            self.lifting_layers.append(lifting_layers)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
                    stage = nn.Sequential(
                        *[IO_ResNetblock(features_=dims[i], 
                                    wavenumber=[modes[i]]*2, 
                                    drop_path=dp_rates[cur + j],
                                    drop =drop) for j in range(depths[i])]
                    )
                    self.stages.append(stage)
                    cur += depths[i]

        self.head = nn.Conv2d(dims[-1], out_chans,1,1)
        

    def forward_features(self, x):
        x=x.permute(0,3,1,2).contiguous()
        x = self.lifting_layers[0](x)
        x = F.pad(x, [0,self.padding, 0, self.padding])
        for i in range(1,4):
            x = self.lifting_layers[i](x)
            x = self.stages[i](x)
        x = x[..., :-self.padding, :-self.padding] 
        return x

    def forward(self, x):
        if self.with_grid:
            grid = get_grid2D(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1) 
            del grid
        x = self.forward_features(x)
        x = self.head(x)
        return x

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


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }
    
#######################################
#  Ensemble of the sFNO_epsilon_v2_proj
# the only diff. is in allowing the projection 
# to be taken as an input
#######################################
class sFNO_epsilon_v2_proj(pl.LightningModule):
    def __init__(self, 
                in_chans = 3, 
                proj = None, 
                modes = [12, 12, 12, 12],
                depths = [3,3,9,3], 
                dims = [36, 36, 32, 34],
                drop_path_rate = 0., 
                drop = 0.,
                head_init_scale=1.,
                padding=9,
                with_grid = True, 
                loss = "rel_l2",
                learning_rate = 1e-3, 
                step_size= 100,
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

        self.with_grid = with_grid
        self.padding = padding
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.lifting_layers = nn.ModuleList()
        steam =  nn.Conv2d(in_chans, dims[0], 1,1)
        self.lifting_layers.append(steam)
        for i in range(3):
            lifting_layers = nn.Sequential(
                                            LayerNorm(dims[i], eps=1e-6, data_format= "channels_first"),
                                            nn.Conv2d(dims[i], dims[i+1], kernel_size = 1, stride = 1)
                                            
                                        )
            self.lifting_layers.append(lifting_layers)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
                    stage = nn.Sequential(
                        *[IO_ResNetblock(features_=dims[i], 
                                    wavenumber=[modes[i]]*2, 
                                    drop_path=dp_rates[cur + j],
                                    drop =drop) for j in range(depths[i])]
                    )
                    self.stages.append(stage)
                    cur += depths[i]

        if  proj is None: 
            self.proj =  FC_nn([features_, features_//2, 1], 
                                activation = "relu",
                                outermost_norm=False
                                    )
        else: 
            self.proj = proj
        
        

    def forward_features(self, x):
        x=x.permute(0,3,1,2).contiguous()
        x = self.lifting_layers[0](x)
        x = F.pad(x, [0,self.padding, 0, self.padding])
        for i in range(1,4):
            x = self.lifting_layers[i](x)
            x = self.stages[i](x)
        x = x[..., :-self.padding, :-self.padding] 
        return x

    def forward(self, x):
        if self.with_grid:
            grid = get_grid2D(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1) 
            del grid
        x = self.forward_features(x)
        x = x.permute(0, 2, 3, 1 ) 
        x = self.proj(x)
        return x

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


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }
