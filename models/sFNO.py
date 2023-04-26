import pytorch_lightning as pl
import torch
from torch import optim, nn
from .FNO import fourier_conv_2d
from .basics_model import LayerNorm, get_grid2D, FC_nn, set_activ
import torch.nn.functional as F
from utilities import LpLoss

#######################################
# Integral Operator Layer
#######################################
class IO_layer(nn.Module):
    def __init__(self,  features_, 
                        wavenumber, 
                        drop = 0., 
                        activation = "relu"):
        super().__init__()
        self.W =  nn.Conv2d(features_, features_, 1)
        self.IO = fourier_conv_2d(features_, features_,*wavenumber)
        self.act = set_activ(activation)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.IO(x)+self.W(x)
        x = self.dropout(x) 
        x = self.act(x)
        return x

#######################################
# Integral Operator Block
#######################################
class IO_Block(nn.Module):
    def __init__(self, features_, 
                        wavenumber,
                        drop = 0.,
                        activation = "relu"):
        super().__init__()
        self.IO = IO_layer(features_=features_,
                            wavenumber=wavenumber, 
                            drop= drop, 
                            activation = activation)
        self.pwconv1 = nn.Linear(features_, 4*features_) # pointwise/1x1 convs, implemented with linear layers
        self.act = set_activ(activation) if activation is not None else set_activ("gelu")
        self.norm = nn.LayerNorm(features_, eps=1e-5)
        self.pwconv2 = nn.Linear(4*features_, features_) #
    def forward(self, x):
        x =(self.IO(x)).permute(0,2,3,1) #B C W H -> B W H C
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x  = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

#######################################
# sFNO: Ensemble of the sFNO
#######################################
class sFNO(pl.LightningModule):
    def __init__(self,     
                    wavenumber, features_, 
                    padding = 9,
                    lifting = None, 
                    proj =  None, 
                    dim_input = 1, 
                    with_grid= True, 
                    loss = "rel_l2",
                    learning_rate = 1e-2, 
                    step_size= 100,
                    gamma= 0.5,
                    weight_decay= 1e-5,
                    drop = 0.,
                    activation = "relu"
                    ):
        super().__init__()
    
        self.with_grid = with_grid
        self.padding = padding   
        self.layers = len(wavenumber)
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        elif loss == "rel_l2":
            self.criterion =LpLoss()

        if with_grid == True: 
            dim_input+=2 
        if lifting is None:
            self.lifting = FC_nn([dim_input, features_//2, features_], 
                                    activation = "relu",
                                    outermost_norm=False
                                    )
        else: 
            self.lifting = lifting
        if  proj is None: 
            self.proj =  FC_nn([features_, features_//2, 1], 
                                activation = "relu", drop = drop, 
                                outermost_norm=False
                                    )
        else: 
            self.proj = proj
        self.fno = []
        for l in range(self.layers):
            self.fno.append(IO_Block(features_ = features_, 
                                        wavenumber=[wavenumber[l]]*2, 
                                        drop= drop, 
                                        activation= activation))
        

        self.fno =nn.Sequential(*self.fno)


    def forward(self, x: torch.Tensor):
        if self.with_grid == True:
          grid = get_grid2D(x.shape, x.device)
          x = torch.cat((x, grid), dim=-1)
        x = self.lifting(x) 
        x = x.permute(0, 3, 1, 2) 
        x = F.pad(x, [0,self.padding, 0,self.padding]) 
        x = self.fno(x)
        x = x[..., :-self.padding, :-self.padding] 
        x = x.permute(0, 2, 3, 1 ) 
        x =self.proj(x)
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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }