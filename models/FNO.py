import pytorch_lightning as pl
import torch
from torch import optim, nn
from .basics_model import get_grid2D, set_activ, FC_nn
from utilities import LpLoss

#######################################
# Fourier Convolution, 
# \int_D k(x-y) v(y) dy 
# = \mathcal{F}^{-1}(P \mathcal{F}(v))
#######################################
class fourier_conv_2d(nn.Module):
    def __init__(self, in_, out_, wavenumber1, wavenumber2):
        super(fourier_conv_2d, self).__init__()
        self.out_ = out_
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_ * out_))
        self.weights1 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, dtype=torch.cfloat))
        # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2], self.weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

#######################################
# Fourier Layer: 
# \sigma( Wx + FourierConv(x))
#######################################
class Fourier_layer(nn.Module):
    def __init__(self,  features_, wavenumber, activation = 'relu', is_last = False):
        super(Fourier_layer, self).__init__()
        self.W =  nn.Conv2d(features_, features_, 1)
        self.fourier_conv = fourier_conv_2d(features_, features_ , *wavenumber)
        if is_last== False: 
            self.act = set_activ(activation)
        else: 
            self.act = set_activ(None)

    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return self.act(x1 + x2)        

#######################################
# FNO: Ensemble of the FNO
#######################################
class FNO(pl.LightningModule):
    def __init__(self,     
                    wavenumber, features_, 
                    padding = 9, 
                    activation= 'relu',
                    lifting = None, 
                    proj =  None, 
                    dim_input = 1, 
                    with_grid= True, 
                    loss = "rel_l2",
                    learning_rate = 1e-2, 
                    step_size= 100,
                    gamma= 0.5,
                    weight_decay= 1e-5,
                    ):
        super(FNO, self).__init__()
    
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
                                activation = "relu",
                                outermost_norm=False
                                    )
        else: 
            self.proj = proj
        self.fno = []
        for l in range(self.layers-1):
            self.fno.append(Fourier_layer(features_ = features_, 
                                        wavenumber=[wavenumber[l]]*2, 
                                        activation = activation))
        
        self.fno.append(Fourier_layer(features_=features_, 
                                        wavenumber=[wavenumber[-1]]*2, 
                                        activation = activation,
                                        is_last= True))
        self.fno =nn.Sequential(*self.fno)


    def forward(self, x: torch.Tensor):
        if self.with_grid == True:
          grid = get_grid2D(x.shape, x.device)
          x = torch.cat((x, grid), dim=-1)
        x = self.lifting(x) 
        x = x.permute(0, 3, 1, 2) 
        x = nn.functional.pad(x, [0,self.padding, 0,self.padding]) 
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
    