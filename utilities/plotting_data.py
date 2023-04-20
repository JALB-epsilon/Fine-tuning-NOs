import matplotlib.pyplot as plt
from .loading_data import to_numpy
import os 

def plotting(in_, NN_out, out, name, database, PATH,
            list_to_plot = None, vmin=-0.5, vmax =0.5, 
            shrink = 0.8, ksample = 0):

    if list_to_plot is None:
        list_to_plot = [0,1,2,3,4,5]
        print("list_to_plot is None, so we plot the first 6 samples")
    assert in_.shape[0] >= len(list_to_plot); "list of samples to plot is bigger than the input size"
    in_ =   to_numpy(in_)[:list_to_plot[-1]+1,...]       
    NN_out = to_numpy(NN_out)[:list_to_plot[-1]+1,...]
    out = to_numpy(out)[:list_to_plot[-1]+1,...]
    s = in_.shape[1]
    for k in list_to_plot:
        print(f"plotting sample {k}") 
        in_k = in_[k,...].reshape(s,s)
        out_k = out[k,...].reshape(s,s,-1)
        NN_k =NN_out[k,...].reshape(s,s,-1)        
        plt.figure(figsize=(20,10))
        plt.subplot(231)
        plt.imshow(in_k, vmin=1., vmax =5., cmap = 'jet')
        plt.colorbar(shrink =shrink)
        plt.title(f'wavespeed: {k}')

        plt.subplot(232)
        plt.imshow(out_k[:,:,0].reshape(s,s), vmin=vmin, vmax =vmax, cmap = 'seismic')
        plt.colorbar(shrink =shrink)
        plt.title(f'HDG (real) sample: {k}')

        plt.subplot(233)
        plt.imshow(NN_k[:,:,0].reshape(s,s), vmin=vmin, vmax =vmax, cmap = 'seismic')
        plt.colorbar(shrink =shrink)
        plt.title(f'{name} (real) sample: {k}')

        plt.subplot(235)
        plt.imshow(out_k[:,:,1].reshape(s,s), vmin=vmin, vmax =vmax, cmap = 'seismic')
        plt.colorbar(shrink =shrink)
        plt.title(f'HDG (imaginary) sample: {k}')

        plt.subplot(236)
        plt.imshow(NN_k[:,:,1].reshape(s,s), vmin=vmin, vmax =vmax, cmap = 'seismic')
        plt.colorbar(shrink =shrink)
        plt.title(f'{name} (imaginary) sample: {k}')

        saving_dir = os.path.join(PATH, "figures", database, name, f"realization_{ksample}")            
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        plt.savefig(f"{saving_dir}/ex_{k}.png")