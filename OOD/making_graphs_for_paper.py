import numpy as np 
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def rel_l2( ref, approx): 
    diff = np.abs((approx-ref).view(np.csingle)).reshape(-1, ref.shape[1], ref.shape[2])
    den = np.linalg.norm(ref.view(np.csingle), ord=2, axis=(1,2)).reshape(-1,1,1)
    return diff/den

def loading_data_OOD_arch(args):
    PATH = args.PATH
    data_files = dict()
    print(PATH)
    #ground truth
    data_files['pressure_hdg'] = np.load(os.path.join(PATH,'pressure_set_{:02d}_freq15.npy'.format(args.ood_sample)))[:args.n_sample]
    data_files['wavespeed'] = np.load(os.path.join(PATH,'wavespeed_set_{:02d}_freq15.npy'.format(args.ood_sample)))[:args.n_sample]
    #loading the data fno
    data_files['fno_approx'] = np.load(os.path.join(PATH,'pressure_FNO_set_{:02d}_freq15.npy'.format(args.ood_sample)))[:args.n_sample]
    data_files['error_fno_approx'] = rel_l2(data_files['pressure_hdg'], data_files['fno_approx'])
    #loading the data sFNO
    data_files['sfno_approx'] = np.load(os.path.join(PATH,'pressure_sFNO_set_{:02d}_freq15.npy'.format(args.ood_sample)))[:args.n_sample]
    data_files['error_sfno_approx'] = rel_l2(data_files['pressure_hdg'], data_files['sfno_approx'])
    #loading the data sFNO_eps v1
    data_files['sfno_eps1_approx'] = np.load(os.path.join(PATH,'pressure_sFNO_epsilon_v1_set_{:02d}_freq15.npy'.format(args.ood_sample)))[:args.n_sample]
    data_files['error_sfno_eps1_approx'] = rel_l2(data_files['pressure_hdg'], data_files['sfno_eps1_approx'])
    #loading the data sFNO
    data_files['sfno_eps2_approx'] = np.load(os.path.join(PATH,'pressure_sFNO_epsilon_v1_long_set_{:02d}_freq15.npy'.format(args.ood_sample)))[:args.n_sample]
    data_files['error_sfno_eps2_approx'] = rel_l2(data_files['pressure_hdg'], data_files['sfno_eps2_approx'])
    #loading the data sFNO
    data_files['residual_fno'] = np.load(os.path.join(PATH,'pressure_FNO_residual_set_{:02d}_freq15.npy'.format(args.ood_sample)))[:args.n_sample]
    data_files['error_residual_fno'] = rel_l2(data_files['pressure_hdg'], data_files['residual_fno'])
    return data_files

def plotting_data_OOD_arch(args, data_files):
    PATH = os.path.join(args.PATH, 'graphs')
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    for name in data_files.keys():
        if name =="wavespeed":
            v_min = 1.5
            v_max = 5
            cmap = 'jet'
        else: 
            v_min = -0.1 
            v_max = 0.1
            cmap = 'seismic'
        for i in range(len(data_files[name])):
            print(name)
            data = data_files[name][i,...]            
            if name == 'wavespeed':
                fig =plt.imshow(data, vmin=v_min, vmax=v_max, cmap=cmap, aspect='equal')
                #eliminate space between the image and the axis
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                if args.save_graph:
                    plt.savefig(os.path.join(PATH, f'cp_{i}_{name}.png'),bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    plt.show()
            #if the name contains error in any part, we plot the error
            elif name.find('error') != -1:
                #make log scale in imshow
                #norm = matplotlib.colors.LogNorm(vmin=1e-4, vmax=1e-1)

                fig =plt.imshow(data, cmap='jet', aspect='equal', 
                                norm=LogNorm(vmin=1e-4, vmax=1e-1))
                #eliminate space between the image and the axis
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                #make log scale
      
                if args.save_graph:
                    plt.savefig(os.path.join(PATH, f'cp_{i}_{name}.png'),bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    plt.show()


            else:
                data_real= data[:,:,0]
                fig =plt.imshow(data_real, vmin=v_min, vmax=v_max, cmap=cmap, aspect='equal')
                #eliminate space between the image and the axis
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                if args.save_graph:
                    plt.savefig(os.path.join(PATH, f'cp_{i}_{name}_real.png'),bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    plt.show()
                data_imag= data[:,:,1]
                fig =plt.imshow(data_imag, vmin=v_min, vmax=v_max, cmap=cmap, aspect='equal')
                #eliminate space between the image and the axis
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                if args.save_graph:
                    plt.savefig(os.path.join(PATH, f'cp_{i}_{name}_imag.png'), bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ood','--ood_sample', type=int, 
                        help='out of distribution set',
                        default=5)
    parser.add_argument('-sg','--save-graph', type=bool,
                        help='Saving Image',
                        default=True)
    parser.add_argument('-rk','--realization_k', type=int,
                        help='realization number',
                        default=0)
    parser.add_argument('-n','--n_sample', type=int,
                        help='number of sample',
                        default=6)
    parser.add_argument('-f','--freq', type=int,
                        help='frequency of the OOD',
                        default=15)
    args = parser.parse_args()


    # getting the name of the dataset 
    dir_skeleton = 'set_{:02d}'.format(args.ood_sample)+f'_freq{args.freq}'
    print(f"dir skel {dir_skeleton}")
    PATH = os.path.join('OOD', dir_skeleton, f'realization_{args.realization_k}')
    args.PATH = PATH
    #getting the data
    data_files = loading_data_OOD_arch(args)
    #plotting the data
    plotting_data_OOD_arch(args, data_files)