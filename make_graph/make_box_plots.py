import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Freq', add_help=False)
    parser.add_argument('-f','--freq', type=int, 
                               default=7)
    parser.add_argument('-min','--min', type=float, 
                               default=0.)
    parser.add_argument('-max','--max', type=float, 
                               default=0.5)

    args=parser.parse_args()
    df = pd.read_csv(f"make_graph/test_GRF_{args.freq}Hz.csv")
    print(df)
    fig, ax = plt.subplots(figsize=(10,6))
    #ax.set_yscale("log")
    '''    box = sns.boxplot(
        data=df, 
        notch=False, showcaps=True,
        flierprops={"marker": "x"},
        boxprops={"facecolor": (.4, .6, .8, .5)},
        medianprops={"color": "coral"}, ax=ax, linewidth =2, orient = "h"
        )'''
    box = sns.violinplot(
        data=df, 
        flierprops={"marker": "x"}, inner="stick",
        palette= "pastel6", width=0.8,
        ax=ax, linewidth =2.5, orient = "h", 
        scale="width", 
        )
    #box.set(xscale="log")
    ax.set_xlim([args.min, args.max])
    

    # Tweak the visual presentation
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    sns.despine(trim=True, left=True)
    box.set_title(f"Test Loss (rel. L2): Helmholtz {args.freq} Hz")
    plt.savefig(f'make_graph/box_plot_test_loss_{args.freq}.png')
