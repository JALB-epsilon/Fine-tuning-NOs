from matplotlib import pyplot as plt

import itertools
import numpy as np

def depth(data):
    if isinstance(data, list):
        d = np.array(data)
        return len(d.shape)
    elif isinstance(data, np.ndarray):
        return len(data)
    else:
        print(f'unable to determine depth of {data}')
        return 0

def scatterplot_matrix(data, names, colors=None, labels=None, markers=None, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if len(data.shape) == 2:
        print('single curve in input')
        numvars, numdata = data.shape
        data = np.array([data])
    elif len(data.shape) == 3:
        print(f'several curves in input ({data.shape[0]})')
        numvars, numdata = data[0].shape

    # numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        #Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        print(f'ax is {ax}')

        # Set up ticks only on one side for the "edge" subplots...
        if ax.get_subplotspec().is_first_col():
            ax.yaxis.set_ticks_position('left')
        # if ax.get_subplotspec().is_last_col():
        #     ax.yaxis.set_ticks_position('right')
        # if ax.get_subplotspec().is_first_row():
        #     ax.xaxis.set_ticks_position('top')
        if ax.get_subplotspec().is_last_row():
            ax.xaxis.set_ticks_position('bottom')
        ax.label_outer()

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for curve_id in range(data.shape[0]):
            cdata = data[curve_id]
            for x, y in [(i,j), (j,i)]:
                if colors is not None:
                    axes[x,y].plot(cdata[x], cdata[y], c=colors[curve_id], marker=markers[curve_id], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig

if __name__ == '__main__':
    np.random.seed(1977)
    numvars, numdata = 4, 10
    data = 10 * np.random.random((numvars, numdata))
    fig = scatterplot_matrix(data, ['mpg', 'disp', 'drat', 'wt'],
            linestyle='none', marker='o', color='black', mfc='none')
    fig.suptitle('Simple Scatterplot Matrix')
    plt.show()
