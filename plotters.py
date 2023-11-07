import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

"""
 
 ########  ##        #######  ######## ######## ######## ########   ######  
 ##     ## ##       ##     ##    ##       ##    ##       ##     ## ##    ## 
 ##     ## ##       ##     ##    ##       ##    ##       ##     ## ##       
 ########  ##       ##     ##    ##       ##    ######   ########   ######  
 ##        ##       ##     ##    ##       ##    ##       ##   ##         ## 
 ##        ##       ##     ##    ##       ##    ##       ##    ##  ##    ## 
 ##        ########  #######     ##       ##    ######## ##     ##  ######  
 
"""

# decorator
def plotter(func):
    def wrapped(*args, **kwargs):
        if 'axes' not in kwargs.keys():
            fig, axes = plt.subplots()
            kwargs['axes'] = axes

        axes = func(*args, **kwargs)

        if 'outpath' in kwargs.keys():
            fig = axes.get_figure()
            fig.savefig(kwargs['outpath'], dpi=600)
            plt.close(fig)
            return None
        else:
            return axes

    return wrapped

@plotter
def plot_coverage(IDs, axes=None, outpath=None):
    n_sessions = len(IDs)

    max_id = np.max([np.max(ids.astype('uint32')) for ids in IDs])
    for i in range(max_id):
        pres = np.zeros(n_sessions)
        for d in range(n_sessions):
            if str(i) in IDs[d]:
                pres[d] = 1
            else:
                pres[d] = 0
        pres = pres.astype('bool')
        axes.plot(np.arange(n_sessions)[pres], np.ones(n_sessions)[pres]*i)

    return axes

@plotter
def plot_Coords(Coords, color_by='session', title=None, axes=None, outpath=None):

    n_sessions = len(Coords)
    # n_cells = len(D[0])

    if color_by == 'session':
        colors = sns.color_palette('viridis', n_colors=n_sessions)
        for i, coords in enumerate(Coords):
            axes.plot(coords[:,0], coords[:,1],'.',alpha=0.75,markersize=6, color=colors[i], label='session %i' % (i+1))

    if color_by == 'chain':
        # this requires Coords to be Coords_chains (all coords have same shape)
        n_cells = Coords[0].shape[0]
        colors = sns.color_palette('tab10', n_colors=n_cells)

        for i in range(n_cells):
            c = np.array([Coords[d][i,:] for d in range(n_sessions)])
            axes.plot(c[:,0],c[:,1],'.',alpha=0.75, markersize=6, color=colors[i])

    if title is None:
        title = 'all sessions'

    axes.set_title(title)
    axes.grid(zorder=-1)
    axes.set_aspect('equal')
    pad = 0.05
    axes.set_xlim(0-pad, 1+pad)
    axes.set_ylim(0-pad, 1+pad)

    return axes

@plotter
def plot_Roi_shapes_chain(R, axes=None, normalize=False, outpath=None):
    """ R is of shape (n_sessions, n_xpx, n_ypx) """

    n_sessions, n_xpx, nypx = R.shape
    f = []
    for d in range(n_sessions):
        data = R[d,:,:]
        if normalize:
            data = data / data.max()
        f.append(data)

    f = np.concatenate(f,axis=1)

    axes.matshow(f)
    axes.set_xticks([])
    axes.set_yticks([])
    for i in range(1, n_sessions):
        axes.axvline(i * n_xpx, color='w', lw=1)

    return axes

# def plot_Roi_shapes(Roi_shapes_chain, ix, normalize=False, axes=None, outpath=None):
#     """
#     Roi_shapes_chain is of shape (n_sessions, n_cells, n_xpx, n_ypx)
#     ix are all the indices to plot
#     """

#     for i, j in enumerate(ix):
#         plot_Roi_shapes_chain(Roi_shapes_chain[:,j,:,:], axes=axes[i], normalize=normalize)

#     return axes

def plot_all_Roi_shapes(Roi_shapes_chain, n_cells_per_figure=10, normalize=False, outpath=None):

    n_sessions, n_cells, n_xpx, n_ypx = Roi_shapes_chain.shape

    # figure out how many figures
    n_figures = n_cells // n_cells_per_figure
    if n_cells % n_cells_per_figure == 0:
        n_figures += 1

    for i in range(n_figures):
        ix = np.arange(i*n_cells_per_figure,(i+1)*n_cells_per_figure).astype('int32')
        ix = ix[ix < n_cells]
        fig, axes = plt.subplots(nrows=ix.shape[0])
        # plot_Roi_shapes(Roi_shapes_chain, ix, normalize=normalize)
        for k, j in enumerate(ix):
            plot_Roi_shapes_chain(Roi_shapes_chain[:,j,:,:], axes=axes[k], normalize=normalize)

        if outpath is not None:
            outpath = (outpath.parent / (outpath.stem + '_' + str(i))).with_suffix(outpath.suffix)
            fig.savefig(outpath, dpi=600)
            plt.close(fig)


@plotter
def plot_tm_scores(tm_scores, tm_scores_rand, tm_chance_lvl, axes=None, outpath=None):

    axes.hist(tm_scores, bins=np.linspace(0,1,100), alpha=0.5)
    axes.hist(tm_scores_rand, bins=np.linspace(0,1,100), alpha=0.5)
    axes.axvline(tm_chance_lvl,linestyle=':',color='k')
    fig = axes.get_figure()
    sns.despine(fig)

    return axes





def plot_coordinates(Coords_reg, Coords_chains, Coords_chains_good, outpath=None):

    fig, axes = plt.subplots(ncols=3)
    plot_Coords(Coords_reg, color_by='session', axes=axes[0], title='sessions')
    plot_Coords(Coords_chains, color_by='chain', axes=axes[1], title='chains')
    plot_Coords(Coords_chains_good, color_by='chain', axes=axes[2], title='chains cleaned')

    if outpath is not None:
        fig.savefig(outpath, dpi=600)
        plt.close(fig)

    return fig