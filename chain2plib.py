# imports
import sys, os
from pathlib import Path

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from scipy.stats import gaussian_kde as kde

from skimage.feature import match_template
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
from itertools import combinations
from copy import copy


"""
 
 ########  ########    ###    ##          ########     ###    ########    ###    
 ##     ## ##         ## ##   ##          ##     ##   ## ##      ##      ## ##   
 ##     ## ##        ##   ##  ##          ##     ##  ##   ##     ##     ##   ##  
 ########  ######   ##     ## ##          ##     ## ##     ##    ##    ##     ## 
 ##   ##   ##       ######### ##          ##     ## #########    ##    ######### 
 ##    ##  ##       ##     ## ##          ##     ## ##     ##    ##    ##     ## 
 ##     ## ######## ##     ## ########    ########  ##     ##    ##    ##     ## 
 
"""

def load_stats(folder, good_ratio=0.5):
    """
    reads suite2p generated stats.npy and iscell.npy
    folder is suite2p output folder (after planeX)
    good_ratio = 0.5 is equivalent to previous good_only=True
    """

    stats = np.load(folder / 'stat.npy', allow_pickle=True)
    iscell = np.load(folder / 'iscell.npy')

    ix_sel = iscell[:,1] > good_ratio
    stats = [stat for j, stat in enumerate(stats) if ix_sel[j] == True]

    return stats, ix_sel

def get_roi_coords(stat):
    """
    from a suite2p stat, compute the center of mass of a given roi
    """
    w = stat['lam']/np.max(stat['lam'])
    x = np.sum(stat['xpix'] * w) / np.sum(w)
    y = np.sum(stat['ypix'] * w) / np.sum(w)
    return (x, y)

def load_data(folders, good_ratio=0.5):
    """
    pipeline function
    folders is list of suite2p folder (for each session)
    get across sessions data
    a list of stats
    a list of coordinates
    """

    Stats = []
    Coords = []
    for i, folder in enumerate(folders):
        # Stats
        stats, _ = load_stats(folder, good_ratio=good_ratio)
        Stats.append(stats)
        
        # get coords from stats
        coords = np.zeros((len(stats),2))
        for j, stat in enumerate(stats):
            coords[j,:] = get_roi_coords(stat)
        Coords.append(coords)

    return Coords, Stats

# %% 
"""
 
 ########  ########  ######   ####  ######  ######## ########     ###    ######## ####  #######  ##    ## 
 ##     ## ##       ##    ##   ##  ##    ##    ##    ##     ##   ## ##      ##     ##  ##     ## ###   ## 
 ##     ## ##       ##         ##  ##          ##    ##     ##  ##   ##     ##     ##  ##     ## ####  ## 
 ########  ######   ##   ####  ##   ######     ##    ########  ##     ##    ##     ##  ##     ## ## ## ## 
 ##   ##   ##       ##    ##   ##        ##    ##    ##   ##   #########    ##     ##  ##     ## ##  #### 
 ##    ##  ##       ##    ##   ##  ##    ##    ##    ##    ##  ##     ##    ##     ##  ##     ## ##   ### 
 ##     ## ########  ######   ####  ######     ##    ##     ## ##     ##    ##    ####  #######  ##    ## 
 
"""

# MAJOR TODO integrate transformation matrices to deal with different transforms - rotation!
# %%

def apply_transform(coords, T):
    X  = np.concatenate([coords, np.ones((coords.shape[0],1))],axis=1).T
    Xt = T @ X
    return Xt[:-1,:].T

def rotate(coords, theta, origin):
    if type(origin) is list:
        origin = np.array(origin)
    T_trans = get_T_trans(origin)
    T_trans_inv = get_T_trans(-1 * origin)
    T_rot = get_T_rot(theta)
    return apply_transform(coords, T_trans @ T_rot @ T_trans_inv)

def translate(coords, point):
    if type(point) is list:
        point = np.array(point)
    T = get_T_trans(point)
    return apply_transform(coords, T)

def get_T_trans(point):
    # point is of shape (2,)
    if type(point) is list:
        point = np.array(point)
    """ the translation matrix to point T """
    T = np.identity(3)
    T[:2,2] = point
    return T

def get_T_rot(theta):
    T = np.zeros((3,3))
    T[0,0] = np.cos(theta)
    T[0,1] = -np.sin(theta)
    T[1,0] = np.sin(theta)
    T[1,1] = np.cos(theta)
    T[2,2] = 1
    return T


# %%
# rewrite in affine trans

def nearest_dist_ssq(shifts, coords_a, coords_b):
    """
    A and B are (n,2) coordinates

    shifts B relative to A
    to be minimized

    conceptual note:
    this is to be replaced with a transformation matrix?
    
    """
    if type(shifts) is not np.ndarray:
        shifts = np.array(shifts)
    
    dists = distance_matrix(coords_a, translate(coords_b, -shifts))
    dists = np.sort(dists, axis=1) # <- sort to nearest distance
    return np.sum(dists[:,0])**2 # <- this makes only the nearest distance count
    
def predict_transform(coords_a, coords_b, mode='translation'):
    if mode == "translation":
        p0 = np.zeros(2)
    if mode == "translation-rotation":
        p0 = np.zeros(5)
        p0[3:] = np.average(coords_b, axis=0)
    pfit = minimize(nearest_dist_ssq, x0=p0, args=(coords_a, coords_b, mode), method='Nelder-Mead')
    return pfit.x

def nearest_dist_ssq(params, coords_a, coords_b, mode='translation'):
    if mode == "translation":
        coords_b = translate(coords_b, -params)

    if mode == "translation-rotation":
        coords_b = rotate(coords_b, params[2], params[3:])
        coords_b = translate(coords_b, -params[:2])

    dists = distance_matrix(coords_a, coords_b)
    dists = np.sort(dists, axis=1) # <- sort to nearest distance
    return np.sum(dists[:,0])**2 # <- this makes only the nearest distance count


def register_coordinates(Coords, mode="translation-rotation"):
    """ pipeline function """

    Coords_reg = [Coords[0]]
    transform_preds = []
    n_sessions = len(Coords)

    for i in range(1, n_sessions):
        transform_pred = predict_transform(Coords[0], Coords[i], mode=mode) # [0] is first session
        if mode == "translation":
            coords_reg = translate(Coords[i], -transform_pred)
        if mode == "translation-rotation":
            coords_reg = rotate(Coords[i], transform_pred[2], transform_pred[3:])
            coords_reg = translate(coords_reg, -transform_pred[:2])

        Coords_reg.append(coords_reg)
        transform_preds.append(transform_pred)

    return Coords_reg, transform_preds

"""
 
 #### ##    ## ########  ######## ##     ## 
  ##  ###   ## ##     ## ##        ##   ##  
  ##  ####  ## ##     ## ##         ## ##   
  ##  ## ## ## ##     ## ######      ###    
  ##  ##  #### ##     ## ##         ## ##   
  ##  ##   ### ##     ## ##        ##   ##  
 #### ##    ## ########  ######## ##     ## 
 
"""

def predict_indices(coords_a, coords_b_reg):
    """
    TODO doc me properly
    """

    # retrieve corresponding indices
    # Bs = B - pred_shift
    # the entire Bs thing is to distinguish Bs from B (which is next session, registered to this session)
    
    # distances from B back to A
    dists = distance_matrix(coords_b_reg, coords_a)

    # retrieve mapping
    ix_pred = np.argsort(dists, axis=1)[:,0]

    return ix_pred


def find_split(samples_a, samples_b):
    """
    TODO doc me properly
    for two distributions, find the best point to seperate them
    logistic regression
    """

    LogReg = LogisticRegression()
    X = np.concatenate([samples_a, samples_b])[:,np.newaxis]
    y = np.concatenate([np.zeros(samples_a.shape[0]), np.ones(samples_b.shape[0])])
    LogReg.fit(X,y)
    return (-LogReg.intercept_[0] / LogReg.coef_[0])[0]

def estimate_dtol(coords_a, coords_b_reg):
    """ 
    TODO 
    doesn'st KDE anymore
    DOC ME
    
    """

    dists = distance_matrix(coords_b_reg, coords_a)
    dists = np.sort(dists, axis=1)
    k = dists[:,0] # the closest distances 
    # l = dists[:,1:].flatten() # <- all others
    l = dists[:,1] # <- the second closest
    return find_split(k,l)

def unassign_indices(coords_a, coords_b, ix_pred, dtol):
    """ 
    doc me precisely
    if ix_pred points to a cell that in the previous session is larger than
    dtol from the current session, assign to -1 
    """

    # 
    dists = distance_matrix(coords_b, coords_a)
    ix = np.argmin(dists, axis=1)

    # deselect if distance larger than dtol
    euclid_dist = np.sqrt(np.sum((coords_b - coords_a[ix])**2, axis=1))
    bad_ix = np.where(euclid_dist > dtol)[0]

    # here - copy necessary??
    ix_pred_ds = copy(ix_pred)
    ix_pred_ds[bad_ix] = -1

    return ix_pred_ds

def assign_IDs(ixs):
    """
    TODO
    doc me - this is the core of the algorithm

    ixs is either true or pred - what is meant by this
    really ??? understand this again
    
    ixs is a list of sessions
    each session a list of indices pointing to the previous closest

    IDs is a list of sessions
    with each ROI being assigned an ID
    """

    n_sessions = len(ixs)

    # ini
    IDs = [ixs[0].astype('U')]
    max_id = IDs[0].shape[0]-1

    # iterate
    for i in range(1, n_sessions):
        ids = []
        for j in ixs[i]:
            if j >= 0 and j < IDs[i-1].shape[0]:
                # j >= 0 filters out all unassigned
                # j < IDs[i-1].shape[0] the number of ROIs in the previous session
                ids.append(IDs[i-1][j])
            else:
                ids.append(str(max_id+1)) # make a new one
                max_id += 1

        IDs.append(np.array(ids))
    return IDs

"""
 
  ######  ##     ##    ###    #### ##    ## #### ##    ##  ######   
 ##    ## ##     ##   ## ##    ##  ###   ##  ##  ###   ## ##    ##  
 ##       ##     ##  ##   ##   ##  ####  ##  ##  ####  ## ##        
 ##       ######### ##     ##  ##  ## ## ##  ##  ## ## ## ##   #### 
 ##       ##     ## #########  ##  ##  ####  ##  ##  #### ##    ##  
 ##    ## ##     ## ##     ##  ##  ##   ###  ##  ##   ### ##    ##  
  ######  ##     ## ##     ## #### ##    ## #### ##    ##  ######   
 
"""

def get_chain_ix(IDs, Coords, Roi_shapes=None):
    """
    TODO DOC me better
   
    # get indices of candidates for a given session - this is to extract 
    # either Roi_shapes or Ds has to be provided

    # IDs is a list of sessions
    # with each ROI being assigned an ID

    the final result
    ix_chains can be used to index into suite2p output
    """


    # consider only IDs that are present in the last session
    # = that are part of complete chains
    # find those in previous sessions
    candidate_ids = [i for i in IDs[-1] if i in IDs[0]]

    # get ix for each session where these cells are
    ix_chains = []
    
    n_sessions = len(IDs)
    for d in range(n_sessions):
        e_ix = []
        for i in candidate_ids: # only check complete
            ix = np.where(IDs[d] == i)[0]
            if ix.shape[0] == 1: # if there is only one ID matching
                e_ix.append(ix[0])
            else: # if ID is present multiple times ()
                if Roi_shapes is not None:
                    # if roi_shapes provided, choose by template matching
                    # TODO shouldn't this be Roi_shapes[-1][int(i)] 
                    # because last day defines the candidates
                    scores = [tm_score_rois(Roi_shapes[0][int(i)], Roi_shapes[d][j]) for j in ix]
                else: # fallback to spatially closest
                    scores = [euclidean(Coords[0][int(i)], Coords[d][j]) for j in ix]
                e_ix.append(ix[np.argmax(scores)])

        ix_chains.append(e_ix)

    return np.array(ix_chains, dtype='int32')

def find_chains(Coords, Roi_shapes=None):
    """ pipeline function

    takes (registered) coordinates, 
    runs the entire shebang until it returns 
    the ix_chains for each session """

    # Coords Coordinates of each session, registered to session 0

    # infer
    n_sessions = len(Coords)

    # initialize
    ixs_pred = [np.arange(Coords[0].shape[0])]

    # the chaining loop
    for i in range(1, n_sessions):
        ix_pred = predict_indices(Coords[i-1], Coords[i])
        dtol = estimate_dtol(Coords[i-1], Coords[i])
        ix_pred_mod = unassign_indices(Coords[i-1], Coords[i], ix_pred, dtol)
        ixs_pred.append(ix_pred_mod)

    # assing IDs
    IDs = assign_IDs(ixs_pred)

    # get the chain_ids
    ix_chains =  get_chain_ix(IDs, Coords, Roi_shapes=Roi_shapes)
    return ix_chains

"""
 
 ########   #######  ####     ######  ##     ##    ###    ########  ########  ######  
 ##     ## ##     ##  ##     ##    ## ##     ##   ## ##   ##     ## ##       ##    ## 
 ##     ## ##     ##  ##     ##       ##     ##  ##   ##  ##     ## ##       ##       
 ########  ##     ##  ##      ######  ######### ##     ## ########  ######    ######  
 ##   ##   ##     ##  ##           ## ##     ## ######### ##        ##             ## 
 ##    ##  ##     ##  ##     ##    ## ##     ## ##     ## ##        ##       ##    ## 
 ##     ##  #######  ####     ######  ##     ## ##     ## ##        ########  ######  
 
"""

def load_roi_shapes(stats, w=8, npx=(512,512)):
    """
    """

    roi_shapes = []
    for i, stat in enumerate(stats):
        # make full image and then cutout around med by w
        I = np.zeros(npx)
        I[stat['xpix'],stat['ypix']] = stat['lam']
        roi_shape = I[stat['med'][1]-w:stat['med'][1]+w,stat['med'][0]-w:stat['med'][0]+w]
        
        # this takes care of cells too close to the border
        # and since the template matching is done in 2d
        # everything is robust against small translations
        if roi_shape.shape != (w*2,w*2): 
            R = copy(roi_shape)
            roi_shape = np.zeros((w*2,w*2))
            roi_shape[:R.shape[0],:R.shape[1]] = R

        roi_shapes.append(roi_shape[np.newaxis,:,:])
    return np.concatenate(roi_shapes)

def tm_score_rois(roi_a, roi_b):
    """ using sklearns template matching and returns best template match score """
    tm_res = match_template(roi_a, roi_b, pad_input=True)
    k,l = np.unravel_index(np.argmax(tm_res.flatten()), tm_res.shape)
    return tm_res[k,l]

def calc_tm_score_dist(Roi_shapes_chains):
    """
    this function computes:
    the chance level based on the split between the tm scores
    obtained from the chains, vs randomly pairing roi shapes
    """
    
    R = np.stack(Roi_shapes_chains)
    n_sessions = R.shape[0]
    session_combos = list(combinations(range(n_sessions),2))
    n_cells = R.shape[1]

    # full
    tm_scores = []
    for a,b in session_combos:
        for i in range(n_cells):
            tm_scores.append(tm_score_rois(R[a,i,:,:], R[b,i,:,:]))
    tm_scores = np.array(tm_scores)

    # a random samples
    n_samples = len(tm_scores)
    Rc = np.concatenate(Roi_shapes_chains)
    ix_a = np.random.randint(Rc.shape[0], size=n_samples)
    ix_b = np.random.randint(Rc.shape[0], size=n_samples)

    tm_scores_rand = []
    for i,j in zip(ix_a, ix_b):
        tm_scores_rand.append(tm_score_rois(Rc[i,:,:], Rc[j,:,:]))
    tm_scores_rand = np.array(tm_scores_rand)

    tm_chance_lvl = find_split(tm_scores, tm_scores_rand)
    return tm_chance_lvl, tm_scores, tm_scores_rand

def evaluate_roi_chains(Roi_shapes_chains, tm_chance_lvl):
    """ DOC ME
    pipeline function """
    Roi_shapes_chains = np.stack(Roi_shapes_chains)
    n_sessions, n_cells, _, _ = Roi_shapes_chains.shape
    combos = list(combinations(range(n_sessions),2))

    pairwise_tm_scores = np.zeros((n_cells, len(combos)))

    for i in range(n_cells): # for each cell
        for j, combo in enumerate(combos): # for each combination of sessions
            # compute ROI tm score
            pairwise_tm_scores[i,j] = tm_score_rois(Roi_shapes_chains[combo[0], i, :, :], Roi_shapes_chains[combo[1], i, :, :])

    # only accept if TM score is consistently higher than threshold (= tm_chance_lvl)
    ix_good = np.where(np.all(pairwise_tm_scores > tm_chance_lvl,axis=1))[0]
    return ix_good

"""
 
  #######  ##     ## ######## ########  ##     ## ######## 
 ##     ## ##     ##    ##    ##     ## ##     ##    ##    
 ##     ## ##     ##    ##    ##     ## ##     ##    ##    
 ##     ## ##     ##    ##    ########  ##     ##    ##    
 ##     ## ##     ##    ##    ##        ##     ##    ##    
 ##     ## ##     ##    ##    ##        ##     ##    ##    
  #######   #######     ##    ##         #######     ##    
 
"""

def save_output(folders, ix_chains_good, good_ratio=0.5, extra_fnames=None):
    """ 
    saving the output
    """

    for i,folder in enumerate(folders):

        print("saving extracted signals for %s" % folder)
        os.makedirs(folder / "chain2p", exist_ok=True)
        stats, ix_sel = load_stats(folder, good_ratio=good_ratio)
        
        # stats
        stats_chain = [stats[j] for j in ix_chains_good[i]] 
        np.save(folder / "chain2p" / "stat.npy", np.array(stats_chain, dtype='object'))
        
        # iscell
        iscell = np.load(folder / "iscell.npy")[ix_sel,:]
        np.save(folder / "chain2p" / "iscell.npy", iscell[ix_chains_good[i],:])

        # apply to data
        fnames = ["F.npy", "Fneu.npy", "spks.npy"] # suite2p defaults
        if extra_fnames is not None:
            [fnames.append(f) for f in extra_fnames]

        for fname in fnames:
            data = np.load(folder / fname)
            data_sel = data[ix_sel,:]
            data_chain = data_sel[ix_chains_good[i],:]
            np.save(folder / "chain2p" / fname, data_chain)


"""
 
 ########  #### ########  ######## ##       #### ##    ## ######## 
 ##     ##  ##  ##     ## ##       ##        ##  ###   ## ##       
 ##     ##  ##  ##     ## ##       ##        ##  ####  ## ##       
 ########   ##  ########  ######   ##        ##  ## ## ## ######   
 ##         ##  ##        ##       ##        ##  ##  #### ##       
 ##         ##  ##        ##       ##        ##  ##   ### ##       
 ##        #### ##        ######## ######## #### ##    ## ######## 
 
"""
from plotters import *

def chain2p_pipeline(folders, params):
    """
    run the complete pipeline on a list of folders
    """


    # parameters
    good_ratio = params['good_ratio']
    n_xpx, n_ypx = params['image_shape_px']
    w = params['soma_size_px']

    # load suite2p soma coordiantes and stats
    Coords, Stats = load_data(folders, good_ratio=good_ratio)

    # normalize coordinates
    n_sessions = len(Coords)
    for d in range(n_sessions):
        Coords[d][:,0] /= n_xpx
        Coords[d][:,1] /= n_ypx

    # load ROIs
    Roi_shapes = [load_roi_shapes(stats, w=w, npx=(n_xpx,n_ypx)) for stats in Stats] # this is across sessions

    # registration
    Coords_reg, reg_params = register_coordinates(Coords, mode='translation-rotation')

    # chaining
    ix_chains = find_chains(Coords_reg, Roi_shapes)

    # extraction by chains
    Coords_chains = [Coords_reg[d][ix_chains[d]] for d in range(n_sessions)]
    Roi_shapes_chains = [np.array(Roi_shapes[d])[ix_chains[d]] for d in range(n_sessions)]

    # and calculating chance level based on those
    # motivation for the two passes:
    # first pass, chains will contain all kind of errors
    # after the first pass, the split should lead to a good separation
    # and the tm_chance_lvl from the first pass can then be applied to all data
    # why not more than 2: 

    Roi_shapes_chains_good = copy(Roi_shapes_chains)

    for i in range(2):
        tm_chance_lvl, tm_scores, tm_scores_rand = calc_tm_score_dist(Roi_shapes_chains_good)
        ix_good = evaluate_roi_chains(Roi_shapes_chains, tm_chance_lvl)
        print(tm_chance_lvl, ix_good.shape[0])
        ix_chains_good = [ix_chains[d][ix_good] for d in range(n_sessions)]
        Roi_shapes_chains_good = [np.array(Roi_shapes[d])[ix_chains_good[d]] for d in range(n_sessions)]

    # cleanup by using Roi information
    Coords_chains_good = [Coords_reg[d][ix_chains_good[d]] for d in range(n_sessions)]

    # some diagnostic output
    
    for i,folder in enumerate(folders):
        plot_folder = folder / "chain2p" / "plots"
        os.makedirs(plot_folder, exist_ok=True)

        # plotting ROI shapes
        # plot_all_Roi_shapes(np.stack(Roi_shapes_chains), n_cells_per_figure=10, normalize=True, outpath=plot_folder / "Roi_shapes.png")
        
        # plotting the template matching score distributions
        _, tm_scores, tm_scores_rand = calc_tm_score_dist(Roi_shapes_chains)
        plot_tm_scores(tm_scores, tm_scores_rand, tm_chance_lvl, outpath=plot_folder / "tm_scores.png")

        # Coordinates
        plot_coordinates(Coords_reg, Coords_chains, Coords_chains_good, outpath=plot_folder / "Coordinates.png")
        