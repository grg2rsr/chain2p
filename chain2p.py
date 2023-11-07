# %%
# imports
%load_ext autoreload
%autoreload 2
%matplotlib qt5
import sys, os
from pathlib import Path

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from scipy.stats import gaussian_kde as kde

from skimage.feature import match_template

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 330
import seaborn as sns

from tqdm import tqdm
from itertools import combinations
from copy import copy


# %%
"""
 
  ######  ######## ##     ## ####    ########  ########    ###    ##       
 ##    ## ##       ###   ###  ##     ##     ## ##         ## ##   ##       
 ##       ##       #### ####  ##     ##     ## ##        ##   ##  ##       
  ######  ######   ## ### ##  ##     ########  ######   ##     ## ##       
       ## ##       ##     ##  ##     ##   ##   ##       ######### ##       
 ##    ## ##       ##     ##  ##     ##    ##  ##       ##     ## ##       
  ######  ######## ##     ## ####    ##     ## ######## ##     ## ######## 
 
"""
folders = []
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05472/2023-03-10_JJP-05472_10/splits_n4_r1/raw_1/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05472/2023-03-10_JJP-05472_10/splits_n4_r1/raw_2/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05472/2023-03-10_JJP-05472_10/splits_n4_r1/raw_3/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05472/2023-03-10_JJP-05472_10/splits_n4_r1/raw_4/suite2p/plane0"))
n_sessions = len(folders)


"""
 
 ########  ########    ###    ##          ########     ###    ########    ###    
 ##     ## ##         ## ##   ##          ##     ##   ## ##      ##      ## ##   
 ##     ## ##        ##   ##  ##          ##     ##  ##   ##     ##     ##   ##  
 ########  ######   ##     ## ##          ##     ## ##     ##    ##    ##     ## 
 ##   ##   ##       ######### ##          ##     ## #########    ##    ######### 
 ##    ##  ##       ##     ## ##          ##     ## ##     ##    ##    ##     ## 
 ##     ## ######## ##     ## ########    ########  ##     ##    ##    ##     ## 
 
"""

# %% wasabi
folders = []
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05425/2023-03-08_JJP-05425_8/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05425/2023-03-09_JJP-05425_9/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05425/2023-03-10_JJP-05425_10/suite2p/plane0"))
n_sessions = len(folders)

# %% pepper
folders = []
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05472/2023-03-08_JJP-05472_8/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05472/2023-03-09_JJP-05472_9/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/JJP-05472/2023-03-10_JJP-05472_10/suite2p/plane0"))
n_sessions = len(folders)


### the second run!
# %% wasabi
folders = []
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/2023-05-09_JJP-05425_12/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/2023-05-10_JJP-05425_13/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/2023-05-11_JJP-05425_14/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/2023-05-12_JJP-05425_15/suite2p/plane0"))
n_sessions = len(folders)

# %% pepper
folders = []
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/2023-05-09_JJP-05472_12/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/2023-05-10_JJP-05472_13/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/2023-05-11_JJP-05472_14/suite2p/plane0"))
folders.append(Path("/media/georg/htcondor/shared-paton/georg/Animals_smelling_imaging_sessions_2/2023-05-12_JJP-05472_15/suite2p/plane0"))
n_sessions = len(folders)

# %% build pipeline function

params = dict(good_ratio=0.5, soma_size_px=8, image_shape_px=(512,512))
from chain2plib import *
chain2p_pipeline(folders, params)

# %%
