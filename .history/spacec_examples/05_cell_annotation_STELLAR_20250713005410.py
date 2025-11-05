# %% [markdown]
# # SPACEc: ML-enabled cell type annotation - STELLAR

# %% [markdown]
# After preprocessing the single-cell data, the next step is to assign cell types. Alternatively to the SVM (see notebook 3_cell_annotation_ml) model we included a wrapper for STELLAR, that allows to use the model in a more user-friendly way. Further information about STELLAR can be found here: http://snap.stanford.edu/stellar/

# %%
# import spacec first
import spacec as sp

#import standard packages
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import sys
import os
from git import Repo
import anndata

# silencing warnings
import warnings
warnings.filterwarnings('ignore')

sc.settings.set_figure_params(dpi=80, facecolor='white')

# %%
# Specify the path to the data
root_path = "/home/user/path/SPACEc/" # inset your own path
data_path = root_path + 'example_data/raw/' # where the data is stored

# where you want to store the output
output_dir = root_path + 'example_data/output/'
os.makedirs(output_dir, exist_ok=True)

# STELLAR path
STELLAR_path = Path(root_path + 'example_data/STELLAR/')

# Test if the path exists, if not create it
if not STELLAR_path.exists():
    STELLAR_path.mkdir(exist_ok=True, parents=True)
    repo_url = 'https://github.com/snap-stanford/stellar.git'
    Repo.clone_from(repo_url, STELLAR_path)

# %% [markdown]
# ## Data Explanation
# Annotated tonsil data is used as training & test data. </br>
# Tonsillitis data is used as validation data.

# %%
# Load training data
adata = sc.read(output_dir + "adata_nn_demo_annotated.h5ad")
adata_train = adata[adata.obs['condition'] == 'tonsil']
adata_val  = adata[adata.obs['condition'] == 'tonsillitis']

# %% [markdown]
# ## Training

# %%
import numpy as np
np.isnan(adata_train.X).sum()

# %%
# downsample the data for demonstration purposes
adata_train = adata_train[0:1000, :]
adata_val = adata_val[0:1000, :]

# %%
adata_new = sp.tl.adata_stellar(adata_train, 
               adata_val, 
               celltype_col = "cell_type", 
               x_col = 'x', 
               y_col = 'y', 
               sample_rate = 0.5, 
               distance_thres = 50,
               STELLAR_path = STELLAR_path)

# %% [markdown]
# ## Inspect the results

# %%
adata_new.obs

# %%
sc.pl.umap(adata_new, color = 'stellar_pred')

# %%
marker_list = [
    'FoxP3', 'HLA-DR', 'EGFR', 'CD206', 'BCL2', 'panCK', 'CD11b', 'CD56', 'CD163', 'CD21', 'CD8', 
    'Vimentin', 'CCR7', 'CD57', 'CD34', 'CD31', 'CXCR5', 'CD3', 'CD38', 'LAG3', 'CD25', 'CD16', 'CLEC9A', 'CD11c', 
    'CD68', 'aSMA', 'CD20', 'CD4','Podoplanin', 'CD15', 'betaCatenin', 'PAX5', 
    'MCT', 'CD138', 'GranzymeB', 'IDO-1', 'CD45', 'CollagenIV', 'Arginase-1']

sc.pl.dotplot(adata_new, marker_list, 'stellar_pred', dendrogram = True)

# %% [markdown]
# ## Single-cell visualzation

# %%
sp.pl.catplot(
    adata_new, color = "stellar_pred", # specify group column name here e.g. celltype_fine)
    unique_region = "condition", # specify unique_regions here
    X='x', Y='y', # specify x and y columns here
    n_columns=1, # adjust the number of columns for plotting here (how many plots do you want in one row?)
    palette=None, #default is None which means the color comes from the anndata.uns that matches the UMAP
    savefig=False, # save figure as pdf
    output_fname = "", # change it to file name you prefer when saving the figure
    output_dir=output_dir, # specify output directory here (if savefig=True)
)

# %%
sp.pl.catplot(
    adata_new, color = "cell_type", # specify group column name here e.g. celltype_fine)
    unique_region = "condition", # specify unique_regions here
    X='x', Y='y', # specify x and y columns here
    n_columns=1, # adjust the number of columns for plotting here (how many plots do you want in one row?)
    palette=None, #default is None which means the color comes from the anndata.uns that matches the UMAP
    savefig=False, # save figure as pdf
    output_fname = "", # change it to file name you prefer when saving the figure
    output_dir=output_dir,) # specify output directory here (if savefig=True)


