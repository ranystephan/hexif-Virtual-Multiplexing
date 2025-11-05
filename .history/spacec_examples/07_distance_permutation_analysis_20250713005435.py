# %% [markdown]
# # SPACEc: Distance Permutation Analysis

# %%
import spacec as sp

#import standard packages
import os
import pandas as pd
import scanpy as sc

# silencing warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# set plotting parameters
sc.settings.set_figure_params(dpi=80, facecolor='white')

# %%
# Specify the path to the data
root_path = "/home/user/path/SPACEc/" # inset your own path

data_path = root_path + 'example_data/raw/' # where the data is stored

# where you want to store the output
output_dir = root_path + 'example_data/output/'
os.makedirs(output_dir, exist_ok=True)

# %%
# Load data
adata = sc.read(output_dir + "adata_nn_demo_annotated_cn.h5ad")
adata

# %% [markdown]
# ## Identify potential interactions

# %%
# compute the potential interactions
distance_pvals, results_dict = sp.tl.identify_interactions(
    adata = adata, # AnnData object
    cellid = "index", # column that contains the cell id (set index if the cell id is the index of the dataframe)
    x_pos = "x", # x coordinate column
    y_pos = "y", # y coordinate column
    cell_type = "cell_type", # column that contains the cell type information
    region = "unique_region", # column that contains the region information
    num_iterations=1000, # number of iterations for the permutation test
    num_cores=12,  # number of CPU threads to use
    min_observed = 10, # minimum number of observed interactions to consider a cell type pair
    comparison = 'condition', # column that contains the condition information we want to compare
    distance_threshold=20/0.5085) # distance threshold in px (20 µm)


# %%
# the results_dict contains the results of the permutation test as well as the observed and shuffled distances
results_dict.keys()

# %%
# the distance_pvals contains the p-values for each cell type pair and is automatically added to the adata.uns
adata.uns['triDist']

# %%
# save adata
adata.write(output_dir + "adata_nn_demo_annotated_cn.h5ad")

# %% [markdown]
# ## Filter for most significant results

# %% [markdown]
# In this example the results are filtered twice. First to remove rare cell types from the analysis because they are overrepresented when comparing distances, and then we filter on statistical significance as well as absolute log fold change.

# %%
distance_pvals_filt = sp.tl.remove_rare_cell_types(adata, 
                       distance_pvals, 
                       cell_type_column="cell_type", 
                       min_cell_type_percentage=1)

# %%
# Identify significant cell-cell interactions
# dist_table_filt is a simplified table used for plotting
# dist_data_filt contains the filtered raw data with more information about the pairs
#  The function outputs two dataframes:  and dist_data_filt that contains all filtered interactions and  dist_table_filt that contains a table for all interactions that show a significant value in both tissues
dist_table_filt, dist_data_filt = sp.tl.filter_interactions(
    distance_pvals = distance_pvals_filt,
    pvalue = 0.05,
    logfold_group_abs = 0.1,
    comparison = 'condition')

print(dist_table_filt.shape)
dist_data_filt

# %% [markdown]
# ## Visualization

# %%
sp.pl.plot_top_n_distances(
    dist_table_filt,
    dist_data_filt,
    n=5,
    colors=None,
    dodge=False,
    savefig=False,
    output_fname="",
    output_dir="./",
    figsize=(5, 5),
    unit="px",
    errorbars=True,
)

# %%
sp.pl.dumbbell(data = dist_table_filt, figsize=(8,12), colors = ['#DB444B', '#006BA2'])

# %%
sp.pl.distance_graph(dist_table = dist_data_filt, # the (filtered) distance data table you want to plot 
                  distance_pvals = distance_pvals, # the full distance data table
                  condition_pair=['tonsil', 'tonsillitis'],
                  node_size=1600, font_size=6,
                  palette=None,
                  dpi = 600,
                  savefig=False,
                  output_fname="",
                  output_dir=output_dir,)


