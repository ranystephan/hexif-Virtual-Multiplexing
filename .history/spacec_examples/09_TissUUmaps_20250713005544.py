# %% [markdown]
# # SPACEc: TissUUmaps for interactive visualization
# Interactive visualization via TissUUmaps might be informative during multiple steps of the analysis. Apart from the general function provided with the TissUUmaps Python package, we provide specific functions that automatically phrase the input during multiple steps of the analysis.

# %% [markdown]
# ## Instructions
# 
# To use the TissUUmaps viewer you need:
# - A pickle file that contains the segmentation output and images 
# - An AnnData object containing the currently used single cell data
# 
# The *tm_prepare_input* function reads the named content for one region. For that, the user has to provide a region column and a region name. The pickle file has to match the specified region. 
# The function creates a folder that contains all necessary input files that are needed to launch the TissUUmaps session. Additionally, the function can launch the TissUUmaps session. If the session is launched from the function a tmap file is created in the input directory that allows to open the session again (both from jupyter and the standalone viewer app).
# Alternatively, the function can be used to prepare the directory and the viewer can be launched separately to modify the display options in jupyter as well as host ports etc.
# 
# If the Jupyter viewer is too small (might be a problem on small monitors), the user can use the link (displayed if function is executed) to display TissUUmaps in the browser. 

# %%
# import spacec first
import spacec as sp

# silencing warnings
import warnings
warnings.filterwarnings('ignore')

#import standard packages
import os
import scanpy as sc


# %%
# Specify the path to the data
root_path = "/home/user/path/SPACEc/" # inset your own path

data_path = root_path + 'example_data/raw/' # where the data is stored

# where you want to store the output
output_dir = root_path + 'example_data/output/'

os.makedirs(output_dir, exist_ok=True)

# %%
# read in the annotated anndata
adata = sc.read(output_dir + 'adata_nn_demo_annotated_cn.h5ad')
adata

# %%
adata.obs.head()

# %% [markdown]
# ## Integrated use

# %% [markdown]
# This function allows the user to reshape the data for TissUUmaps and plot cells from a selected region on top of the original image.

# %%
#create cache direction to store tissuumaps cache
os.makedirs(output_dir + "cache", exist_ok=True)

image_list, csv_paths = sp.tl.tm_viewer(
    adata,
    images_pickle_path= output_dir + 'seg_output_tonsil2.pickle',
    directory = output_dir + "cache", # Or inset your own path where you want to cache your images for TM visualization (you can delete this once you are done with TM)
    region_column = "unique_region",
    region = "reg002",
    xSelector = "y",
    ySelector = "x",
    color_by = "cell_type",
    keep_list = None,
    open_viewer=True)

# %% [markdown]
# ## Interactive Catplot via the TissUUmaps viewer

# %% [markdown]
# This function starts a simplified version that only shows the cell centroid without the original image. It can be used for fast and interactive visualization. Different from the function above, this function allows visualizing all regions at once. 

# %%
sp.tl.tm_viewer_catplot(
    adata, # anndata object
    directory=None, # directory to save the generated csv files
    region_column="unique_region", # column with the region information (user can select region in tm viewer)
    x="x", # x coordinates
    y="y", # y coordinates
    color_by="cell_type", # cathegorical column to color by
    open_viewer=True, # open the tm viewer 
    add_UMAP=True, # add UMAP to the tm viewer for exploring the feature space along with the spatial data
    keep_list=None) # List of columns to keep from `adata.obs`. If None, defaults to [region_column, x, y, color_by]


