# %% [markdown]
# # SPACEc: Preprocessing - Signal Preprocessing

# %%
# import spacec first
import spacec as sp

# import standard packages
import os
import numpy as np
import pandas as pd

# silencing warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# Specify the path to the data
root_path = "/home/user/path/SPACEc/" # inset your own path
data_path = root_path + 'example_data/raw/' # where the data is stored

# where you want to store the output
output_dir = root_path + 'example_data/output/'
os.makedirs(output_dir, exist_ok=True)

# %%
#read in segmentation csv files
#Read and concatenate the csv files (outputs from the cell segmentation algorithms). 
df_seg = sp.pp.read_segdf(
    segfile_list = [ # list of segmented files
        output_dir + "tonsil1_mesmer_result.csv", 
        output_dir + "tonsil2_mesmer_result.csv"
    ],
    seg_method = 'mesmer',
    region_list =["reg001", "reg002"],
    meta_list = ["tonsil", "tonsillitis"]
)

#Get the shape of the data
print(df_seg.shape)

#See what it looks like
df_seg.head()

# %% [markdown]
# ## Filter cells by DAPI intensity and area

# %%
# print smallest 1% of cells by area
one_percent_area = np.percentile(df_seg.area, 1)
one_percent_area

# %%
# print smallest 1% of cells by DAPI intensity
one_percent_nuc = np.percentile(df_seg.DAPI, 1)
one_percent_nuc

# %%
# If necessary filter the dataframe to remove too small objects or cells without a nucleus. 
# Identify the lowest 1% for cell size and nuclear marker intensity to get a better idea of potential segmentation artifacts.
df_filt = sp.pp.filter_data(
    df_seg, 
    nuc_thres=one_percent_nuc, # remove cells with DAPI intensity below threshold
    size_thres=one_percent_area, # remove cells with area below threshold
    nuc_marker="DAPI", # name of nuclear marker
    cell_size = "area", # name of cell size column
    region_column = "region_num", # column with region numbers
    color_by = "region_num", # color by region number
    log_scale=False) # log scale for size

# %% [markdown]
# ## Normalize data

# %%
# Normalize data with one of the four available methods (zscore as default)
df_filt.columns

# %%
# This is to normalize the data per region/tif
dfz = pd.DataFrame()

for region in df_filt.unique_region.unique():
    df_reg = df_filt[df_filt.unique_region == region]
    df_reg_norm = sp.pp.format(
        data=df_reg, 
        list_out= ['eccentricity', 'perimeter', 'convex_area', 'axis_major_length', 'axis_minor_length',  "label"], # list of features to remove
        list_keep = ["DAPI",'x','y', 'area','region_num',"unique_region", 'condition'], # list of meta information that you would like to keep but don't want to normalize
        method = "zscore") # choose from "zscore", "double_zscore", "MinMax", "ArcSin"
    dfz = pd.concat([dfz,df_reg_norm], axis = 0)

dfz.shape

# %% [markdown]
# ## Remove noisy cells

# %%
#This section is used to remove noisy cells. This is very important to ensure proper identification of the cells via clustering.
dfz.columns

# %%
# get the column index for the last marker 
col_num_last_marker = dfz.columns.get_loc('GATA3')
print(col_num_last_marker)

# %%
# This function helps to figure out what the cut-off should be for each region
for region in dfz.unique_region.unique():
    print(region)
    df_reg = dfz[dfz.unique_region == region]
    sp.pl.zcount_thres(dfz = df_reg, 
                col_num = col_num_last_marker, # last antibody index
                cut_off=0.01, #top 1% of cells
                count_bin=50)


# %%
# This is to remove top 1 % of all cells that are highly expressive for all antibodies
df_nn = pd.DataFrame()
cutoff_list = [[41, 38.38], [45, 46.22]]

for i in range(len(dfz.unique_region.unique())):
    df_reg = dfz[dfz.unique_region == dfz.unique_region.unique()[i]]
    df_reg_nn,cc = sp.pp.remove_noise(
        df=df_reg, 
        col_num=col_num_last_marker, # this is the column index that has the last protein feature
        z_count_thres=cutoff_list[i][0], # number obtained from the function above
        z_sum_thres=cutoff_list[i][1] # number obtained from the function above
    )
    print(df_reg_nn.shape)
    df_nn = pd.concat([df_nn,df_reg_nn], axis = 0)
df_nn.shape

# %%
#Save the df as a backup. We strongly recommend the Anndata format for further analysis!
df_nn.to_csv(output_dir + "df_nn_demo.csv")

# %%
# inspect which markers work, and drop the ones that did not work from the clustering step
# make an anndata to be compatible with the downstream clustering step
adata = sp.hf.make_anndata(
    df_nn = df_nn,
    col_sum = col_num_last_marker, # this is the column index that has the last protein feature # the rest will go into obs
    nonFuncAb_list = [] # Remove the antibodies that are not working from the clustering step
)
adata

# %%
# save the anndata object to a file
adata.write_h5ad(output_dir + 'adata_nn_demo.h5ad')

# %% [markdown]
# ## Show the spatial distribution for size (Optional)

# %%
import pickle
with open(output_dir + 'overlay_tonsil1.pickle', 'rb') as f:
    overlay_data1 = pickle.load(f)

with open(output_dir + 'overlay_tonsil2.pickle', 'rb') as f:
    overlay_data2 = pickle.load(f)

# %%
df_nn.columns

# %%
sp.pl.coordinates_on_image(
    df = df_nn.loc[df_nn['unique_region'] == 'reg001',:], 
    overlay_data = overlay_data1, color='area',  
    scale=False, # whether to scale to 1 or not
    dot_size=5,
    convert_to_grey=True, 
    fig_width=10, fig_height=10)

# %%
sp.pl.coordinates_on_image(
    df = df_nn.loc[df_nn['unique_region'] == 'reg002',:], 
    overlay_data = overlay_data2, 
    color='area', 
    scale=False, # whether to scale to 1 or not
    dot_size=5,
    convert_to_grey=True, 
    fig_width=10, fig_height=10 )

# %% [markdown]
# This function can also be used to inspect where certain markers are expressed in the tissue.

# %%
import matplotlib.pyplot as plt
plt.rc('axes', grid=False)  # remove gridlines

# %%
marker_list = ['EGFR', 'CD21', 'CD8']

for marker in marker_list:
    sp.pl.coordinates_on_image(
        df = df_nn.loc[df_nn['unique_region'] == 'reg002',:], 
        overlay_data = overlay_data2, 
        color=marker, 
        scale=False, # whether to scale to 1 or not
        dot_size=2,
        convert_to_grey=True, 
        fig_width=3, fig_height=3 )


