# %% [markdown]
# # SPACEc: Patch Proximity Analysis

# %% [markdown]
# Patch proximity analysis (PPA) analyses neighborhoods as patches of closely connected cells. The goal of the analysis is to analyze was surrounds these patches within a user defined radius. In our example we will use PPA to identify germinal centers as CN patches and then analyze what surrounds them based on the tissue condition (tonsil vs. tonsillitis). 

# %%
# import spacec
import spacec as sp

#import standard packages
import os
import scanpy as sc

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

# %% [markdown]
# Load the anndata object that contains the previously generated CN annotations. 

# %%
# read in the annotated anndata
adata = sc.read(output_dir + 'adata_nn_demo_annotated_cn.h5ad')
adata

# %% [markdown]
# Setting the plotting parameter to True illustrates what the PPA function is detecting. This might be helpful if the min cluster size is unknown or the user wants to perform quality control. The results are stored as df in the adata.uns slot.

# %% [markdown]
# ## Compute for proximal cells

# %% [markdown]
# Patch proximity analysis can be executed using two methods that detect patches by applying a concave hull to a selected neighborhood. The first method, 'border_cell_radius', identifies the cells outlining the hull along with their nearest neighbors. It then draws a circle with a specified radius around these cells to detect those in spatial proximity, while ignoring cells that belong to the same patch. This approach results in a highly precise border but can be more computationally intensive than the 'hull_expansion' method. Additionally, the 'border_cell_radius' method is sensitive to internal borders, effectively identifying holes within a patch if the radius extends over them.

# %%
# this region result is also saved to adata.uns
results, outlines_results = sp.tl.patch_proximity_analysis(
    adata, # the annotated adata object
    region_column = "unique_region", # column with the region information
    patch_column = "CN_k20_n6_annot", # column with the patch information (derive patches from this column)
    group="Germinal Center", # group to consider
    min_cluster_size=50, # minimum cluster size to consider
    x_column='x', y_column='y', # spatial coordinates
    radius = [20, 40, 60, 80, 100], # to get the distance in µm
    edge_neighbours = 1, # number of neighbours to consider for edge detection if set to 1 only the hull is considered
    plot = True, # plot the results for demonstration and/or documentation (set to False to skip plotting - improves speed)
    original_unit_scale = 1.96656, # scale factor for the units (1 = 1px per unit e.g. µm)
    method= "border_cell_radius", # method to use for the edge detection
    key_name = "ppa_result_20_40_60_80_100_border_cell_radius", # key name to store the result in adata.uns
    save_geojson = False, # save the results as geojson
    ) # plot detection for demonstration purposes

# %% [markdown]
# On the other hand, we provide the hull_expansion method. It runs faster and works well when patches are evenly filled with cells. This method takes the detected hull and expands the polygon outward by a specified radius. For more fragmented patches, we recommend using the border_cell_radius method described above.

# %%
# this region result is also saved to adata.uns
results, outlines_results = sp.tl.patch_proximity_analysis(
    adata, # the annotated adata object
    region_column = "unique_region", # column with the region information
    patch_column = "CN_k20_n6_annot", # column with the patch information (derive patches from this column)
    group="Germinal Center", # group to consider
    min_cluster_size=50, # minimum cluster size to consider
    x_column='x', y_column='y', # spatial coordinates
    radius = [20,40, 60, 80, 100], # to get the distance in µm
    edge_neighbours = 1, # number of neighbours to consider for edge detection if set to 1 only the hull is considered
    plot = True, # plot the results for demonstration and/or documentation (set to False to skip plotting - improves speed)
    original_unit_scale = 1.96656, # scale factor for the units (1 = 1px per unit e.g. µm)
    method= "hull_expansion", # method to use for the edge detection
    key_name = "ppa_result_20_40_60_80_100_border_cell_radius", # key name to store the result in adata.uns
    save_geojson = False, # save the results as geojson
    ) # plot detection for demonstration purposes

# %% [markdown]
# The exported results dataframe stores the cells that were detected in spatial proximity to the patch. Every patch has a unique_patch_ID. The distance_from_patch column shows in which radius the cell was detected.

# %%
results

# %% [markdown]
# Outlines_results holds the coordinates of the outlining cells. Additionally, users can save the outline of the hull as a polygon in the GeoJSON format. This allows to later load the patch coordinates into an interactive viewer such as TissUUmaps.

# %%
outlines_results

# %% [markdown]
# Often it is more informative to derive the cellular content within a range of distances.

# %%
# save adata
adata.write(output_dir + 'adata_nn_demo_annotated_cn.h5ad')

# %% [markdown]
# SPACEc can visualize the PPA results as donut plot showing the percentages of cell types or CNs within a given radius around the patches. Percentages are averaged over all regions in the selected condition. The donut plot can show up to five distances.

# %% [markdown]
# ## Visualization

# %% [markdown]
# The donut plots can be plotted with two options: within or between
# - "within": Includes all cells up to the specified distance. Rings represent cumulative proportions.
# - "between": Includes only cells between the current distance ring's outer radius and the previous ring's outer radius. Rings represent proportions in discrete intervals.

# %%
# Donut plots for cell types around Germinal Center
sp.pl.ppa_res_donut(
    adata,
    cat_col = 'cell_type',
    key_name="ppa_result_20_40_60_80_100_border_cell_radius",
    palette=None,
    distance_mode="within",  # "within" or "between"
    unit="µm",
    figsize=(10, 10),
    add_guides=True,
    text="Cell types around Germinal Center",
    label_color="black",
    group_by= 'condition',
    title="PPA",
) 

# %%
# Donut plots for cell types around Germinal Center
sp.pl.ppa_res_donut(
    adata,
    cat_col = 'cell_type',
    key_name="ppa_result_20_40_60_80_100_border_cell_radius",
    palette=None,
    distance_mode="between",  # "within" or "between"
    unit="µm",
    figsize=(10, 10),
    add_guides=True,
    text="Cell types around Germinal Center",
    label_color="black",
    group_by= 'condition',
    title="PPA",
) 

# %%
# Donut plots for cell types around Germinal Center
sp.pl.ppa_res_donut(
    adata,
    cat_col = 'CN_k20_n6_annot',
    key_name="ppa_result_20_40_60_80_100_border_cell_radius",
    palette=None,
    distance_mode="within",  # "within" or "between"
    unit="µm",
    figsize=(10, 10),
    add_guides=True,
    text="Cell types around Germinal Center",
    label_color="black",
    group_by= 'condition',
    title="PPA",
) 

# %%
# Donut plots for cell types around Germinal Center
sp.pl.ppa_res_donut(
    adata,
    cat_col = 'CN_k20_n6_annot',
    key_name="ppa_result_20_40_60_80_100_border_cell_radius",
    palette=None,
    distance_mode="between",  # "within" or "between"
    unit="µm",
    figsize=(10, 10),
    add_guides=True,
    text="Cell types around Germinal Center",
    label_color="black",
    group_by= 'condition',
    title="PPA",
) 


