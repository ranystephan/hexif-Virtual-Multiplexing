# %% [markdown]
# # SPACEc: Cell Segmentation - The effect of channel selection on segmentation

# %% [markdown]
# To illustrate the effect of choosing different marker combinations on segmentation we created this brief notebook.

# %%
# import spacec first
import spacec as sp

#import standard packages
import os
import warnings
import matplotlib
import pickle
warnings.filterwarnings('ignore')

# set the default color map to viridis, the below paramters can be chanaged
matplotlib.rcParams["image.cmap"] = 'viridis'

# %%
# where you want to store the output
output_dir = "your_output" # inset your own path
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## Cell segmentation

# %% [markdown]
# Load cropped image

# %%
img_dir ="tonsil_tma_crop2.tif"
names="channelnames.txt"

# %% [markdown]
# Using CD45 and betaCatenin in combination as membrane markers covers all cells sufficiently. 

# %%
# (optional, one can just use nuclei for segmentation)
# Visualize membrane channels to use for cell segmentation 

sp.pl.segmentation_ch(
    file_name = img_dir, # image for segmentation
    channel_file = names, # all channels used for staining
    output_dir = output_dir, #
    extra_seg_ch_list = ["CD45", "betaCatenin"], #default is None; if provide more than one channel, then they will be combined
    nuclei_channel = 'DAPI', # channel to use for nuclei segmentation
    input_format = 'Multichannel', 
)

# %% [markdown]
# Using CD3 as membrane marker only covers T cells

# %%
# (optional, one can just use nuclei for segmentation)
# Visualize membrane channels to use for cell segmentation 

sp.pl.segmentation_ch(
    file_name = img_dir, # image for segmentation
    channel_file = names, # all channels used for staining
    output_dir = output_dir, #
    extra_seg_ch_list = ["CD3"], #default is None; if provide more than one channel, then they will be combined
    nuclei_channel = 'DAPI', # channel to use for nuclei segmentation
    input_format = 'Multichannel', 
)

# %% [markdown]
# Using all markers in combination gives us an uneven coverage.

# %%
# (optional, one can just use nuclei for segmentation)
# Visualize membrane channels to use for cell segmentation 

sp.pl.segmentation_ch(
    file_name = img_dir, # image for segmentation
    channel_file = names, # all channels used for staining
    output_dir = output_dir, #
    extra_seg_ch_list = [
    "FoxP3", "HLA-DR", "CD103", "CHGA", "EGFR", "CD206", "GFAP", "PD-1", "BCL2", "panCK",
    "CD45RO", "CD11b", "CD56", "CD163", "CD21", "CD8", "S100", "Vimentin", "PDGFRb", "CCR7",
    "CD57", "CD34", "Synaptophysin", "CD31", "CXCR5", "CD3", "CD38", "LAG3", "CD25", "CD16",
    "IL-10", "Ki67", "CLEC9A", "p53", "CD69", "CD11c", "CD68", "Ox40", "aSMA", "CD20", "CD4",
    "MUC-1", "Podoplanin", "CD45RA", "CD15", "betaCatenin", "PAX5", "MCT", "FAP", "CD138",
    "Tbet", "GranzymeB", "IDO-1", "CD45", "CollagenIV", "PD-L1", "Arginase-1", "GATA3"
], #default is None; if provide more than one channel, then they will be combined
    nuclei_channel = 'DAPI', # channel to use for nuclei segmentation
    input_format = 'Multichannel', 
)

# %% [markdown]
# ## Perform segmentation

# %% [markdown]
# Nuclear segmentation only

# %%
# choose between cellpose or mesmer for segmentation
# first image
# seg_output contains {'img': img, 'image_dict': image_dict, 'masks': masks}
nuc_only = sp.tl.cell_segmentation(
    file_name = img_dir,
    channel_file = names,
    output_dir = output_dir,
    seg_method ='mesmer', # cellpose or mesmer
    nuclei_channel = 'DAPI',
    output_fname = 'tonsil1',
    membrane_channel_list = [], #default is None; if provide more than one channel, then they will be combined
    compartment = 'whole-cell', # mesmer # segment whole cells or nuclei only
    input_format ='Multichannel', # Phenocycler or codex
    resize_factor=1, # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
    size_cutoff = 0)

# %% [markdown]
# Nuclear + CD3

# %%
# choose between cellpose or mesmer for segmentation
# first image
# seg_output contains {'img': img, 'image_dict': image_dict, 'masks': masks}
CD3 = sp.tl.cell_segmentation(
    file_name = img_dir,
    channel_file = names,
    output_dir = output_dir,
    seg_method ='mesmer', # cellpose or mesmer
    nuclei_channel = 'DAPI',
    output_fname = 'tonsil1',
    membrane_channel_list = [
     "CD3",
], #default is None; if provide more than one channel, then they will be combined
    compartment = 'whole-cell', # mesmer # segment whole cells or nuclei only
    input_format ='Multichannel', # Phenocycler or codex
    resize_factor=1, # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
    size_cutoff = 0)

# %% [markdown]
# Nuclear + CD45 + betaCatenin

# %%
# choose between cellpose or mesmer for segmentation
# first image
# seg_output contains {'img': img, 'image_dict': image_dict, 'masks': masks}
selected_membrane = sp.tl.cell_segmentation(
    file_name = img_dir,
    channel_file = names,
    output_dir = output_dir,
    seg_method ='mesmer', # cellpose or mesmer
    nuclei_channel = 'DAPI',
    output_fname = 'tonsil1',
    membrane_channel_list = ["CD45", "betaCatenin"], #default is None; if provide more than one channel, then they will be combined
    compartment = 'whole-cell', # mesmer # segment whole cells or nuclei only
    input_format ='Multichannel', # Phenocycler or codex
    resize_factor=1, # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
    size_cutoff = 0)

# %% [markdown]
# All marker

# %%
# choose between cellpose or mesmer for segmentation
# first image
# seg_output contains {'img': img, 'image_dict': image_dict, 'masks': masks}
all_marker = sp.tl.cell_segmentation(
    file_name = img_dir,
    channel_file = names,
    output_dir = output_dir,
    seg_method ='mesmer', # cellpose or mesmer
    nuclei_channel = 'DAPI',
    output_fname = 'tonsil1',
    membrane_channel_list = [
    "FoxP3", "HLA-DR", "CD103", "CHGA", "EGFR", "CD206", "GFAP", "PD-1", "BCL2", "panCK",
    "CD45RO", "CD11b", "CD56", "CD163", "CD21", "CD8", "S100", "Vimentin", "PDGFRb", "CCR7",
    "CD57", "CD34", "Synaptophysin", "CD31", "CXCR5", "CD3", "CD38", "LAG3", "CD25", "CD16",
    "IL-10", "Ki67", "CLEC9A", "p53", "CD69", "CD11c", "CD68", "Ox40", "aSMA", "CD20", "CD4",
    "MUC-1", "Podoplanin", "CD45RA", "CD15", "betaCatenin", "PAX5", "MCT", "FAP", "CD138",
    "Tbet", "GranzymeB", "IDO-1", "CD45", "CollagenIV", "PD-L1", "Arginase-1", "GATA3"
], #default is None; if provide more than one channel, then they will be combined
    compartment = 'whole-cell', # mesmer # segment whole cells or nuclei only
    input_format ='Multichannel', # Phenocycler or codex
    resize_factor=1, # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
    size_cutoff = 0)

# %% [markdown]
# ## Viusalizing the segmentation result

# %%
overlay_data1, rgb_images1 = sp.pl.show_masks(
    seg_output=nuc_only, # output from cell segmentation
    nucleus_channel = 'DAPI', # channel used for nuclei segmentation (displayed in blue)
    additional_channels = ["CD45", "betaCatenin"], # additional channels to display (displayed in green - channels will be combined into one image)
    show_subsample = True, # show a random subsample of the image
    n=2, #need to be at least 2
    tilesize = 100,# number of subsamples and tilesize
    rand_seed = 3)

# %%
overlay_data1, rgb_images1 = sp.pl.show_masks(
    seg_output=CD3, # output from cell segmentation
    nucleus_channel = 'DAPI', # channel used for nuclei segmentation (displayed in blue)
    additional_channels = ["CD45", "betaCatenin"], # additional channels to display (displayed in green - channels will be combined into one image)
    show_subsample = True, # show a random subsample of the image
    n=2, #need to be at least 2
    tilesize = 100,# number of subsamples and tilesize
    rand_seed = 3)

# %%
overlay_data1, rgb_images1 = sp.pl.show_masks(
    seg_output=selected_membrane, # output from cell segmentation
    nucleus_channel = 'DAPI', # channel used for nuclei segmentation (displayed in blue)
    additional_channels = ["CD45", "betaCatenin"], # additional channels to display (displayed in green - channels will be combined into one image)
    show_subsample = True, # show a random subsample of the image
    n=2, #need to be at least 2
    tilesize = 100,# number of subsamples and tilesize
    rand_seed = 3)

# %%
overlay_data1, rgb_images1 = sp.pl.show_masks(
    seg_output=all_marker, # output from cell segmentation
    nucleus_channel = 'DAPI', # channel used for nuclei segmentation (displayed in blue)
    additional_channels = ["CD45", "betaCatenin"], # additional channels to display (displayed in green - channels will be combined into one image)
    show_subsample = True, # show a random subsample of the image
    n=2, #need to be at least 2
    tilesize = 100,# number of subsamples and tilesize
    rand_seed = 3)


