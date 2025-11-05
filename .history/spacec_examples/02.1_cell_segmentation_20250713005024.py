# %% [markdown]
# # SPACEc: Cell Segmentation

# %% [markdown]
# The aim of cell segmentation is to identify each cell within a given image and derive single cell data from the image. For that we provide two commonly used approaches: Deepcell Mesmer and Cellpose. Mesmer is a deep learning-enabled segmentation algorithm that works out-of-the-box for most multiplexed images. Apart from that we provide Cellpose that is a deep learning-enabled segmentation algorithm as well but provides different models that can be directly employed. Additionally, Cellpose allows users to easily train their own models. 
# 
# For the purpose of this tutorial we will use Mesmer.
# 
# The steps of the script are:
# 
# 1. Deciding on the image channels for segmentation 
# 2. Running segmentation 
# 3. Quality control the segmented images
# 4. Store the data for further processing

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
# Specify the path to the data
root_path = "/home/user/path/SPACEc/" # inset your own path
data_path = root_path + 'example_data/raw/' # where the data is stored

# where you want to store the output
output_dir = root_path + 'example_data/output/' # inset your own path
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# If you want to use GPU acceleration for the segmentation you should ensure that CUDA is installed and the GPU detected by the python environment. If you have a compatible Nvidia card but never installed CUDA Toolkit before you can find installation instructions here: https://developer.nvidia.com/cuda-downloads

# %%
#check if GPU availability
!nvcc --version
!nvidia-smi

# %%
sp.hf.check_for_gpu()

# %% [markdown]
# ## Cell segmentation

# %% [markdown]
# **NOTE:** Our segmentation function features a parameter called 'input_format'. This parameter defines what input data the function accepts. If set to 'Multichannel' the function expects a single multichannel tiff file, if set to 'Channels' the function expects a folder with single Tiff files (no channelnames.txt required) and if set to 'CODEX' the function reads the output of the classic first gen CODEX setup.

# %% [markdown]
# Before committing to potentially time intense segmentation it might be useful to visualize the segmentation channels. In this tutorial we provide both nuclei and membrane channels. Especially if no general membrane marker is available it is useful to combine membrane markers as shown below. 

# %%
# (optional, one can just use nuclei for segmentation)
# Visualize membrane channels to use for cell segmentation 

sp.pl.segmentation_ch(
    file_name = output_dir + 'reg001_X01_Y01_Z01.tif', # image for segmentation
    channel_file = data_path + 'channelnames.txt', # all channels used for staining
    output_dir = output_dir, #
    extra_seg_ch_list = ["CD45", "betaCatenin"], #default is None; if provide more than one channel, then they will be combined
    nuclei_channel = 'DAPI', # channel to use for nuclei segmentation
    input_format = 'Multichannel', 
)

# %% [markdown]
# After deciding on the channels for segmentation, segmentation is performed with the 'cell_segmentation' function. Besides choosing the channels the function allows to select the segmentation model. The function expects a multichannel tif file and a channel names file as input (please see our example data as an example). The segmentation output is stored as csv file.

# %%
# choose between cellpose or mesmer for segmentation
# first image
# seg_output contains {'img': img, 'image_dict': image_dict, 'masks': masks}
seg_output1 = sp.tl.cell_segmentation(
    file_name = output_dir + 'reg001_X01_Y01_Z01.tif',
    channel_file = data_path + 'channelnames.txt',
    output_dir = output_dir,
    seg_method ='mesmer', # cellpose or mesmer
    nuclei_channel = 'DAPI',
    output_fname = 'tonsil1',
    membrane_channel_list = ["CD45", "betaCatenin"], #default is None; if provide more than one channel, then they will be combined
    compartment = 'whole-cell', # mesmer # segment whole cells or nuclei only
    input_format ='Multichannel', # Phenocycler or codex
    resize_factor=1, # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
    size_cutoff = 0)

# %%
# second image
# choose the method that is consistent of the first image for a more comparable result
# seg_output contains {'img': img, 'image_dict': image_dict, 'masks': masks}
seg_output2 = sp.tl.cell_segmentation(
    file_name = output_dir + 'reg002_X01_Y01_Z01.tif',
    channel_file = data_path + 'channelnames.txt',
    output_dir = output_dir,
    output_fname = 'tonsil2',
    seg_method ='mesmer', # cellpose or mesmer
    nuclei_channel = 'DAPI',
    membrane_channel_list = ["CD45", "betaCatenin"], #default is None #default is None; if provide more than one channel, then they will be combined
    input_format ='Multichannel', # Phenocycler or codex
    compartment = 'whole-cell', # mesmer # segment whole cells or nuclei only
    resize_factor=1, # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
    size_cutoff = 0) 

# %% [markdown]
# In addition to the mesmer segmentation that is used in our example you can use cellpose as shown in the example below.

# %%
seg_output_cellpose = sp.tl.cell_segmentation(
    file_name = output_dir + 'reg002_X01_Y01_Z01.tif',
    channel_file = data_path + 'channelnames.txt',
    output_dir = output_dir,
    output_fname = 'tonsil2',
    seg_method ='cellpose', # cellpose or mesmer
    model='cyto3', # cellpose model
    diameter=28, # average cell diameter (in pixels). If set to None, it will be automatically estimated.
    nuclei_channel = 'DAPI',
    membrane_channel_list = ["CD45", "betaCatenin"], #default is None #default is None; if provide more than one channel, then they will be combined
    input_format ='Multichannel', # Phenocycler or codex
    resize_factor=1, # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
    size_cutoff = 0) 

# %% [markdown]
# You can also load a custom fine-tuned cellpose model into SPACEc as shown below. The required input file is the zip file that cellpose outputs after training. 

# %%
seg_output_cellpose = sp.tl.cell_segmentation(
    file_name = output_dir + 'reg002_X01_Y01_Z01.tif',
    channel_file = data_path + 'channelnames.txt',
    output_dir = output_dir,
    output_fname = 'tonsil2',
    seg_method ='cellpose', # cellpose or mesmer
    model='/home/user/path_to_custom_model/models/CP_XXXX_XXXX', # cellpose model
    diameter=28, # average cell diameter (in pixels). If set to None, it will be automatically estimated.
    nuclei_channel = 'DAPI',
    membrane_channel_list = ["CD45", "betaCatenin"], #default is None #default is None; if provide more than one channel, then they will be combined
    input_format ='Multichannel', # Phenocycler or codex
    resize_factor=1, # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
    size_cutoff = 0,
    custom_model=True) 

# %% [markdown]
# ## Viusalizing the segmentation result

# %% [markdown]
# Not every dataset works equally well with all segmentation models due to differences in tissue type, structure or image quality. Therefore, it is of major importance to check the segmentation results before continuing with the data analysis. the 'show_masks' function selects random tiles of a user defined size from the image to provide examples to evaluate the segmentation quality. If the segmentation quality is not acceptable a different model should be tried. For especially challenging datasets users can also try to retrain a model specifically for their images. 

# %%
overlay_data1, rgb_images1 = sp.pl.show_masks(
    seg_output=seg_output1,
    nucleus_channel = 'DAPI', # channel used for nuclei segmentation (displayed in blue)
    additional_channels = ["CD45", "betaCatenin"], # additional channels to display (displayed in green - channels will be combined into one image)
    show_subsample = True, # show a random subsample of the image
    n=2, #need to be at least 2
    tilesize = 300,# number of subsamples and tilesize
    rand_seed = 1)

# %%
overlay_data2, rgb_images2 = sp.pl.show_masks(
    seg_output=seg_output2,
    nucleus_channel = 'DAPI', # channel used for nuclei segmentation (displayed in blue)
    additional_channels = ["CD45", "betaCatenin"], # additional channels to display (displayed in green - channels will be combined into one image)
    show_subsample = True, # show a random subsample of the image
    n=2, #need to be at least 2
    tilesize = 300, # number of subsamples and tilesize
    rand_seed = 3) 

# %% [markdown]
# ## Save the segmentation result

# %% [markdown]
# After successful segmentation, the images and masks can be stored in a pickle file for later easy access.

# %%
#Save segmentation output
with open(output_dir + 'seg_output_tonsil1.pickle', 'wb') as f:
    pickle.dump(seg_output1, f)

with open(output_dir + 'seg_output_tonsil2.pickle', 'wb') as f:
    pickle.dump(seg_output2, f)
    
#Save the overlay of the data
with open(output_dir + 'overlay_tonsil1.pickle', 'wb') as f:
    pickle.dump(overlay_data1, f)

with open(output_dir + 'overlay_tonsil2.pickle', 'wb') as f:
    pickle.dump(overlay_data2, f)


