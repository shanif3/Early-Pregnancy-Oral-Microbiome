import os
import umap
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sewar.full_ref import mse, sam
import matplotlib.pyplot as plt
import MIPMLP
from .microbiome2matrix import micro2matrix
from scipy.stats import norm
from functools import reduce
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


def load_img(folder_path, tag=None):
    """
    Reads all the images saved in a certain folder path and in the tag file
    :param folder_path: Path of the folder where the images from micro2matrix are saved (str)
    :param Default is None, but if we want to work only on images which we have theri tag
            a tag dataframe or series should be passed too (pandas)
    :return: (1) final array with all the loaded images from the folder (list)
             (2) names list with the loaded images names (list)
    """
    arrays = []
    names = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            if file == "bact_names.npy":
                continue
            file_path = os.path.join(folder_path, file)
            if tag is None:
                arrays.append(np.load(file_path, allow_pickle=True, mmap_mode='r'))
            else:
                if file_path.split("\\")[-1].replace(".npy", "") in tag.index:
                    arrays.append(np.load(file_path, allow_pickle=True, mmap_mode='r'))

            names.append(file_path.split("\\")[-1].replace(".npy", ""))

    final_array = np.stack(arrays, axis=0)
    return final_array, names


def fft_process(x, cutoff):
    """
    Apply FFT on each images with the cutoff given.
    :param x: A single image (ndarray)
    :param cutoff: Cutoff frequency as a fraction of the maximum possible frequency (float)
    :return: A filtered image (ndarray)
    """
    fft = np.fft.fft2(x)

    # Shift the zero-frequency component to the center of the array
    fft_shifted = np.fft.fftshift(fft)

    # Define the cutoff frequency (as a fraction of the maximum possible frequency)
    cutoff_freq = cutoff

    # Create a mask to keep only the low-frequency components
    rows, cols = fft_shifted.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols), dtype=bool)
    mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
    ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = True

    # Apply the mask to the FFT
    fft_cutoff = np.copy(fft_shifted)
    fft_cutoff[~mask] = 0

    # Inverse FFT to get the filtered image
    img_filtered_x = np.fft.ifft2(np.fft.ifftshift(fft_cutoff))
    return img_filtered_x


def final_distance(output1, output2, f):
    """
    Calculate the distance between 2 filtered images.
    :param output1: Filtered image 1 (ndarray)
    :param output2: Filtered image 2 (ndarray)
    :param f: Metric to calculate the distance according to
              One of  "d1","d2","d3","sam","mse"
    :return:
    """
    if f == "d1":
        # Euclidean distance
        return np.linalg.norm(output1 - output2)
    elif f == "d2":
        # Absolute difference
        return np.sum(np.abs(np.abs(output1) - np.abs(output2)))
    elif f == "d3":
        # Difference of angles
        return np.sum(np.abs(np.angle(output1) - np.angle(output2)))
    elif f == "sam":
        return sam(output1, output2)
    elif f == "mse":
        return mse(output1, output2)


def build_SAMBA_distance_matrix(folder_path, metric="sam", cutoff=0.8, tag=None,imgs=None,ordered_df=None,bact_names=None,class_=False):
    """
    Build SAMBA distance matrix of the FFT processed images using the metric as the
    final distance metric between the processed images, and the cutoff as the FFT cutoff
    :param folder_path: Path of the folder where the images from micro2matrix are saved if save was True(str)
    :param metric: Metric to calculate the distance according to.
                   One of  "d1","d2","d3","sam","mse"
    :param cutoff: Cutoff frequency as a fraction of the maximum possible frequency (float)
    :param tag: Default is None, but if we want to work only on images which we have their tag
                a tag dataframe or series should be passed too (pandas dataframe)
    :param imgs: If one wants to work with saved images from folder it is None, else it is an array of 2D images (ndarray)
    :param ordered_df: If one wants to work with saved images from folder it is None, else it is a pandas dataframe with
    the new order of the taxa as its columns (pandas dataframe)
    :return: Distance matrix dataframe (pandas dataframe)
    """
    # Load images from the folder
    if imgs is None:
        imgs, names = load_img(folder_path, tag)
    else:
        if tag is None:
            names = list(ordered_df.index)
        else:
            names = list(tag.index.intersection(ordered_df.index))


    # Image shape
    x_axis = imgs.shape[-1]
    y_axis = imgs.shape[-2]

    def create_classes(bact_names):
        names = list(bact_names[3])
        list_classes = []
        for bact in names:
            if len(bact.split(";")) >= 3:
                c = bact.split(";")[2]
            else:
                c = bact.split(";")[-1]
            list_classes.append(c)
        return list_classes

    def make_class_img(diff,list_classes):
        #diff = diff.T
        diff_df = pd.DataFrame(data=diff,
                               columns=["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"],
                               index=list_classes)
        diff_df.index = [i.replace("_0", "") for i in diff_df.index]
        diff_c = diff_df.groupby(diff_df.index).mean()
        # only classes, remove p__ and k__
        if "0.0" in list(diff_c.index):
            diff_c = diff_c.drop(index="0.0")
        return diff_c
    # Function for images adjusting (FFT) and calculating the pairwise distance
    def fft_dist(x, y):
        x = x.reshape(x_axis, y_axis)
        y = y.reshape(x_axis, y_axis)
        x_after = fft_process(x, cutoff)
        y_after = fft_process(y, cutoff)
        ######### TRY###############
        if class_:
            list_classes = create_classes(bact_names)
            x_after = make_class_img(x_after,list_classes)
            y_after = make_class_img(y_after,list_classes)
            x_after = x_after.values
            y_after = y_after.values
        return final_distance(x_after, y_after, metric)

    # Build the SAMBA distance matrix
    dm = cdist(imgs.reshape(imgs.shape[0], -1), imgs.reshape(imgs.shape[0], -1), metric=fft_dist)

    if tag is None:
        dm = pd.DataFrame(dm, index=names, columns=names)
    else:
        dm = pd.DataFrame(dm, index=tag.index, columns=tag.index)

    return dm


def plot_umap(dm, tag, save):
    umap_embedding = umap.UMAP(metric='precomputed').fit_transform(dm)
    umap_embedding_df = pd.DataFrame(data=umap_embedding, index=tag.index, columns=["PCA1", "PCA2"])
    tag0 = tag[tag.values == 0]
    tag1 = tag[tag.values == 1]

    umap_embedding_df0 = umap_embedding_df.loc[tag0.index]
    umap_embedding_df1 = umap_embedding_df.loc[tag1.index]

    plt.scatter(umap_embedding_df0["PCA1"], umap_embedding_df0["PCA2"], color="red", label="Control")
    plt.scatter(umap_embedding_df1["PCA1"], umap_embedding_df1["PCA2"], color="blue", label="Condition")
    plt.xlabel("UMAP dim 1", fontdict={"fontsize": 15})
    plt.ylabel("UMAP dim 2", fontdict={"fontsize": 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()
    if not os.path.exists(save):
        os.makedirs(save)
    plt.savefig(f"{save}/umap_plot.png")
    plt.show()

def compute_var_values(img,cutoff=None):
    if cutoff == None:
        # Compute the FFT of the image
        fft_result = np.fft.fft2(img)

        # Shift the zero frequency component to the center of the spectrum
        fft_result = np.fft.fftshift(fft_result)
        fft_var = fft_result.var()
    else:
        fft = np.fft.fft2(img)

        # Shift the zero-frequency component to the center of the array
        fft_shifted = np.fft.fftshift(fft)

        # Define the cutoff frequency (as a fraction of the maximum possible frequency)
        cutoff_freq = cutoff

        # Create a mask to keep only the low-frequency components
        rows, cols = fft_shifted.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols), dtype=bool)
        mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
        ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = True

        # Apply the mask to the FFT
        fft_cutoff = np.copy(fft_shifted)
        fft_cutoff[~mask] = 0

        # Inverse FFT to get the filtered image
        img_filtered_x = np.fft.ifft2(np.fft.ifftshift(fft_cutoff))
    return img_filtered_x

def create_classes(bact_names):
    names = list(bact_names[3])
    list_classes = []
    for bact in names:
        if len(bact.split(";")) >= 3:
            c = bact.split(";")[2]
        else:
            c = bact.split(";")[-1]
        list_classes.append(c)
    return list_classes

def calc_avg(imgs_,CUTOFF):
    list_imgs_after_fft = list()
    for img_index, img in enumerate(imgs_):
        fft_img = compute_var_values(img, CUTOFF)
        list_imgs_after_fft.append(np.abs(fft_img))
    # Stack the list into a single 3D ndarray
    data_stack = np.stack(list_imgs_after_fft)
    # Compute the variance along the first axis (axis=0)
    variance_array = np.mean(data_stack, axis=0)
    return variance_array

def apply_class_analysis(imgs1, imgs0, CUTOFF,bact_names):
    variance_array0 = calc_avg(imgs0, CUTOFF)
    variance_array = calc_avg(imgs1, CUTOFF)
    diff = variance_array - variance_array0
    list_classes = create_classes(bact_names)
    diff = diff.T
    diff= diff[:,1:]
    diff_df = pd.DataFrame(data=diff,columns= ["Kingdom","Phylum","Class","Order","Family","Genus","Species"],index= list_classes)
    diff_df.index = [i.replace("_0","") for i in diff_df.index]
    diff_c = diff_df.groupby(diff_df.index).mean()
    if '0.0' in diff_c.index:
        diff_c = diff_c.drop('0.0')
    return diff_c

def make_consistent_imgs(dfs):
    # Step 1: Create a list with unique indexes of all DataFrames
    unique_indexes = set()
    for df in dfs.values():
        unique_indexes.update(df.index)

    unique_indexes = list(unique_indexes)  # Convert to list

    # Step 2: Fill each DataFrame with the missing indexes with zeros
    for key in dfs:
        dfs[key] = dfs[key].reindex(unique_indexes, fill_value=0)

    # Step 3: Reorder the indexes in all DataFrames to be in the same order
    for key in dfs:
        dfs[key] = dfs[key].loc[unique_indexes]
    return dfs

def adjust_df_values(df):
    # Step 1: Compute the Z-score for each entry
    mean = df.mean().mean()
    std = df.stack().std()
    z_scores = (df - mean) / std

    # Step 2: Calculate the p-value for each Z-score
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))  # two-tailed test

    # Step 3: Apply the condition to modify the DataFrame values
    df_adjusted = df.where(p_values < 0.05, 0)

    return df_adjusted

def plot_class_matrix(dfs_t1_vs_t2,dfs_t2_vs_t3,title,cohort):
    SIZE = 20
    dfs_t1_vs_t2 = make_consistent_imgs(dfs_t1_vs_t2)
    dfs_t2_vs_t3= make_consistent_imgs(dfs_t2_vs_t3)

    list_of_sums_t1_vs_t2 = list()
    list_of_sums_t2_vs_t3 = list()
    list_binary_sums_t1_vs_t2 = list()
    list_binary_sums_t2_vs_t3 = list()
    for idx, (key, df) in enumerate(dfs_t1_vs_t2.items()):
        df =adjust_df_values(df)
        indices_to_remove = ['c__']
        df = df.drop(index=indices_to_remove, errors='ignore')
        list_of_sums_t1_vs_t2.append(df)
        b_df = (df!=0.0).astype(float)
        list_binary_sums_t1_vs_t2.append(b_df)


    for idx, (key, df) in enumerate(dfs_t2_vs_t3.items()):
        df =adjust_df_values(df)
        indices_to_remove = ['c__']
        df = df.drop(index=indices_to_remove, errors='ignore')
        list_of_sums_t2_vs_t3.append(df)
        b_df = (df!=0.0).astype(float)
        list_binary_sums_t2_vs_t3.append(b_df)

    # plot_circle_plot(list_of_sums,list_binary_sums,title,SIZE)
    swapped_visualization(list_of_sums_t1_vs_t2,list_binary_sums_t1_vs_t2,
                          list_of_sums_t2_vs_t3,list_binary_sums_t2_vs_t3,title,SIZE,cohort)

def plot_circle_plot(list_of_sums, list_binary_sums, title, SIZE):
    # Calculate sum and average DataFrames
    sum_df = reduce(lambda x, y: x + y, list_of_sums)
    average_df = sum_df / len(list_of_sums)
    b_sum_df = reduce(lambda x, y: x + y, list_binary_sums)

    # Normalize the sizes to avoid overlapping
    max_size = b_sum_df.max().max()-30
    size_scale = 0.5 / max_size  # Increase the scaling factor for larger circles

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 8))

    patches = []
    colors = []
    sizes = []

    # Create circles based on DataFrame values
    for (i, col) in enumerate(average_df.columns):  # Switch to columns for y-axis
        for (j, row) in enumerate(average_df.index):  # Switch to rows for x-axis
            value = sum_df.at[row, col]
            size = b_sum_df.at[row, col] if b_sum_df.at[row, col] != 0 else 0.01  # Avoid zero sizes
            size = size * size_scale  # Adjust size scale
            circle = Circle((j, i), size)  # (x, y) coordinates and size of the circle
            patches.append(circle)
            colors.append(value)
            sizes.append(size)

    # Create a PatchCollection
    p = PatchCollection(patches, cmap='bwr_r', edgecolor='black')
    p.set_array(np.array(colors))
    p.set_clim([-0.2, 0.2])  # Set color limits

    # Add circles to the plot
    ax.add_collection(p)

    # Set axis limits
    ax.set_xlim(-0.5, len(average_df.index) - 0.5)  # Adjusted for transposed axes
    ax.set_ylim(-0.5, len(average_df.columns) - 0.5)  # Adjusted for transposed axes

    # Set axis labels and title
    ax.set_xticks(np.arange(len(average_df.index)))  # Adjusted for transposed axes
    ax.set_yticks(np.arange(len(average_df.columns)))  # Adjusted for transposed axes
    ax.set_xticklabels(average_df.index, rotation=90)  # Adjusted for transposed axes
    ax.set_yticklabels(average_df.columns,fontsize=SIZE)  # Adjusted for transposed axes

    # ax.set_title(f'{title}', fontsize=SIZE, fontweight='bold')

    # Add a grid
    ax.grid(True)

    # Add a color bar
    cbar = plt.colorbar(p, ax=ax)
    cbar.set_label('Sum Value',fontsize=SIZE)
    cbar.ax.tick_params(labelsize=SIZE)
    plt.tight_layout()
    plt.savefig(f"figure_plots/circle_{title}.png")
    plt.show()

def swapped_visualization(list_of_sums_t1_vs_t2,list_binary_sums_t1_vs_t2,
                          list_of_sums_t2_vs_t3,list_binary_sums_t2_vs_t3, title, SIZE,cohort):
    # Calculate sum and average DataFrames
    sum_df_t1_t2 = reduce(lambda x, y: x + y, list_of_sums_t1_vs_t2)
    average_df_t1_t2 = sum_df_t1_t2 / len(list_of_sums_t1_vs_t2)
    b_sum_df_t1_t2 = reduce(lambda x, y: x + y, list_binary_sums_t1_vs_t2)

    sum_df_t2_t3 = reduce(lambda x, y: x + y, list_of_sums_t2_vs_t3)
    average_df_t2_t3 = sum_df_t2_t3 / len(list_of_sums_t2_vs_t3)
    b_sum_df_t2_t3 = reduce(lambda x, y: x + y, list_binary_sums_t2_vs_t3)

    # Normalize the sizes to avoid overlapping
    max_size = max(b_sum_df_t1_t2.max().max(), b_sum_df_t2_t3.max().max())
    size_scale = 0.46 / max_size  # Reduced scaling since we have two circles per position

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 16))

    patches_t1_t2 = []
    colors_t1_t2 = []
    patches_t2_t3 = []
    colors_t2_t3 = []

    # Create circles for t1_vs_t2 (green edges) - slightly offset to left
    for (i, col) in enumerate(average_df_t1_t2.columns):
        for (j, row) in enumerate(average_df_t1_t2.index):
            value = sum_df_t1_t2.at[row, col]
            size = b_sum_df_t1_t2.at[row, col] if b_sum_df_t1_t2.at[row, col] != 0 else 0.01
            size = size * size_scale
            # Offset slightly to the left
            circle = Circle((i - 0.15, j), size)
            patches_t1_t2.append(circle)
            colors_t1_t2.append(value)

    # Create circles for t2_vs_t3 (pink edges) - slightly offset to right
    for (i, col) in enumerate(average_df_t2_t3.columns):
        for (j, row) in enumerate(average_df_t2_t3.index):
            value = sum_df_t2_t3.at[row, col]
            size = b_sum_df_t2_t3.at[row, col] if b_sum_df_t2_t3.at[row, col] != 0 else 0.01
            size = size * size_scale
            # Offset slightly to the right
            circle = Circle((i + 0.15, j), size)
            patches_t2_t3.append(circle)
            colors_t2_t3.append(value)

    # Create PatchCollections with different edge colors
    p1 = PatchCollection(patches_t1_t2, cmap='bwr_r', edgecolor='green', linewidth=7)
    p1.set_array(np.array(colors_t1_t2))
    p1.set_clim([-0.2, 0.2])

    p2 = PatchCollection(patches_t2_t3, cmap='bwr_r', edgecolor='black', linewidth=7)
    p2.set_array(np.array(colors_t2_t3))
    p2.set_clim([-0.2, 0.2])

    # Add both collections to the plot
    if cohort=='Israel':
        ax.add_collection(p1)
    ax.add_collection(p2)

    # Set axis limits (assuming both dataframes have same dimensions)
    ax.set_xlim(-0.5, len(average_df_t1_t2.columns) - 0.5)
    ax.set_ylim(-0.5, len(average_df_t1_t2.index) - 0.5)

    # Set axis labels and title
    ax.set_xticks(np.arange(len(average_df_t1_t2.columns)))
    ax.set_yticks(np.arange(len(average_df_t1_t2.index)))
    ax.set_xticklabels(average_df_t1_t2.columns, rotation=45, fontsize=SIZE)
    ax.set_yticklabels(average_df_t1_t2.index, fontsize=SIZE)

    # Add a grid
    ax.grid(True)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)

    # Create colorbar axis at bottom
    cax = divider.append_axes("bottom", size="5%", pad=1.2)
    # Use one of the patch collections for the colorbar
    cbar = plt.colorbar(p1, cax=cax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label('Sum Value', fontsize=SIZE)
    cbar.ax.tick_params(labelsize=SIZE)

    # Add legend to distinguish the two types
    from matplotlib.lines import Line2D
    if cohort=='Israel':
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='green', markersize=18, markeredgewidth=7, label='T1 vs T2'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=18, markeredgewidth=7, label='T2 vs T3')
        ]
    else:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=18, markeredgewidth=7, label='T2 vs T3')
        ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=25)

    plt.tight_layout(pad=4.0)

    root_dir = os.path.dirname(os.getcwd())
    path_to_save= os.path.join(root_dir, 'Paper_plots')
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    plt.savefig(os.path.join(path_to_save,f'combined_swapped_circle_{title}_{cohort}.png'), bbox_inches='tight', dpi=300)
    plt.show()

def is_numeric_index(index):
    try:
        pd.to_numeric(index)
        return True
    except ValueError:
        return False

def apply_meta_analysis(folder,list_datasets,CUTOFF,cohort):
    dict_class_diffs = dict()
    for name in list_datasets:
        # Load raw data and tag. Make sure they are in right format
        raw = pd.read_csv(f"{folder}/{name}/for_preprocess.csv", index_col=0)
        tag = pd.read_csv(f"{folder}/{name}/tag.csv",index_col=0)["Tag"]

        # Apply the MIPMLP with the defaultive parameters
        processed =raw

        # Check if the processed index is numeric and convert if possible
        if is_numeric_index(processed.index):
            processed.index = pd.to_numeric(processed.index, errors='coerce')

        # Check if the tag index is numeric and convert if possible
        if is_numeric_index(tag.index):
            tag.index = pd.to_numeric(tag.index, errors='coerce')

        common = list(processed.index.intersection(tag.index))
        tag = tag.loc[common]
        processed = processed.loc[common]
        # micro2matrix there is an option to save the images in a prepared folder
        array_of_imgs, bact_names, ordered_df = micro2matrix(processed, f"{folder}/{name}/2D_images", save=False)
        tag0 = tag[tag.values == 0.0]
        tag1 = tag[tag.values == 1.0]
        imgs_names = list(ordered_df.index)
        # Find the indices of the common values between tag0 and imgs_names
        img0_index = [imgs_names.index(name) for name in tag0.index if name in imgs_names]

        # Find the indices of the common values between tag1 and imgs_names
        img1_index = [imgs_names.index(name) for name in tag1.index if name in imgs_names]
        imgs0 = array_of_imgs[img0_index]
        imgs1 = array_of_imgs[img1_index]

        diff_c = apply_class_analysis(imgs1, imgs0, CUTOFF=CUTOFF, bact_names=bact_names)
        dict_class_diffs[name] = diff_c
    dict_class_diffs_t1_vs_t2 = {key: dict_class_diffs[key] for key in dict_class_diffs if list_datasets[0] in key}
    dict_class_diffs_t2_vs_t3 = {key: dict_class_diffs[key] for key in dict_class_diffs if list_datasets[1] in key}


    plot_class_matrix(dict_class_diffs_t1_vs_t2,dict_class_diffs_t2_vs_t3,"T1 vs T2 GIMIC",cohort)




