import anndata
import scipy
import os
import numpy as np
import pandas as pd
import skimage.io as io

from itertools import combinations
from alpineer.misc_utils import verify_in_list
from alpineer import io_utils
from ark.segmentation import marker_quantification

pd.set_option("future.no_silent_downcasting", True)


def create_single_positive_table(marker_vals, threshold_list):
    """ Determine whether a cell is positive for a marker based on the provided threshold.
    Args:
        marker_vals (pd.DataFrame): dataframe containing the marker intensity values
        threshold_list (list): list of functional markers and their pre-determined thresholds

    Returns:
        pd.DataFrame:
            contains the marker intensities as well as the single positive marker data
    """
    # create binary functional marker table, append to anndata table
    for marker, threshold in threshold_list:
        marker_vals[marker + '+'] = marker_vals[marker].values >= threshold

    return marker_vals


def create_double_positive_table(marker_vals, threshold_list):
    """ Determine whether a cell is double positive for a marker based on single positive data.
    Args:
        marker_vals (pd.DataFrame): dataframe containing marker intensity and single positive data
        threshold_list (list): list of functional markers and their pre-determined thresholds

    Returns:
        pd.DataFrame:
            contains the marker intensities, and both the single positive/double positive info
    """
    # pairwise marker thresholding to determine cells positive for combinations of any two markers
    functional_markers = [x[0] for x in threshold_list]
    for marker1, marker2 in combinations(functional_markers, 2):
        dp_positvity = pd.DataFrame({marker1 + '__' + marker2 + '+': []})
        dp_positvity[marker1 + '__' + marker2 + '+'] = np.logical_and(marker_vals[marker1 + '+'],
                                                                      marker_vals[marker2 + '+'])
        marker_vals = pd.concat((marker_vals, dp_positvity), axis=1)

    return marker_vals


def create_functional_tables(adata_table, threshold_list):
    """ Take in a cell table and return a new table, with functional marker positivity info for
    each cell.
    Args:
        adata_table (anndata): cell table containing intensity data for each marker
        threshold_list (list): list of functional markers and their pre-determined thresholds

    Returns:
        anndata:
            a new table with the marker positivity information included
    """
    marker_vals = adata_table.to_df()

    # add functional marker positivity to anndata cell table
    sp_table = create_single_positive_table(marker_vals, threshold_list)
    dp_table = create_double_positive_table(sp_table, threshold_list)
    dp_table = dp_table.replace({True: 1, False: 0})

    # create new anndata table with marker positivity contained in .X
    adata_new = anndata.AnnData(np.array(dp_table))
    adata_new.X = adata_new.X.astype(np.float32)
    adata_new.var_names = dp_table.columns
    adata_new.obs = adata_table.obs.copy()
    adata_new.obsm = adata_table.obsm.copy()
    adata_new.obsp = adata_table.obsp.copy()

    return adata_new


def calculate_compartment_areas(mask_dir, fovs):
    """Calculate the area of each mask per fov

    Args:
        mask_dir (str): path to directory containing masks for each fov
        fovs (list): list of fovs to calculate mask areas for

    Returns
        pd.DataFrame: dataframe containing the area of each mask per fov
    """
    # get list of masks
    mask_files = io_utils.list_files(os.path.join(mask_dir, fovs[0]))
    mask_names = [os.path.splitext(os.path.basename(x))[0] for x in mask_files]

    # loop through fovs and masks to calculate area
    area_dfs = []
    for fov in fovs:
        mask_areas = []
        for mask_file in mask_files:
            mask = io.imread(os.path.join(mask_dir, fov, mask_file))
            if not np.all(np.isin(np.unique(mask), np.array([0, 1]))):
                raise ValueError(f"{mask_file} is not binary. Ensure the mask values are 0 or 1.")
            mask_areas.append(np.sum(mask))

        area_df = pd.DataFrame({'compartment': mask_names, 'compartment_area': mask_areas, 'fov': fov})
        area_dfs.append(area_df)

    return pd.concat(area_dfs, axis=0)


def assign_cells_to_compartment(seg_dir, mask_dir, fovs, seg_mask_substr):
    """Assign cells an image to the mask they overlap most with

    Args:
        seg_dir (str): path to segmentation directory
        mask_dir (str): path to mask directory, with masks for each FOV in a dedicated folder
        fovs (list): list of fovs to process
        seg_mask_substr (str): substr attached to segmentation masks after fov name

    Returns:
        pandas.DataFrame: dataframe with cell assignments to masks
    """

    # extract counts of each mask per cell
    normalized_cell_table, _ = marker_quantification.generate_cell_table(
        segmentation_dir=seg_dir, tiff_dir=mask_dir, fovs=fovs, img_sub_folder='',
        fast_extraction=True, mask_types=[seg_mask_substr])

    # drop cell_size column
    normalized_cell_table = normalized_cell_table.drop(columns=['cell_size'])

    # move fov column to front
    fov_col = normalized_cell_table.pop('fov')
    normalized_cell_table.insert(0, 'fov', fov_col)

    # remove all columns after label
    normalized_cell_table = normalized_cell_table.loc[:, :'label']

    # move label column to front
    label_col = normalized_cell_table.pop('label')
    normalized_cell_table.insert(1, 'label', label_col)

    # create new column with name of column max for each row
    normalized_cell_table['compartment'] = normalized_cell_table.iloc[:, 2:].idxmax(axis=1)

    return normalized_cell_table[['fov', 'label', 'compartment']]


def preprocess_compartment_masks(seg_dir, mask_dir, seg_mask_substr):
    """ Assign cells to compartment based on provided masks.
    Args:
        seg_dir (str): path to the directory containing the cell segmentation masks
        mask_dir (str): path to the directory containing the compartment masks
        seg_mask_substr (str): substr attached to segmentation masks after fov name

    Returns:
        pd.DataFrame:
            table with the compartment and compartment area data for each cell in each image
    """
    fovs = io_utils.list_folders(mask_dir)

    # compute the area of each mask
    area_df = calculate_compartment_areas(mask_dir, fovs)

    # assign cells to the correct compartment
    all_assignment_table = pd.DataFrame()
    for i in range(0, len(fovs), 100):
        assignment_table = assign_cells_to_compartment(
            seg_dir, mask_dir, fovs=fovs[i:i + 100], seg_mask_substr=seg_mask_substr)
        all_assignment_table = pd.concat([all_assignment_table, assignment_table])

    compartment_cell_data = all_assignment_table.merge(area_df, on=['fov', 'compartment'], how='left')
    compartment_cell_data.to_csv(os.path.join(mask_dir, 'compartment_cell_annotations.csv'), index=False)

    return compartment_cell_data


def preprocess_table(adata_table, threshold_list, image_key, seg_label_key, seg_dir=None,
                     mask_dir=None, seg_mask_substr='whole_cell'):
    """ Take in a cell table and return a processed table.
    Args:
        adata_table (anndata): cell table containing intensity data for each marker
        threshold_list (list): list of functional markers and their pre-determined thresholds
        image_key (str): column identifying specific regions of tissue in your data
        seg_label_key (str): column identifying the segmentation label for each cell in the image
        seg_dir (str): path to the directory containing the cell segmentation masks
        mask_dir (str): path to the directory containing compartment masks for the image data
        seg_mask_substr (str): substr attached to segmentation masks after fov name

    Returns:
        anndata:
            a new anndata table with the relevant information added
    """

    # check threshold list is subset of adata_table.X columns & that provided keys are in .obs
    verify_in_list(provided_thresholds=[x[0] for x in threshold_list], all_markers=adata_table.var_names)
    verify_in_list(provided_keys=[image_key, seg_label_key], all_keys=adata_table.obs.columns)

    # check for sparse .X, convert to dense if necessary
    if isinstance(adata_table.X, scipy.sparse.csr_matrix):
        adata_table.X = adata_table.X.todense()

    # ensure .X only contains numeric values
    try:
        _ = adata_table.to_df().apply(pd.to_numeric)
    except ValueError:
        raise ValueError("Ensure all values contained in adata.X are numeric values.")

    # add functional marker positivity data
    adata_new = create_functional_tables(adata_table, threshold_list)

    # add compartment data
    if seg_dir and mask_dir:
        # check if previous compartment assignments exist
        compartment_cell_data_path = os.path.join(mask_dir, 'compartment_cell_annotations.csv')
        if os.path.exists(compartment_cell_data_path):
            compartment_cell_data = pd.read_csv(compartment_cell_data_path)
        else:
            # assign each cell to a compartment and calculate compartment areas
            compartment_cell_data = preprocess_compartment_masks(seg_dir, mask_dir, seg_mask_substr)

        # rename image and seg label column if needed
        compartment_cell_data = compartment_cell_data.rename(
            columns={'fov': image_key, 'label': seg_label_key})

        # append data to .obs
        adata_new.obs = adata_new.obs.merge(compartment_cell_data, on=[image_key, seg_label_key],
                                            how='left')

    return adata_new
