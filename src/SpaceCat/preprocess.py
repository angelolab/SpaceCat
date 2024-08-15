import anndata
import numpy as np
import pandas as pd

from itertools import combinations
from alpineer import verify_in_list


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
        marker_vals[marker1 + '__' + marker2 + '+'] = np.logical_and(marker_vals[marker1 + '+'],
                                                                     marker_vals[marker2 + '+'])

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
    marker_vals = pd.DataFrame(adata_table.X, columns=adata_table.var_names).copy()

    # add functional marker positivity to anndata cell table
    sp_table = create_single_positive_table(marker_vals, threshold_list)
    dp_table = create_double_positive_table(sp_table, threshold_list)
    dp_table = dp_table.replace({True: 1, False: 0})

    # create new anndata table with marker positivity contained in .X
    adata_new = anndata.AnnData(dp_table)
    adata_new.obs = adata_table.obs.copy()
    adata_new.obsm = adata_table.obsm.copy()
    adata_new.obsp = adata_table.obsp.copy()

    return adata_new


def preprocess_table(adata_table, threshold_list):
    """ Take in a cell table and return a processed table.
    Args:
        adata_table (anndata): cell table containing intensity data for each marker
        threshold_list (list): list of functional markers and their pre-determined thresholds

    Returns:
        anndata:
            a new anndata table with the relevant information added
    """
    # check threshold list is subset of adata_table.X columns
    verify_in_list(provided_thresholds=[x[0] for x in threshold_list], all_markers=adata_table.var_names)

    # add functional marker positivity data
    adata_new = create_functional_tables(adata_table, threshold_list)

    ## TO DO ##
    # generate distance matrices

    # generate neighborhood matrices

    return adata_new
