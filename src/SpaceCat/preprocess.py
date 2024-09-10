import anndata
import scipy
import numpy as np
import pandas as pd

from itertools import combinations
from alpineer.misc_utils import verify_in_list

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

    ## TO DO ##
    # generate distance matrices

    # generate neighborhood matrices

    return adata_new
