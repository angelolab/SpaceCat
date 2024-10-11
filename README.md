# SpaceCat: Generate a Spatial Catalogue from multiplexed imaging experiments

SpaceCat allows you to easily generate hundreds of distinct features from multiplexed imaging data to enable easy downstream quantification. These features are all interpretable and easy to caluclate, 
and don't require complex postprocessing or heavy computational resources! 

All that you  need to get started is a segmentation mask (identifying the location of each cell) and a cell table (which assigns each cell to a cell type). If you haven't segmented and classified your cells yet,
there are many options for segmenting cells in multiplexed imaging data, for example [Mesmer](https://www.nature.com/articles/s41587-021-01094-0), [CellPose](https://www.nature.com/articles/s41592-022-01663-4), or [StarDist](https://arxiv.org/abs/1806.03535).
Similarly, if you're looking for a good option for assigning cell types, a couple options to look at are [Nimbus](https://www.biorxiv.org/content/10.1101/2024.06.02.597062v1), [Pixie](https://www.nature.com/articles/s41467-023-40068-5), or 
[Stellar](https://www.nature.com/articles/s41592-022-01651-8). However, no matter which algorithms you choose for segmentation and cell assignment, you can still run SpaceCat on your data! You should pick the approach you find to generate
the most accurate results, as that will directly influence the quality of the features that are extracted. 

In order to take advantage of all the features in SpaceCat, we recommend you generate a hierarchy of cell type assignments. This will allow you to tailor the level of granularity of the cell types you use for different features. For example, 
you might want to look more broadly at total T cell abundance for some features, whereas in others it would be more beneficial to look specifically at exhausted CD8 T cells. SpaceCat cannot generate these hierarchies for you; it's up to you
to decide how to group your cells together. However, once you've generated this grouping, you are then ready to run SpaceCat!

We have provided an [example dataset](https://github.com/angelolab/SpaceCat/tree/main/data) to help you understand what your data should look like before utilizing SpaceCat. 
If you want to test out our tool, you can quickly run SpaceCat using the code below and the example files!

## Table of Contents
- [1. Installation](#installation)
  - [conda environment](#conda-environment)
  - [pip install (coming soon)](#pip-install-coming-soon)
- [2. AnnData Conversion](#anndata-conversion)
  - [cell table format](#cell-table-format)
  - [metadata format](#metadata-format)
  - [conversion steps](#conversion-steps)
- [3. Preprocessing](#preprocessing)
  - [functional marker positivity](#functional-marker-positivity)
  - [compartment cell assignment](#compartment-cell-assignment)
- [4. Feature Generation](#feature-generation)
  - [running SpaceCat](#running-spacecat)
  - [output tables](#output-tables)
  - [adding metadata](#adding-metadata)
  - [feature descriptions](#feature-descriptions)


## Installation
### conda environment
You can clone the git repository and install the necessary dependencies using the provided yml file. 
First clone the repository with 
```commandline
git clone https://github.com/angelolab/SpaceCat
```
Then change the working directory and create the conda environment using the provided yml file.
```commandline
cd SpaceCat
conda env create -f environment.yml
```
Once the environment is created, you can activate it with
```commandline
conda activate spacecat_env
```

### pip install (coming soon)
We are currently working on making SpaceCat pip installable!

## AnnData Conversion

### Cell Table Format
Your cell table will require a minimum amount of data to generate fundamental features. You can refer to `cell_table.csv` in the [example data](https://github.com/angelolab/SpaceCat/tree/main/data) for reference.
- The **image name** column identifies specific regions of tissue in your data, denoted by `fov` in the example cell table.
- The **segmentation label** column identifies the label for each cell in the image, denoted by `label` in the example cell table.
- The **cell area** column identifies the area of each cell, denoted by `area` in the example cell table.
- The two **centroid columns** should detail the x and y location of each cell in the image, denoted by `centroid-0` and `centroid-1` in the example cell table. 
- At least one **cluster** assignment column needs to specify each cell type in order to generate SpaceCat features; however we recommend you include two or more levels of granularity.
These columns are denoted by `cell_meta_cluster`, `cell_cluster`, and `cell_cluster_broad` in the example cell table.
- Columns detailing the normalized **signal intensity** in each cell are included in the example cell table, so that functional marker features to be generated. 
You may include all markers in the initial cell table conversion, and filter which functional markers you would specifically like to include as features in the [preprocessing section](#preprocessing) below.

### Metadata Format
The metadata file should contain any image level and tissue specific information. 
The one required column is **image column**, named the same as in the cell table, so that the metadata can be merged into the cell table and stored in the anndata object created in the next step.
The example metadata.csv contains the image column `fov`, a column denoting which tissue sample an image belongs to `Tissue_ID`,	
a column denoting which patient the tissue belongs to `Patient_ID`, information on which timepoint the sample was collected at `Timepoint`, and 
the location of the sample `Localization`.

### Conversion Steps
If your single cell table is in csv format, it can be quickly converted to anndata using the below code.

Step 1: Read in the data files.
```commandline
import os
import anndata
import pandas as pd

# read in data
data_dir = '/Documents/SpaceCat/data'
cell_table = pd.read_csv(os.path.join(data_dir, 'cell_table.csv'))
```

Step 2: Provide the necessary input variables.

The required variables are:
* `markers`: list of columns in your cell table representing marker intensity 
* `centroid_cols`: list of two columns that denote the centroid values of each cell
* `cell_data_cols`: list including the necessary columns [above](#cell-table-format), as well as any other columns from the table containing cell level features you would like to include 

```commandline
# define column groupings
markers = ['Ki67', 'CD38', 'CD45RB', 'CD45RO']
centroid_cols = ['centroid-0', 'centroid-1']
cell_data_cols = ['fov', 'label', 'cell_meta_cluster', 'cell_cluster', 'cell_cluster_broad', 
                  'compartment_example', 'compartment_area_example', 'area', 'major_axis_length', 'cell_size']
```

Step 3: Create the anndata object and save.
```commandline
# create anndata from table subsetted for marker info, which will be stored in adata.X
adata = anndata.AnnData(cell_table.loc[:, markers])

# store all other cell data in adata.obs
adata.obs = cell_table.loc[:, cell_data_cols, ]
adata.obs_names = [str(i) for i in adata.obs_names]

# store cell centroid data in adata.obsm
adata.obsm['spatial'] = cell_table.loc[:, centroid_cols].values

# save the anndata object
os.makedirs(os.path.join(data_dir, 'adata'), exist_ok=True)
adata.write_h5ad(os.path.join(data_dir, 'adata', 'adata.h5ad'))
```

Once your data is stored in an anndata object, simply read in the single cell data with 
```commandline
adata = anndata.read_h5ad(os.path.join(data_dir, 'adata', 'adata.h5ad'))
```

## Preprocessing
Your single cell data will require some preprocessing before SpaceCat can generate features. 
This preprocessed anndata output will only need to be generated once and saved. Preprocessing consists of two steps, checking 
functional marker positivity within cells and assigning each cell to a compartment region based on any provided masks.

### Functional Marker Positivity 
**You will need to provide appropriate thresholds to indicate whether a cell is positive for each marker. 
This information will be used to generate functional marker [features](#feature-descriptions) in SpaceCat.**
```commandline
functional_marker_thresholds = [['Ki67', 0.002], ['CD38', 0.004], ['CD45RB', 0.001], ['CD45RO', 0.002]]
```

### Compartment Cell Assignment
Provided compartment masks must be individual binary tiffs per compartment, with the file name indicating the compartment
name you would like to use for feature generation; example compartment masks can be found in [data/compartment_masks](https://github.com/angelolab/SpaceCat/tree/main/data/compartment_masks). 
You can use the [Generalized Masking script](https://github.com/angelolab/ark-analysis/blob/main/templates/Generalized_Masking.ipynb) in ark-analysis to create channel and/or cell masks for your data. 
We recommend you perform additional processing to ensure your compartment masks do not overlap.

The required variables are:
* `image_key`, `seg_label_key` as described [above](#cell-table-format)
* `seg_dir`: the directory where the segmentation masks for each image are stored
* `mask_dir`: the directory where the various compartment masks are stored in a single folder for each image
* `seg_mask_substr`: the substring indicating the file is a segmentation mask, in the example data note the files are 
of the form 'TONIC_TMA4_R9C6_whole_cell.tiff', where '_whole_cell' is the `seg_mask_substr`
  * if your segmentation mask file names are simply the image names and have no suffix, set  `seg_mask_substr=None`

**If you do not have compartment masks set the `seg_dir` and `mask_dir` variables to `None`, so cell to 
compartment assignment can be skipped.**

Then, you can run the preprocessing function and save the resulting anndata.
```commandline
from SpaceCat.preprocess import preprocess_table

adata_processed = preprocess_table(adata, functional_marker_thresholds, image_key='fov', 
                                   seg_label_key='label', seg_dir=seg_dir, mask_dir=mask_dir,
                                   seg_mask_substr='_whole_cell')
adata_processed.write_h5ad(os.path.join(data_dir, 'adata', 'adata_processed.h5ad'))
```

## Feature Generation
### Running SpaceCat
Once we have the appropriate input, we can set the required parameters and generate some features!

The required variables are:
* `image_key`, `seg_label_key`, `cell_area_key`,  `cluster_key` as described [above](#cell-table-format)
* `functional_feature_level`: which of the cluster levels to generate functional features for
* `diversity_feature_level`: which of the cluster levels to generate cell diversity features for
* `pixel_radius`: radius in pixels which will be used to define the neighbors of each cell, for the example data 50 pixels (~20 microns) is used

Optional variables are:
* `compartment_key`: column name in .obs which contains cell assignments to various region types in the image, will be 'compartment' if cells were assigned using the preprocessing step
* `compartment_area_key`: column name in .obs which stores the compartment areas, will be 'compartment_area' if cells were assigned using the preprocessing step
* `specified_ratios_cluster_key`: the cluster level you wish to compute specific cell ratios for, in the example below: `'cell_cluster'`
  * ratios are already calculated across all pairwise combinations of your most broad cell cluster, this is an optional additional ratio specification
* `specified_ratios`: a list of cell type pairs to compute ratios for, all specified cell types must belong to the `specified_ratios_cluster_key` classification, see below for an example
* `per_cell_stats`: list of specifications so SpaceCat can pull additional features from the cell table, there are 3 required inputs for each cell stat list:
  * 1: the category name you would like to give the set of features, in the example below: `'morphology'`
  * 2: the cell cluster level to calculate this statistic at, in the example below: `'cell_cluster'`
  * 3: the list of columns in the cell table for each statistic you would like to include, in the example below: `['area', 'major_axis_length']`
* `per_img_stats`: list of specifications so SpaceCat can include any additional image level features,  there are 2 required inputs for each image stat list:
  * 1: the category name you would like to give the set of features, in the example below: `'fiber'` or `'mixing_score'`
  * 2: the dataframe containing the image level stats, one column must be the `image_key`,  while the other columns will indicate individual feature names to be included

When provided with compartment information, SpaceCat will calculate region specific features for your data, as well as at the image level.

**If you do not have compartment assignments and areas for each cell, set both of these variables to `None` to direct
SpaceCat to compute only the image level features.**

**If you do not have an additional `per_cell_stats` or `per_img_stats` then you can exclude these from the `run_spacecat()` function call.  
You can also exclude the `specified_ratios_cluster_key` and `specified_ratios` variables if you are not interested in this feature.**
```commandline
from SpaceCat.features import SpaceCat

# Initialize the class
adata_processed = anndata.read_h5ad(os.path.join(data_dir, 'adata', 'adata_processed.h5ad'))

features = SpaceCat(adata_processed, image_key='fov', seg_label_key='label', cell_area_key='area',
                    cluster_key=['cell_cluster', 'cell_cluster_broad'], 
                    compartment_key='compartment', compartment_area_key='compartment_area')
```
```commandline
# read in image level dataframes
fiber_df = pd.read_csv(os.path.join(data_dir, 'fiber_stats_table.csv'))
mixing_df = pd.read_csv(os.path.join(data_dir, 'mixing_scores.csv'))

# specify cell type pairs to compute a ratio for
ratio_pairings = [('CD8T', 'CD4T'), ('CD4T', 'Treg'), ('CD8T', 'Treg'), ('CD68_Mac', 'CD163_Mac')]

# specify addtional per cell and per image stats
per_cell_stats=[
    ['morphology', 'cell_cluster', ['area', 'major_axis_length']]
]
per_img_stats=[
    ['fiber', fiber_df], 
    ['mixing_score', mixing_df]
]

# Generate features and save anndata
adata_processed = features.run_spacecat(functional_feature_level='cell_cluster', diversity_feature_level='cell_cluster', pixel_radius = 50,
                                        specified_ratios_cluster_key='cell_cluster', specified_ratios=ratio_pairings,
                                        per_cell_stats=per_cell_stats, per_img_stats=per_img_stats)

adata_processed.write_h5ad(os.path.join(data_dir, 'adata', 'adata_processed.h5ad'))
```

### Output Tables
The output feature tables are stored in adata_processed.uns, and can be saved out to csvs if preferred.
```commandline
# Save finalized tables to csv 
os.makedirs(os.path.join(data_dir, 'SpaceCat'), exist_ok=True)

adata_processed.uns['combined_feature_data'].to_csv(os.path.join(data_dir, 'SpaceCat', 'combined_feature_data.csv'), index=False)
adata_processed.uns['combined_feature_data_filtered'].to_csv(os.path.join(data_dir, 'SpaceCat', 'combined_feature_data_filtered.csv'), index=False)
adata_processed.uns['feature_metadata'].to_csv(os.path.join(data_dir, 'SpaceCat', 'feature_metadata.csv'), index=False)
adata_processed.uns['excluded_features'].to_csv(os.path.join(data_dir, 'SpaceCat', 'excluded_features.csv'), index=False)
```

### Adding Metadata
If you would like to merge your metadata into any of the above tables, you can do so using the shared image key
```commandline
metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
features_filtered = pd.read_csv(os.path.join(data_dir, 'SpaceCat', 'combined_feature_data_filtered.csv'))

# merge metadata into output table
merge_table = pd.merge(features_filtered, metadata, on=['fov'])
```

You can also add the metadata in your anndata object and save with
```commandline
adata_processed.uns['metadata'] = metadata
adata_processed.write_h5ad(os.path.join(data_dir, 'adata', 'adata_processed.h5ad'))
```

### Feature Descriptions
All features are computed separately in each image. In addition, if you provided optional compartment assignments, the features will also be computed within each compartment.
- `density`: The number of cells divided by the area of the region.
- `density_ratio`: The ratio between the densities of cell types. This is done for every pairwise combination of cells at the broadest level of clustering.
- `functional_marker`: For each cell type/functional marker combination, the proportion of cells positive for that marker, using the supplied marker-specific thresholds to determine positivity.
- `cell_diversity`: This diversity feature is based on cell proximity. For each cell in the image, the proportions of each cell type within a specified pixel radius was computed. Then the Shannon diversity index was calculated on these proportions.
- `region_diversity`: This diversity feature is based on cell abundance. For the broadest cell cluster level, the proportion of cells of each lower cell type was extracted. We then compute the diversity of cell types within each broad category using the Shannon Diversity index.
  - This feature was computed for cells at a broad level of clustering that were composed of at least two distinct lower cell types.
- `density_proportion`: For each lower level cell type in a given broader cell type category, the proportion of the number of broader cells that the lower cell type represented was calculated. 
  - This feature was computed for cells at a broad level of clustering that were composed of at least two distinct lower cell types.

Coming soon:
- `kmeans_cluster`: Using k-means clustering to define cell neighborhoods in each image, we then calculated the proportion of cells belonging to each of the identified clusters across the region.
