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


## Installation
You can install the package with pip by
```commandline
pip install SpaceCat
```

## Example Usage
Once your data is stored in an anndata object, simply read in the single cell data with 
```commandline
import os
import anndata

data_dir = 'path/to/data'
adata = anndata.read_h5ad(os.path.join(data_dir, 'adata', 'adata.h5ad'))
```
If your data is in csv format and needs to be converted, see [AnnData Conversion](#anndata-conversion) below.

### Preprocessing
Your single cell data, will require some preprocessing before SpaceCat can generate features. 
This preprocessed anndata output will only need to be generated once and saved.

**You will need to provide appropriate thresholds to indicate whether a cell is positive for each marker. 
This information will be used to generate functional marker features in SpaceCat.**
```commandline
from SpaceCat.preprocess import preprocess_table

functional_marker_thresholds = [['Ki67', 0.002], ['CD38', 0.004], ['CD45RB', 0.001], ['CD45RO', 0.002]]

adata_processed = preprocess_table(adata, functional_marker_thresholds)
adata_processed.write_h5ad(os.path.join(data_dir, 'adata', 'adata_processed.h5ad'))
```

### Feature Generation
Once we have the appropriate input, we can set the required parameters and generate some features!

The required variables are:
* `image_key`: column name in .obs which denotes the image name
* `seg_label_key`: column name in .obs which denotes the cell segmentation label
* `cell_area_key`: column name in .obs which denotes the cell area
* `cluster_key`: list of column names in .obs containing the various cell assignments
* `functional_feature_level`: which of the cluster levels to generate functional features for

Optional variables are:
* `compartment_key`: column name in .obs which contains cell assignments to various region types in the image
* `compartment_area_key`: column name in .obs which stores the compartment areas

When provided with compartment information, SpaceCat will calculate region specific features for your data, as well as at the image level.

**If you do not have compartment assignments and areas for each cell, set both of these variables to `None` to direct
SpaceCat to compute only the image level features.**
```commandline
from SpaceCat.features import SpaceCat

# Initialize class
adata_processed = anndata.read_h5ad(os.path.join(data_dir, 'adata', 'adata_processed.h5ad'))

features = SpaceCat(adata_processed, image_key='fov', seg_label_key='label', cell_area_key='area',
                    cluster_key=['cell_cluster', 'cell_cluster_broad'], 
                    compartment_key='compartment', compartment_area_key='compartment_area')


# Generate features and save anndata
adata_processed = features.run_spacecat(functional_feature_level='cell_cluster')

adata_processed.write_h5ad(os.path.join(data_dir, 'adata', 'adata_processed.h5ad'))
```

### Output
The output feature tables are stored in adata_processed.uns, and can be saved out to csvs if preferred.
```commandline
# Save finalized tables to csv 
os.makedirs(os.path.join(data_dir, 'SpaceCat'), exist_ok=True)

adata_processed.uns['combined_feature_data'].to_csv(os.path.join(data_dir, 'SpaceCat', 'combined_feature_data.csv'), index=False)
adata_processed.uns['combined_feature_data_filtered'].to_csv(os.path.join(data_dir, 'SpaceCat', 'combined_feature_data_filtered.csv'), index=False)
adata_processed.uns['feature_metadata'].to_csv(os.path.join(data_dir, 'SpaceCat', 'feature_metadata.csv'), index=False)
adata_processed.uns['excluded_features'].to_csv(os.path.join(data_dir, 'SpaceCat', 'excluded_features.csv'), index=False)
```

### AnnData Conversion
If your single cell table is in csv format, it can be quickly converted to anndata using the below code.

Provide the necessary input variables.
```commandline
# read in data
data_dir = 'path/to/data'
cell_table = pd.read_csv('path/to/cell_table.csv')
metadata = pd.read_csv('path/to/metadata_table.csv')

# merge metadata into cell table if needed
cell_table = pd.merge(cell_table, metadata, on=['fov'])

# define column groupings
markers = ['Ki67', 'CD38', 'CD45RB', 'CD45RO']
cell_data_cols = ['fov', 'label', 'cell_meta_cluster', 'cell_cluster', 'cell_cluster_broad', 
                  'compartment', 'compartment_area', 'area', 'cell_size', 'metadata_colum1', 'metadata_colum2']
centroid_cols = ['centroid-0', 'centroid-1']
```
Create the anndata object and save.
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