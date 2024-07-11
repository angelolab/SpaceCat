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

We are currently in the process of pulling out the key components of the SpaceCat pipeline into this independent repository. Right now, the code is baked into the scripts for the paper. If you're curious, you can take a look and get started here: https://github.com/angelolab/publications/tree/main/2024-Greenwald_Nederlof_etal_TONIC
