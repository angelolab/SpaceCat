import numpy as np
import pandas as pd

from itertools import combinations
from scipy.stats import spearmanr
from alpineer.misc_utils import verify_in_list


class FeatureSpace:

    def __init__(self, adata_table, image_col, label_col, cluster_columns, compartment_col):
        self.adata_table = adata_table.copy()
        self.image_col = image_col
        self.label_col = label_col
        self.cluster_columns = cluster_columns
        self.compartment_col = compartment_col
        self.compartment_list = list(np.unique(adata_table.obs[compartment_col])) + ['all']

        self.feature_data_list = []
        self.combined_feature_data = None
        self.excluded_features = None
        self.combined_feature_data_filtered = None
        self.feature_metadata = None

    def exclude_empty_compartments(self, table):
        compartment_counts = table[[self.image_col, 'subset', 'value']].groupby(
            by=[self.image_col, 'subset'], observed=True).sum().reset_index()
        exclude_compartments = compartment_counts[compartment_counts.value != 0].\
            drop(columns=['value'])
        table = pd.merge(table, exclude_compartments, how='inner')

        return table

    def cluster_df_helper(self, table, cluster_col_name, drop_cols, result_name, var_name,
                          cluster_stats, normalize):
        """Function to summarize input data by cell type

        Args:
            table (pd.DataFrame): table containing input data
            cluster_col_name (str): name of the column that contains the cluster information
            drop_cols (list): list of columns to drop from the table
            result_name (str): name of the statistic in the summarized information df
            var_name (str): name of the column that will containin the computed values
            cluster_stats (bool): whether we are calculating cluster counts and frequencies
            normalize (bool): whether to report the total or normalized counts in the result

        Returns:
            pd.DataFrame: long format dataframe containing the summarized data"""

        verify_in_list(cell_type_col=cluster_col_name, cell_table_columns=table.columns)
        verify_in_list(drop_cols=drop_cols, cell_table_columns=table.columns)

        # drop columns from table
        table_small = table.loc[:, ~table.columns.isin(drop_cols)]

        # group by specified columns
        grouped_table = table_small.groupby([self.image_col, cluster_col_name], observed=True)
        if cluster_stats:
            # transformed = grouped_table.agg('count')
            counts = grouped_table[cluster_col_name].value_counts(normalize=normalize)
            transformed = counts.unstack(level=cluster_col_name, fill_value=0).stack()
            transformed = transformed.reset_index()
            long_df = transformed.rename(columns={cluster_col_name: 'cell_type', 0: 'value'})
        else:
            if normalize:
                transformed = grouped_table.agg('mean')
            else:
                transformed = grouped_table.agg('sum')
            transformed.reset_index(inplace=True)
            long_df = pd.melt(
                transformed, id_vars=[self.image_col, cluster_col_name], var_name=var_name)
            long_df = long_df.rename(columns={cluster_col_name: 'cell_type'})

        long_df['metric'] = result_name

        return long_df

    def create_long_df(self, table, cluster_col_name, result_name, var_name, subset_col=None,
                       cluster_stats=False, normalize=False, drop_cols=[],
                       exclude_missing_compartments=True):
        """Summarize functional marker positivity by cell type, with the option to subset by an additional feature

        Args:
            table (pd.DataFrame): the dataframe containing information on each cell
            cluster_col_name (str): the column name in cell_table that contains the cluster information
            drop_cols (list): list of columns to drop from cell_table
            result_name (str): the name of this statistic in the returned df
            var_name (str): name of the column that will containin the computed values
            subset_col (str): the column name in cell_table to subset by
            normalize (bool): whether to report the total or normalized counts in the result

        Returns:
            pd.DataFrame: long format dataframe containing the summarized data"""

        # first generate df without subsetting
        drop_cols_all = drop_cols.copy()
        if subset_col is not None:
            drop_cols_all = drop_cols_all + [subset_col]

        long_df_all = self.cluster_df_helper(table, cluster_col_name, drop_cols_all, result_name,
                                             var_name, cluster_stats, normalize)
        long_df_all['subset'] = 'all'

        # if a subset column is specified, create df stratified by subset
        if subset_col is not None:
            verify_in_list(subset_col=subset_col, cell_table_columns=table.columns)

            # drop columns from table
            table_small = table.loc[:, ~table.columns.isin(drop_cols)]

            # group by specified columns
            grouped_table = table_small.groupby(
                [self.image_col, subset_col, cluster_col_name], observed=True)

            if cluster_stats:
                # transformed = grouped_table.agg('count')
                counts_vals = grouped_table[cluster_col_name].value_counts(normalize=normalize)
                transformed = counts_vals.unstack(level=cluster_col_name, fill_value=0).stack()
                transformed = transformed.reset_index()
                long_df = transformed.rename(
                    columns={cluster_col_name: 'cell_type', 0: 'value', subset_col: 'subset'})
            else:
                if normalize:
                    transformed = grouped_table.agg('mean')
                else:
                    transformed = grouped_table.agg('sum')
                transformed.reset_index(inplace=True)
                # reshape to long df
                long_df = pd.melt(
                    transformed, id_vars=[self.image_col, subset_col, cluster_col_name],
                    var_name=var_name)
                long_df = long_df.rename(columns={cluster_col_name: 'cell_type',
                                                  self.label_col: 'value', subset_col: 'subset'})

            if cluster_stats:
                if exclude_missing_compartments:
                    long_df = self.exclude_empty_compartments(long_df)

            # combine the two dataframes
            long_df['metric'] = result_name
            long_df_all = pd.concat([long_df_all, long_df], axis=0, ignore_index=True)

        return long_df_all

    def generate_cluster_stats(self, cell_table_clusters, cluster_df_params, comparmtent_area_df,
                               exclude_missing_compartments=True):
        cluster_dfs = []
        for result_name, cluster_col_name, normalize in cluster_df_params:
            drop_cols = []
            # remove cluster_names except for the one specified for the df
            cluster_names = self.cluster_columns.copy()
            cluster_names.remove(cluster_col_name)
            drop_cols.extend(cluster_names)

            cluster_dfs.append(self.create_long_df(table=cell_table_clusters,
                                                   cluster_col_name=cluster_col_name,
                                                   result_name=result_name,
                                                   var_name='cell_type',
                                                   subset_col=self.compartment_col,
                                                   cluster_stats=True,
                                                   normalize=normalize,
                                                   drop_cols=drop_cols))

        # calculate total number of cells per image
        grouped_cell_counts = cell_table_clusters[[self.image_col]].groupby(
            self.image_col, observed=True).value_counts()
        grouped_cell_counts = pd.DataFrame(grouped_cell_counts)
        grouped_cell_counts.columns = ['value']
        grouped_cell_counts.reset_index(inplace=True)
        grouped_cell_counts['metric'] = 'total_cell_count'
        grouped_cell_counts['cell_type'] = 'all'
        grouped_cell_counts['subset'] = 'all'

        # calculate total number of cells per region per image
        grouped_cell_counts_region = cell_table_clusters[[self.image_col, self.compartment_col]].\
            groupby([self.image_col, self.compartment_col], observed=True).value_counts()
        grouped_cell_counts_region = pd.DataFrame(grouped_cell_counts_region)
        grouped_cell_counts_region.columns = ['value']
        grouped_cell_counts_region.reset_index(inplace=True)
        grouped_cell_counts_region['metric'] = 'total_cell_count'
        grouped_cell_counts_region.rename(columns={self.compartment_col: 'subset'}, inplace=True)
        grouped_cell_counts_region['cell_type'] = 'all'
        if exclude_missing_compartments:
            grouped_cell_counts_region = self.exclude_empty_compartments(grouped_cell_counts_region)

        # calculate proportions of cells per region per image
        grouped_cell_freq_region = cell_table_clusters[[self.image_col, self.compartment_col]].\
            groupby([self.image_col], observed=True)
        grouped_cell_freq_region = grouped_cell_freq_region[self.compartment_col].\
            value_counts(normalize=True)
        grouped_cell_freq_region = pd.DataFrame(grouped_cell_freq_region)
        grouped_cell_freq_region.columns = ['value']
        grouped_cell_freq_region.reset_index(inplace=True)
        grouped_cell_freq_region['metric'] = 'total_cell_freq'
        grouped_cell_freq_region.rename(columns={self.compartment_col: 'subset'}, inplace=True)
        grouped_cell_freq_region['cell_type'] = 'all'
        if exclude_missing_compartments:
            grouped_cell_freq_region = self.exclude_empty_compartments(grouped_cell_freq_region)

        # add manually defined dfs to overall list
        cluster_dfs.extend([grouped_cell_counts,
                            grouped_cell_counts_region,
                            grouped_cell_freq_region])

        # create single df with appropriate metadata
        total_df_clusters = pd.concat(cluster_dfs, axis=0)

        # compute density of cells for counts-based metrics
        count_metrics = total_df_clusters.metric.unique()
        count_metrics = [x for x in count_metrics if 'count' in x]

        count_df = total_df_clusters.loc[total_df_clusters.metric.isin(count_metrics), :]
        comparmtent_area_df = comparmtent_area_df.rename(columns={self.compartment_col: 'subset'})
        all_area = comparmtent_area_df[[self.image_col, self.compartment_col + '_area']].groupby(
            by=[self.image_col]).sum().reset_index()
        all_area['subset'] = 'all'
        area_df = pd.concat([comparmtent_area_df, all_area])
        count_df = count_df.merge(area_df, on=[self.image_col, 'subset'], how='left')
        count_df['value'] = count_df['value'] / count_df[self.compartment_col + '_area']
        count_df['value'] = count_df['value'] * 1000

        # rename metric from count to density
        count_df['metric'] = count_df['metric'].str.replace('count', 'density')
        count_df = count_df.drop(columns=[self.compartment_col + '_area'])
        total_df_clusters = pd.concat([total_df_clusters, count_df], axis=0)

        self.adata_table.uns['cluster_stats'] = total_df_clusters
        return total_df_clusters

    def format_helper(self, compartment_df, compartment, cell_pop_level, feature_type):
        if compartment == 'all':
            compartment_df['feature_name_unique'] = compartment_df['feature_name']
        else:
            compartment_df['feature_name_unique'] = compartment_df['feature_name'] + '__' + \
                                                    compartment

        compartment_df[self.compartment_col] = compartment
        compartment_df['cell_pop_level'] = cell_pop_level
        compartment_df['feature_type'] = feature_type
        compartment_df = compartment_df[
            [self.image_col, 'value', 'feature_name', 'feature_name_unique', self.compartment_col,
             'cell_pop_level', 'feature_type']]
        ## TO DO: add in cell_pop ??

        return compartment_df

    def generate_high_level_stats(self, stats_df, abundance_params, ratio_params,
                                  minimum_density=0.0005):

        # add total density stats to list
        abundance_params.append(['total_cell_density', 'total_density', 'total'])

        # compute abundance of cell types
        for cluster_name, feature_name, cell_pop_level in abundance_params:
            input_df = stats_df[stats_df['metric'].isin([cluster_name])]
            for compartment in self.compartment_list:
                compartment_df = input_df[input_df.subset == compartment].copy()
                compartment_df['feature_name'] = compartment_df.cell_type + '__' + feature_name

                compartment_df_formatted = self.format_helper(
                    compartment_df, compartment, cell_pop_level, feature_type='density')
                self.feature_data_list.append(compartment_df_formatted)

        # compute ratio of broad cell type abundances
        ratio_cluster_level, cell_pop_level = ratio_params
        input_df = stats_df[stats_df['metric'].isin([ratio_cluster_level])]
        for compartment in self.compartment_list:
            compartment_df = input_df[input_df.subset == compartment].copy()
            cell_types = compartment_df.cell_type.unique()

            for cell_type1, cell_type2 in combinations(cell_types, 2):
                cell_type1_df = compartment_df[compartment_df.cell_type == cell_type1].copy()
                cell_type2_df = compartment_df[compartment_df.cell_type == cell_type2].copy()

                # only keep FOVS with at least one cell type over the minimum density
                cell_type1_mask = cell_type1_df.value > minimum_density
                cell_type2_mask = cell_type2_df.value > minimum_density
                cell_mask = cell_type1_mask.values | cell_type2_mask.values
                cell_type1_df = cell_type1_df[cell_mask]
                cell_type2_df = cell_type2_df[cell_mask]

                # add minimum density to avoid log2(0)
                cell_type1_df['ratio'] = np.log2((cell_type1_df.value.values + minimum_density) /
                                                 (cell_type2_df.value.values + minimum_density))
                cell_type1_df['value'] = cell_type1_df.ratio.values
                cell_type1_df['feature_name'] = cell_type1 + '__' + cell_type2 + '__ratio'

                cell_type1_df_formatted = self.format_helper(
                    cell_type1_df, compartment, cell_pop_level, feature_type='density_ratio')
                self.feature_data_list.append(cell_type1_df_formatted)

    def remove_correlated_features(self, feature_df, correlation_thresh):
        # filter FOV features based on correlation in compartments
        feature_df = self.combined_feature_data

        # filter out features that are highly correlated in compartments
        feature_names = feature_df.feature_name.unique()
        exclude_list = []

        ## ADJUST THIS
        # set minimum number of FOVs for compartment feature
        min_fovs = 100

        for feature_name in feature_names:
            fov_data_feature = feature_df.loc[feature_df.feature_name == feature_name, :]

            # get the compartments present for this feature
            compartments = fov_data_feature.compartment.unique()

            # if only one compartment, skip
            if len(compartments) == 1:
                continue

            fov_data_wide = fov_data_feature.pivot(
                index=self.image_col, columns=self.compartment_col, values='raw_value')

            # filter out features that are nans or mostly zeros
            for compartment in compartments:
                nan_count = fov_data_wide[compartment].isna().sum()
                zero_count = (fov_data_wide[compartment] == 0).sum()

                if len(fov_data_wide) - nan_count - zero_count < min_fovs:
                    exclude_list.append(feature_name + '__' + compartment)
                    fov_data_wide = fov_data_wide.drop(columns=compartment)

            # compute correlations
            compartments = fov_data_wide.columns
            compartments = compartments[compartments != 'all']

            for compartment in compartments:
                corr, _ = spearmanr(fov_data_wide['all'].values, fov_data_wide[compartment].values,
                                    nan_policy='omit')
                if corr > correlation_thresh:
                    exclude_list.append(feature_name + '__' + compartment)

        # remove features from dataframe
        exclude_df = pd.DataFrame({'feature_name_unique': exclude_list})
        self.excluded_features = exclude_df
        feature_df_filtered = \
            feature_df.loc[~feature_df.feature_name_unique.isin(
                exclude_df.feature_name_unique.values), :]

        return feature_df_filtered

    def combine_features(self, correlation_filtering=0.7):
        # compute z-scores for each feature
        feature_df = pd.concat(self.feature_data_list).reset_index(drop=True)
        feature_df = feature_df.rename(columns={'value': 'raw_value'})
        feature_df_wide = feature_df.pivot(
            index=self.image_col, columns='feature_name_unique', values='raw_value')
        zscore_df = (feature_df_wide - feature_df_wide.mean()) / feature_df_wide.std()

        # add z-scores to feature_df
        zscore_df = zscore_df.reset_index()
        zscore_df_long = pd.melt(zscore_df, id_vars=self.image_col, var_name='feature_name_unique',
                                 value_name='normalized_value')
        feature_df = pd.merge(
            feature_df, zscore_df_long, on=[self.image_col, 'feature_name_unique'], how='left')

        # rearrange columns
        feature_df = feature_df[[self.image_col, 'raw_value', 'normalized_value', 'feature_name',
                                 'feature_name_unique', self.compartment_col, 'cell_pop_level',
                                 'feature_type']]

        ## TO DO ##
        # add metadata

        self.combined_feature_data = feature_df
        self.adata_table.uns['combined_feature_data'] = feature_df
        feature_metadata = feature_df[['feature_name', 'feature_name_unique', self.compartment_col,
                                       'cell_pop_level', 'feature_type']]

        if correlation_filtering:
            feature_df_filtered = self.remove_correlated_features(
                feature_df, correlation_filtering)
            self.combined_feature_data_filtered = feature_df_filtered
            self.adata_table.uns['combined_feature_data_filtered'] = feature_df

            feature_metadata = feature_df_filtered[['feature_name', 'feature_name_unique',
                                                    self.compartment_col, 'cell_pop_level',
                                                    'feature_type']]

        # save feature metadata
        feature_metadata = feature_metadata.drop_duplicates()
        self.feature_metadata = feature_metadata
        self.adata_table.uns['feature_metadata'] = feature_metadata

    def run_spacecat(self):
        # Generate counts and proportions of cell clusters per FOV
        cluster_params = []
        for column in self.cluster_columns:
            cluster_params.append([column + '_freq', column, True])
            cluster_params.append([column + '_count', column, False])

        # subset table for cluster data
        cell_table_clusters = self.adata_table.obs[
            [self.image_col, self.label_col, self.compartment_col] + self.cluster_columns]
        compartment_area_df = self.adata_table.obs[
            [self.image_col, self.compartment_col, self.compartment_col + '_area']].\
            drop_duplicates()
        output = self.generate_cluster_stats(
            cell_table_clusters, cluster_params, compartment_area_df)

        # high level cluster stats
        stats_df = self.adata_table.uns['cluster_stats']
        abundance_params = []
        for column in self.cluster_columns:
            abundance_params.append([column + '_density', column + '_density', column])

        # TO DO: generalize this
        ratio_params = ['cell_cluster_broad_density', 'cell_cluster_broad']

        self.generate_high_level_stats(stats_df, abundance_params, ratio_params)
        self.combine_features()

        return self.adata_table
