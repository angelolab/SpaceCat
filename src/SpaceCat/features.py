import numpy as np
import pandas as pd

from itertools import combinations
from scipy.stats import spearmanr
from alpineer.misc_utils import verify_in_list


class SpaceCat:

    def __init__(self, adata_table, image_key, seg_label_key, cell_area_key, cluster_key,
                 compartment_key, compartment_area_key):
        self.adata_table = adata_table.copy()
        self.image_key = image_key
        self.seg_label_key = seg_label_key
        self.cell_area_key = cell_area_key
        self.cluster_key = cluster_key
        self.compartment_key_none = False if compartment_key else True
        self.compartment_key = compartment_key if compartment_key else 'compartment'
        self.compartment_area_key = compartment_area_key if compartment_area_key else 'compartment_area'
        self.compartment_list = list(
            np.unique(adata_table.obs[compartment_key])) if not self.compartment_key_none else []
        self.compartment_list = self.compartment_list + ['all']

        # validation checks
        verify_in_list(provided_columns=[self.image_key, self.seg_label_key],
                       columns_in_table=self.adata_table.obs.columns)
        verify_in_list(provided_cluster_columns=self.cluster_key,
                       columns_in_table=self.adata_table.obs.columns)
        if not self.compartment_key_none:
            verify_in_list(provided_columns=[self.compartment_key, self.compartment_area_key],
                           columns_in_table=self.adata_table.obs.columns)

        self.feature_data_list = []
        self.combined_feature_data = None
        self.excluded_features = None
        self.combined_feature_data_filtered = None
        self.feature_metadata = None

    ## HELPER FUNCTIONS ##
    def exclude_empty_compartments(self, table):
        """ Remove rows containing a compartment that does not exist in the image.
        Args:
            table (pd.DataFrame): contains the features calculated within each compartment

        Returns:
            pd.DataFrame:
                table with any compartments not contained in the image excluded
        """
        # calculate which compartment have zero values for every cell type
        compartment_counts = table[[self.image_key, 'subset', 'value']].groupby(
            by=[self.image_key, 'subset'], observed=True).sum().reset_index()
        include_compartments = compartment_counts[compartment_counts.value != 0].\
            drop(columns=['value'])

        # drop regions that do not exist in the image from the feature table
        table = pd.merge(table, include_compartments, how='inner')

        return table

    def calculate_density_stats(self, total_df_clusters, compartment_area_df):
        """Function to calculate density statistics based on cell counts and compartment area
        Args:
            total_df_clusters (pd.DataFrame): table containing cluster stats
            compartment_area_df (pd.DataFrame): the dataframe containing the areas

        Returns:
            pd.DataFrame:
                table with the frequencies of each cell type in the compartments / image
        """

        # retrieve counts values
        count_metrics = total_df_clusters.metric.unique()
        count_metrics = [x for x in count_metrics if 'count' in x]
        count_df = total_df_clusters.loc[total_df_clusters.metric.isin(count_metrics), :]

        # calculate density stats
        compartment_area_df = compartment_area_df.rename(columns={self.compartment_key: 'subset'})
        density_df = count_df.merge(compartment_area_df, on=[self.image_key, 'subset'], how='left')
        density_df['value'] = density_df['value'] / density_df[self.compartment_area_key] * 1000

        # rename metric from count to density
        density_df['metric'] = density_df['metric'].str.replace('count', 'density')
        density_df = density_df.drop(columns=[self.compartment_area_key])

        return density_df

    def get_frequencies(self, counts_df, groupby_cols):
        """Function to calculate frequencies based on count values.
        Args:
            counts_df (pd.DataFrame): table containing cell counts data
            groupby_cols (str): list of table columns to groupby

        Returns:
            pd.DataFrame:
                dataframe containing the frequencies of cells
        """
        # total cell counts in image
        total_counts = counts_df[groupby_cols + [self.seg_label_key]]. \
            groupby(groupby_cols, observed=False).sum().reset_index()
        total_counts = total_counts.rename(columns={self.seg_label_key: 'total_counts'})

        # get frequencies by dividing cell type counts by total counts
        transformed = counts_df.merge(total_counts, on=groupby_cols, how='left')
        transformed[self.seg_label_key] = transformed[self.seg_label_key] / transformed['total_counts']
        transformed = transformed.drop(columns=['total_counts'])

        return transformed

    def get_compartment_areas(self):
        """ Helper function to get the appropriate compartment area table for the data.
        Returns:
            pd.DataFrame:
                the table with compartment areas, calculates all cell area if no compartments
        """
        if self.compartment_key_none:
            # calculate image wide area by cell area
            cell_area_df = self.adata_table.obs[[self.image_key, self.cell_area_key]]
            area_df = cell_area_df.groupby(by=[self.image_key]).sum().reset_index()
            area_df[self.compartment_key] = 'all'
            area_df = area_df.rename(columns={self.cell_area_key: self.compartment_area_key})
        else:
            # subset df for compartment area
            compartment_area_df = self.adata_table.obs[
                [self.image_key, self.compartment_key, self.compartment_area_key]]. \
                drop_duplicates()

            # calculate total image area and append to area df
            all_area = compartment_area_df[[self.image_key, self.compartment_area_key]].groupby(
                by=[self.image_key]).sum().reset_index()
            all_area[self.compartment_key] = 'all'
            area_df = pd.concat([compartment_area_df, all_area])

        return area_df

    def long_df_helper(self, table, cluster_col_name, drop_cols, var_name, cluster_stats,
                       normalize, subset_col=None):
        """Function to summarize input data by cell type.
        Args:
            table (pd.DataFrame): table containing input data
            cluster_col_name (str): name of the column that contains the cluster information
            drop_cols (list): list of columns to drop from the table
            var_name (str): name of the column that will contain the computed values
            cluster_stats (bool): whether we are calculating cluster counts and frequencies
            normalize (bool): whether to report the total or normalized counts in the result
            subset_col (str): name of the image subset column, defaults to None

        Returns:
            pd.DataFrame:
                long format dataframe containing the summarized data
        """

        verify_in_list(cell_type_col=cluster_col_name, cell_table_columns=table.columns)
        verify_in_list(drop_cols=drop_cols, cell_table_columns=table.columns)
        verify_in_list(subset_col=subset_col, cell_table_columns=table.columns)

        # drop columns from table
        table_small = table.drop(columns=drop_cols, errors="ignore")

        # group by specified columns
        groupby_cols = [self.image_key, cluster_col_name, subset_col] if subset_col else \
            [self.image_key, cluster_col_name]
        grouped_table = table_small.groupby(groupby_cols, observed=False)

        if cluster_stats:
            # get cell counts
            transformed = grouped_table.count().reset_index()
            if normalize:
                # get cell frequencies
                groupby_cols.remove(cluster_col_name)
                transformed = self.get_frequencies(transformed, groupby_cols)

        else:
            # get sum or average of values
            transformed = grouped_table.agg('mean') if normalize else grouped_table.agg('sum')
            transformed.reset_index(inplace=True)

            # reshape to long df
            transformed = pd.melt(transformed, id_vars=groupby_cols, var_name=var_name)

        long_df = transformed.rename(
            columns={cluster_col_name: 'cell_type', self.seg_label_key: 'value', subset_col: 'subset'})

        return long_df

    def create_long_df(self, table, cluster_col_name, result_name, var_name, subset_col=None,
                       cluster_stats=False, normalize=False, drop_cols=None):
        """Summarize input data by cell type, with the option to subset by an additional feature.
        Args:
            table (pd.DataFrame): the dataframe containing information on each cell
            cluster_col_name (str): the column name in cell_table that contains the cluster info
            result_name (str): the name of this statistic in the returned df
            var_name (str): name of the column that will contain the computed values
            subset_col (str): the column name in cell_table to subset by
            cluster_stats (bool): whether we are calculating cluster counts and frequencies
            normalize (bool): whether to report the total or normalized counts in the result
            drop_cols (list): list of columns to drop from cell_table

        Returns:
            pd.DataFrame:
                long format dataframe containing the summarized data
        """
        # change none to empty list
        drop_cols = [] if not drop_cols else drop_cols

        # first generate df without subsetting
        drop_cols_all = drop_cols + [subset_col] if subset_col is not None else drop_cols.copy()
        long_df_all = self.long_df_helper(table, cluster_col_name, drop_cols_all, var_name,
                                          cluster_stats, normalize)
        long_df_all['metric'] = result_name
        long_df_all['subset'] = 'all'

        # if a subset column is specified, create df stratified by subset
        if subset_col is not None:
            long_df = self.long_df_helper(table, cluster_col_name, drop_cols, var_name,
                                          cluster_stats, normalize, subset_col=subset_col)

            # combine the two dataframes
            long_df['metric'] = result_name
            long_df_all = pd.concat([long_df_all, long_df], axis=0, ignore_index=True)

        return long_df_all

    def format_helper(self, compartment_df, compartment, cell_pop_level, feature_type):
        """ Add informative metadata columns to the feature dataframe.
        Args:
            compartment_df (pd.DataFrame): table with features for the specified compartment
            compartment (str): which compartment is being subsetted
            cell_pop_level (str): level of cell clustering for the feature
            feature_type (str): broad name of the feature type

        Returns:
            pd.DataFrame:
                table with feature metadata columns included
        """
        # add useful metadata about the feature
        if compartment == 'all':
            compartment_df['feature_name_unique'] = compartment_df['feature_name']
        else:
            compartment_df['feature_name_unique'] = compartment_df['feature_name'] + '__' + \
                                                    compartment

        compartment_df[self.compartment_key] = compartment
        compartment_df['cell_pop_level'] = cell_pop_level
        compartment_df['feature_type'] = feature_type
        compartment_df = compartment_df[
            [self.image_key, 'value', 'feature_name', 'feature_name_unique', self.compartment_key,
             'cell_pop_level', 'feature_type']]

        return compartment_df

    def format_computed_features(self, stats_df, col_name, metrics):
        """ Add relevant metadata to the features and append them to the final table.
        Args:
            stats_df (pd.DataFrame): table with the features
            col_name (str): name of the column storing the feature value
            metrics (list): list of which features to format and add to final table
        Returns:
            adds features to the final table
        """

        # takes in an input_df, and for each compartment, adds necessary feature information
        for metric_name, cell_pop_level in metrics:
            input_df = stats_df[stats_df['metric'].isin([metric_name])]
            for compartment in self.compartment_list:
                compartment_df = input_df[input_df.subset == compartment].copy()
                compartment_df['feature_name'] = compartment_df[col_name] + '__' + compartment_df.cell_type

                compartment_df_formatted = self.format_helper(compartment_df, compartment, cell_pop_level, col_name)

                # add to final dfs list
                self.feature_data_list.append(compartment_df_formatted)

    ## FEATURE GENERATION FUNCTIONS ##
    def generate_cluster_stats(self, cell_table_clusters, cluster_df_params, compartment_area_df,
                               exclude_missing_compartments=True):
        """ Create dataframe containing cell counts and frequency statistics.
        Args:
            cell_table_clusters (pd.DataFrame): the dataframe containing cell classifications
            cluster_df_params (list): list of which features to generate
            compartment_area_df (pd.DataFrame): the dataframe containing the areas of each
                compartment
            exclude_missing_compartments (bool): whether to exclude a compartment when it's not
                contained in the image, or to included as 0 value, defaults to True

        Returns:
            saves the cluster_stats to the class which contains counts, freqs, & densities of each
            cell type in the compartments/image, as well as stats across all cell types
        """
        subset_col = None if self.compartment_key_none else self.compartment_key

        cluster_dfs = []
        for result_name, cluster_col_name, normalize in cluster_df_params:
            drop_cols = []
            # remove cluster_names except for the one specified for the df
            cluster_names = self.cluster_key.copy()
            cluster_names.remove(cluster_col_name)
            drop_cols.extend(cluster_names)

            cluster_dfs.append(self.create_long_df(table=cell_table_clusters,
                                                   cluster_col_name=cluster_col_name,
                                                   result_name=result_name,
                                                   var_name='cell_type',
                                                   subset_col=subset_col,
                                                   cluster_stats=True,
                                                   normalize=normalize,
                                                   drop_cols=drop_cols))

        def add_feature_metadata(panda_series, metric, compartment_col=False):
            df = pd.DataFrame(panda_series)
            df.columns = ['value']
            df.reset_index(inplace=True)
            df['metric'] = metric
            df['cell_type'] = 'all'
            if compartment_col:
                df.rename(columns={self.compartment_key: 'subset'}, inplace=True)
            else:
                df['subset'] = 'all'

            return df

        # calculate total number of cells per image
        grouped_cell_counts = cell_table_clusters[[self.image_key]].groupby(
            self.image_key, observed=True).value_counts()
        grouped_cell_counts = add_feature_metadata(grouped_cell_counts, metric='total_cell_count')
        total_stats = [grouped_cell_counts]

        if not self.compartment_key_none:
            # calculate total number of cells per region per image
            grouped_cell_counts_region = cell_table_clusters[[self.image_key, self.compartment_key]].\
                groupby([self.image_key, self.compartment_key], observed=True).value_counts()
            grouped_cell_counts_region = add_feature_metadata(
                grouped_cell_counts_region, metric='total_cell_count', compartment_col=True)

            # calculate proportions of cells per region per image
            grouped_cell_freq_region = cell_table_clusters[[self.image_key, self.compartment_key]].\
                groupby([self.image_key], observed=True)[self.compartment_key].\
                value_counts(normalize=True)
            grouped_cell_freq_region = add_feature_metadata(
                grouped_cell_freq_region, metric='total_cell_freq', compartment_col=True)

            total_stats.extend([grouped_cell_counts_region, grouped_cell_freq_region])

        # add manually defined dfs to overall df
        cluster_dfs.extend(total_stats)
        total_df_clusters = pd.concat(cluster_dfs, axis=0)

        # compute density of cells for counts-based metrics and add to overall df
        density_df = self.calculate_density_stats(total_df_clusters, compartment_area_df)
        total_df_clusters = pd.concat([total_df_clusters, density_df], axis=0)

        # drop any zero rows for compartments not in the image
        if exclude_missing_compartments:
            total_df_clusters = self.exclude_empty_compartments(total_df_clusters)

        self.adata_table.uns['cluster_stats'] = total_df_clusters

    def generate_abundance_features(self, stats_df, density_params, ratio_cluster_key,
                                    minimum_density=0.0005):
        """ Create feature dataframes for cell abundance.
        Args:
            stats_df (pd.DataFrame): table created by generate_cluster_stats() containing density
                stats for each cell type
            density_params (list): list of which density features to generate
            ratio_cluster_key (str): cluster level to calculate ratios for
            minimum_density (float): minimum cell density required to generate the feature
        Returns:
            saves the abundance feature dataframes to the class
        """
        # add total density stats to list
        density_params.append(['total_cell_density', 'total_density', 'total'])

        # format density features
        for cluster_name, feature_name, cell_pop_level in density_params:
            input_df = stats_df[stats_df['metric'].isin([cluster_name])]
            for compartment in self.compartment_list:
                compartment_df = input_df[input_df.subset == compartment].copy()
                compartment_df['feature_name'] = compartment_df.cell_type + '__' + feature_name

                compartment_df_formatted = self.format_helper(
                    compartment_df, compartment, cell_pop_level, feature_type='density')
                self.feature_data_list.append(compartment_df_formatted)

        # compute ratio of broad cell type densities
        ratio_cluster_level, cell_pop_level = [f'{ratio_cluster_key}_density', ratio_cluster_key]
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

    def generate_stats(self, table, params, df_name, var_name, filter_stats, deduplicate_stats):
        """ Create dataframe containing stats per cell type and compartment.
        Args:
            table (pd.DataFrame): table appropriately filtered to contain the values needed to generate features
            params (list): list of which features to generate
            df_name (str): name to save for the dataframe
            var_name (str): name of the column containing the values
            filter_stats (bool): whether to filter features by minimum cell count
            deduplicate_stats (bool): whether to deduplicate highly correlated features
        Returns:
            generates and saves feature dataframe, as well as the filtered dataframe
        """
        subset_col = None if self.compartment_key_none else self.compartment_key

        stats_dfs = []
        for result_name, cluster_col_name, normalize in params:
            drop_cols = [self.seg_label_key]
            cluster_names = self.cluster_key.copy()
            cluster_names.remove(cluster_col_name)
            drop_cols.extend(cluster_names)

            stats_dfs.append(self.create_long_df(table=table,
                                                 result_name=result_name,
                                                 var_name=var_name,
                                                 cluster_col_name=cluster_col_name,
                                                 drop_cols=drop_cols,
                                                 normalize=normalize,
                                                 subset_col=subset_col))
        stats_df_comb = pd.concat(stats_dfs, axis=0)
        self.adata_table.uns[df_name] = stats_df_comb.reset_index(drop=True)

        if df_name == 'functional_stats':
            self.filter_functional_features(table, stats_df_comb, filter_stats, deduplicate_stats)
        else:
            if filter_stats:
                # filter stats by minimum cell count
                cell_filtered_df = self.filter_stats_by_cell_count(stats_df_comb)
                self.adata_table.uns[df_name + '_filtered'] = cell_filtered_df

    def remove_correlated_features(self, correlation_filtering_thresh, image_prop=0.1):
        """  A function to filter out features that are highly correlated in compartments.
        Args:
            correlation_filtering_thresh (float): the max correlation value the features have to be
                included the feature table, any features with correlation above it will be excluded
            image_prop (float): minimum proportion of images for compartment feature to include
        Returns:
            pd.DataFrame
                table with highly correlated features removed
        """
        # filter FOV features based on correlation in compartments
        feature_df = self.combined_feature_data

        # filter out features that are highly correlated in compartments
        feature_names = feature_df.feature_name.unique()
        exclude_list = []

        for feature_name in feature_names:
            fov_data_feature = feature_df.loc[feature_df.feature_name == feature_name, :]

            # get the compartments present for this feature
            compartments = fov_data_feature.compartment.unique()

            # if only one compartment, skip
            if len(compartments) == 1:
                continue

            fov_data_wide = fov_data_feature.pivot(
                index=self.image_key, columns=self.compartment_key, values='raw_value')

            # filter out features that are nans or mostly zeros
            for compartment in compartments:
                nan_count = fov_data_wide[compartment].isna().sum()
                zero_count = (fov_data_wide[compartment] == 0).sum()

                if (len(fov_data_wide) - nan_count - zero_count) / len(fov_data_wide) < image_prop:
                    exclude_list.append(feature_name + '__' + compartment)
                    fov_data_wide = fov_data_wide.drop(columns=compartment)

            # compute correlations
            compartments = fov_data_wide.columns
            compartments = compartments[compartments != 'all']
            for compartment in compartments:
                corr, _ = spearmanr(fov_data_wide['all'].values, fov_data_wide[compartment].values,
                                    nan_policy='omit')
                if corr > correlation_filtering_thresh:
                    exclude_list.append(feature_name + '__' + compartment)

        # remove features from dataframe
        exclude_df = pd.DataFrame({'feature_name_unique': exclude_list})
        self.excluded_features = exclude_df
        self.adata_table.uns['excluded_features'] = exclude_df
        feature_df_filtered = \
            feature_df.loc[~feature_df.feature_name_unique.isin(
                exclude_df.feature_name_unique.values), :]

        return feature_df_filtered.reset_index(drop=True)

    def combine_features(self, correlation_filtering_thresh=0.7):
        """ Combines the previously generated feature tables into a single dataframe.
        Args:
            correlation_filtering_thresh (float): threshold for correlation excluding, defaults to 0.7
                but can be set to None to avoid generating a filtered table
        Returns:
            generates and saves combined_feature_data, combined_feature_data_filtered,
            feature_metadata, and excluded_features tables to the class
        """
        # compute z-scores for each feature
        feature_df = pd.concat(self.feature_data_list).reset_index(drop=True)
        feature_df = feature_df.rename(columns={'value': 'raw_value'})
        feature_df_wide = feature_df.pivot(
            index=self.image_key, columns='feature_name_unique', values='raw_value')
        zscore_df = (feature_df_wide - feature_df_wide.mean()) / feature_df_wide.std()

        # add z-scores to feature_df
        zscore_df = zscore_df.reset_index()
        zscore_df_long = pd.melt(zscore_df, id_vars=self.image_key, var_name='feature_name_unique',
                                 value_name='normalized_value')
        feature_df = pd.merge(
            feature_df, zscore_df_long, on=[self.image_key, 'feature_name_unique'], how='left')

        # rearrange columns
        feature_df = feature_df[[self.image_key, 'raw_value', 'normalized_value', 'feature_name',
                                 'feature_name_unique', self.compartment_key, 'cell_pop_level',
                                 'feature_type']]

        # save full feature df
        self.combined_feature_data = feature_df
        self.adata_table.uns['combined_feature_data'] = feature_df

        if correlation_filtering_thresh:
            # save filtered feature df
            feature_df_filtered = self.remove_correlated_features(correlation_filtering_thresh)
            self.combined_feature_data_filtered = feature_df_filtered
            self.adata_table.uns['combined_feature_data_filtered'] = feature_df_filtered

            feature_df = feature_df_filtered

        # save feature metadata
        feature_metadata = feature_df[['feature_name', 'feature_name_unique', self.compartment_key,
                                       'cell_pop_level', 'feature_type']]
        feature_metadata = feature_metadata.drop_duplicates()
        self.feature_metadata = feature_metadata
        self.adata_table.uns['feature_metadata'] = feature_metadata

    def run_spacecat(self, functional_feature_level, filter_stats=True, deduplicate_stats=True,
                     correlation_filtering_thresh=0.7):
        """ Main function to calculate all cell stats and generate the final feature table.
        Args:
            functional_feature_level (str): clustering level to check all functional marker positivities against
            filter_stats (bool): whether to filter features by minimum cell count
            deduplicate_stats (bool): whether to deduplicate highly correlated features
            correlation_filtering_thresh (float): the correlation threshold for final feature filtering
        Returns:
             anndata:
                the anndata table with all intermediate and final tables appended
        """
        # validation checks
        verify_in_list(marker_positivity_level=functional_feature_level, all_cluster_levels=self.cluster_key)

        # Generate counts and proportions of cell clusters per FOV
        cluster_params = []
        for column in self.cluster_key:
            cluster_params.append([column + '_freq', column, True])
            cluster_params.append([column + '_count', column, False])

        # subset table for cluster data
        compartment_col = [] if self.compartment_key_none else [self.compartment_key]
        cell_table_clusters = self.adata_table.obs[
            [self.image_key, self.seg_label_key] + compartment_col + self.cluster_key]
        compartment_area_df = self.get_compartment_areas()

        self.generate_cluster_stats(cell_table_clusters, cluster_params, compartment_area_df)

        # set density feature parameters
        stats_df = self.adata_table.uns['cluster_stats']
        density_params = []
        for column in self.cluster_key:
            density_params.append([column + '_density', column + '_density', column])

        # determine broadest cluster column (least number of unique cell classifications)
        broadest_cluster_col = self.cluster_key[0]
        for col in self.cluster_key:
            if len(np.unique(cell_table_clusters[col])) < \
                    len(np.unique(cell_table_clusters[broadest_cluster_col])):
                broadest_cluster_col = col

        # generate abundance features
        self.generate_abundance_features(stats_df, density_params, ratio_cluster_key=broadest_cluster_col)

        # generate functional features
        # Create summary dataframe with proportions and counts of different functional marker populations
        cell_table_func = pd.concat(
            [cell_table_clusters, self.adata_table[:, [col for col in self.adata_table.var_names
                                                       if '+' in col]].to_df()], axis=1)
        self.generate_stats(cell_table_func, params=cluster_params, df_name='functional_stats',
                            var_name='functional_marker', filter_stats=filter_stats,
                            deduplicate_stats=deduplicate_stats)

        # format previous features
        df_name = 'functional_stats_filtered' if filter_stats or deduplicate_stats else 'functional_stats'
        feature_set = [[self.adata_table.uns[df_name], 'functional_marker']]
        # TO DO: add functional morphology features, etc.

        # only include functional marker features at the specified level, as well as total features
        metrics = [[f'{functional_feature_level}_freq', functional_feature_level], ['total_freq', 'all']]
        for stats_df, col_name in feature_set:
            self.format_computed_features(stats_df, col_name, metrics)

        # combine into full feature df
        self.combine_features(correlation_filtering_thresh)

        return self.adata_table

    ## FILTERING FUNCTIONS ##
    def filter_stats_by_cell_count(self, total_df, min_cell_count=5):
        """ Filters a feature table by minimum cell count.
        Args:
            total_df (pd.DataFrame): table with all features
            min_cell_count (int): minimum number of cells needed to keep the feature in the table
        Returns:
            pd.DataFrame:
                dataframe filtered by cell count
        """

        # remove features for cell populations less than min_cell_count
        filtered_dfs = []
        cluster_df = self.adata_table.uns['cluster_stats']

        for cluster in self.cluster_key:
            # subset count df to include cells at the relevant clustering resolution
            for compartment in self.compartment_list:
                count_df = cluster_df[cluster_df.metric == cluster + '_count']
                count_df = count_df[count_df.subset == compartment]

                # subset functional df to only include functional markers at this resolution
                subset_df = total_df[total_df.metric.isin([cluster + '_count', cluster + '_freq'])]
                subset_df = subset_df[subset_df.subset == compartment]

                # for each cell type, determine which FOVs have high enough counts to be included
                for cell_type in count_df.cell_type.unique():
                    keep_df = count_df[count_df.cell_type == cell_type]
                    keep_df = keep_df[keep_df.value >= min_cell_count]
                    keep_fovs = keep_df.fov.unique()

                    # subset functional df to only include FOVs with high enough counts
                    keep_stats = subset_df[subset_df.cell_type == cell_type]
                    keep_stats = keep_stats[keep_stats.fov.isin(keep_fovs)]

                    # append to list of filtered dfs
                    filtered_dfs.append(keep_stats)

        filtered_df = pd.concat(filtered_dfs)

        return filtered_df

    def filter_functional_features(self, table, stats_df, filter_stats, deduplicate_stats):
        """ Filter features with protocol specifically for functional marker interactions.
        Args:
            table (pd.DataFrame): table appropriately filtered to contain the values needed to generate features
            stats_df (pd.DataFrame): dataframe with all functional marker features
            filter_stats (bool): whether to filter features by minimum cell count
            deduplicate_stats (bool): whether to deduplicate highly correlated features
        Returns:
            generates and saves feature dataframe, as well as the filtered dataframe
        """

        total_df = stats_df

        if filter_stats:
            # filter stats by minimum cell count
            cell_filtered_df = self.filter_stats_by_cell_count(stats_df)

            # identify combinations of functional markers and cell types to include in analysis based on threshold
            # single positive markers
            sp_markers = [x for x in cell_filtered_df.functional_marker.unique() if '__' not in x]
            single_positive_df = self.subset_functional_markers(cell_filtered_df, sp_markers, prefix='sp',
                                                                mean_percent_positive=0.05)

            # double positive_markers
            dp_markers = [x for x in cell_filtered_df.functional_marker.unique() if '__' in x]
            double_positive_df = self.subset_functional_markers(cell_filtered_df, dp_markers, prefix='dp',
                                                                mean_percent_positive=0.05)

            filtered_df = pd.concat([single_positive_df, double_positive_df]).reset_index(drop=True)

            total_df = filtered_df
            self.adata_table.uns['functional_stats_filtered'] = total_df

        if deduplicate_stats:
            # remove highly related stats
            total_df = self.deduplicate_functional_stats(total_df).reset_index(drop=True)
            self.adata_table.uns['functional_stats_filtered'] = total_df

        # total freq stats
        marker_df = cell_filtered_df if filter_stats else stats_df
        dp_markers = [x for x in marker_df.functional_marker.unique() if '__' in x]
        sub_table = table.loc[:, ~table.columns.isin([self.seg_label_key, self.compartment_key]
                                                     + self.cluster_key.copy())]
        sub_table = sub_table.loc[:, ~sub_table.columns.isin(dp_markers)]

        # average values per image
        transformed = sub_table.groupby(self.image_key).agg('mean').reset_index()

        # reshape to long df
        long_df = pd.melt(transformed, id_vars=[self.image_key], var_name='functional_marker')
        long_df['metric'], long_df['cell_type'], long_df['subset'] = ['total_freq', 'all', 'all']

        # save to filtered df or full df
        total_df = pd.concat([total_df, long_df]).reset_index(drop=True)
        if filter_stats or deduplicate_stats:
            self.adata_table.uns['functional_stats_filtered'] = total_df
            self.adata_table.uns['functional_stats'] = pd.concat([self.adata_table.uns['functional_stats'], long_df])
        else:
            self.adata_table.uns['functional_stats'] = total_df

        return total_df

    def subset_functional_markers(self, filtered_df, marker_list, prefix, mean_percent_positive=0.05):
        """ Subsets functional marker stats for each cell type by thresholding average intensity.
        Args:
            filtered_df (pd.DataFrame): functional stats table already filtered by min cell count
            marker_list (list): list of all markers to consider
            prefix (str): prefix used when saving the exclusion matrix
                ('sp' for single positive markers, 'dp' for double positive)
            mean_percent_positive (float): mean intensity must be above this value for a marker to
                be included for the cell type
        Returns:
            pd.DataFrame:
                a dataframe with only the desired markers included for various cell types
        """

        combo_dfs = []
        for cluster in self.cluster_key:
            # subset functional stats table for cluster level
            broad_df = filtered_df[filtered_df.metric == f'{cluster}_freq']
            broad_df = broad_df[np.logical_and(broad_df.functional_marker.isin(marker_list), broad_df.subset == 'all')]
            broad_df_agg = broad_df[['functional_marker', 'cell_type', 'value']].groupby(
                ['cell_type', 'functional_marker']).agg('mean')
            broad_df_agg = broad_df_agg.reset_index()

            # determine whether avg marker intensity is above the threshold and save matrix
            broad_df = broad_df_agg.pivot(index='cell_type', columns='functional_marker', values='value')
            include_df = broad_df > mean_percent_positive
            self.adata_table.uns[f'{prefix}_marker_inclusion_{cluster}'] = include_df

            # subset functional df to only include functional markers at this resolution
            func_df = filtered_df[filtered_df.metric.isin([f'{cluster}_count', f'{cluster}_freq'])]

            # loop over each cell type, and get the corresponding markers
            for cell_type in include_df.index:
                markers = include_df.columns[include_df.loc[cell_type] == True]

                # subset functional df to only include this cell type and indicated markers
                func_df_cell = func_df[func_df.cell_type == cell_type]
                func_df_cell = func_df_cell[func_df_cell.functional_marker.isin(markers)]

                # append to list of dfs
                combo_dfs.append(func_df_cell)

        combo_df = pd.concat(combo_dfs)

        return combo_df

    def identify_correlated_marker_stats(self, total_df, correlation_thresh):
        """ Identify double positive marker features highly correlated with single positive features.
        Args:
            total_df (pd.DataFrame): functional stats table
            correlation_thresh (float): the max correlation value the features can have to be
                included the feature table, any features with correlation above it will be excluded
        Returns:
            pd.DataFrame:
                details which functional marker features to exclude
        """

        # get list of double positive markers
        dp_markers = [x for x in total_df.functional_marker.unique() if '__' in x]

        exclude_lists = []
        for cluster in self.cluster_key:
            # subset the df to look at just image-wide frequency at this resolution
            func_df_subset = total_df[total_df.metric == f'{cluster}_freq']
            func_df_subset = func_df_subset[func_df_subset.subset == 'all']

            # loop over each cell type, and each double positive functional marker
            exclude_markers = []
            cell_types = func_df_subset.cell_type.unique()
            for cell_type in cell_types:
                for marker in dp_markers:
                    # get the two markers that make up the double positive marker
                    marker_1, marker_2 = marker.split('__')
                    marker_1 = marker_1 + '+'

                    # subset to only include this cell type and these markers
                    current_df = func_df_subset.loc[func_df_subset.cell_type == cell_type, :]
                    current_df = current_df.loc[current_df.functional_marker.isin([marker, marker_1, marker_2]), :]

                    # these individual markers/double positive marker are not present in this cell type
                    if len(current_df) == 0 or marker not in current_df.functional_marker.unique():
                        continue

                    current_df_wide = current_df.pivot(index=self.image_key, columns='functional_marker',
                                                       values='value')

                    # the double positive marker is present, but both single positives are not; exclude it
                    if len(current_df_wide.columns) != 3:
                        exclude_markers.append(marker)
                        continue

                    corr_1, _ = spearmanr(current_df_wide[marker_1].values, current_df_wide[marker].values)
                    corr_2, _ = spearmanr(current_df_wide[marker_2].values, current_df_wide[marker].values)

                    # add to exclude list
                    if (corr_1 > correlation_thresh) | (corr_2 > correlation_thresh):
                        exclude_markers.append(cell_type + '__' + marker)

            cluster_exclude_df = pd.DataFrame(
                {'metric': [f'{cluster}_freq'] * len(exclude_markers), 'feature_name': exclude_markers})
            exclude_lists.append(cluster_exclude_df)

        # construct df to hold list of exlcuded cells
        exclude_df = pd.concat(exclude_lists).reset_index(drop=True)
        self.adata_table.uns['exclude_double_positive_markers'] = exclude_df

        return exclude_df

    def deduplicate_functional_stats(self, total_df, correlation_thresh=0.7):
        """ Remove highly correlated functional features.
        Args:
            total_df (pd.DataFrame): functional stats table
            correlation_thresh (float): the max correlation value the features can have to be
                included the feature table, any features with correlation above it will be excluded
        Returns:
            pd.DataFrame:
                a dataframe with highly correlated double positive marker features removed
        """

        # get df of highly correlated features to exclude
        exclude_df = self.identify_correlated_marker_stats(total_df, correlation_thresh)

        # remove any features in exclude_df
        dedup_dfs = []
        for cluster in self.cluster_key:
            # subset functional df to only include functional markers at this resolution
            func_df = total_df[total_df.metric.isin([f'{cluster}_count', f'{cluster}_freq'])]

            # add unique identifier for cell + marker combo
            func_df['feature_name'] = func_df['cell_type'] + '__' + func_df['functional_marker']

            # remove double positive markers that are highly correlated with single positive scores
            exclude_names = exclude_df.loc[exclude_df.metric == f'{cluster}_freq', 'feature_name'].values
            func_df = func_df[~func_df.feature_name.isin(exclude_names)]

            dedup_dfs.append(func_df)

        deduped_df = pd.concat(dedup_dfs)
        deduped_df = deduped_df.drop('feature_name', axis=1).reset_index(drop=True)

        return deduped_df
