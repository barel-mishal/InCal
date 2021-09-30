import pandas as pd
import plotly.express as px
import numpy as np
from pandas.api.types import CategoricalDtype
from IPython.display import display, HTML
from statsmodels.formula.api import ols
from collections import OrderedDict, Counter
from jupyter_dash import JupyterDash
from dash import html
from dash import dcc
import itertools
from dash import no_update
from dash import dash_table
import dash
from dash.dependencies import Input, Output, State


def incal_create_df_incal_format(df, dict_groups):
    categories_subjects = flat_list(list(dict_groups.values()))
    categories_groups = list(dict_groups.keys())

    date_time_level = pd.Series((pd.DatetimeIndex(df['Date_Time_1'])),
                                name='Date_Time_1')
    subjects_level = pd.Series(pd.Categorical(df['subjectID'],
                                              categories=categories_subjects,
                                              ordered=True),
                               name='subjectsID')
    group_level = pd.Series(pd.Categorical(df['Group'],
                                           categories=categories_groups,
                                           ordered=True),
                            name='Group')

    df = df.drop(columns=['Date_Time_1', 'subjectID', 'Group'])

    multi_index_dataframe = pd.concat(
        [date_time_level, subjects_level, group_level], axis=1)

    return pd.DataFrame(df.values,
                        index=pd.MultiIndex.from_frame(multi_index_dataframe),
                        columns=df.columns.values.tolist())


def get_data(__global_df__, categories_columns_names):
    # need to find solution to the use of global var
    time_series = __global_df__.reset_index(level=categories_columns_names)
    order_categoreis_columns(time_series,
                             subjectID=dict_groups.values(),
                             Group=dict_groups.keys())
    return time_series


def get_start_and_end_time(tuple_start_end, dict_time_stamps):
    start, end = (str(i) for i in tuple_start_end)
    return dict_time_stamps[start], dict_time_stamps[end]


def trim_df_datetime(df, start_time, end_time):
    return df.loc[start_time:end_time]


def remove_data(df, outliers_true, start_time, end_time):
    if not outliers_true:
        return trim_df_datetime(df, start_time, end_time)
    outliers_removed = remove_outliers_mixed_df(df, 'subjectID')
    return trim_df_datetime(outliers_removed, start_time, end_time).dropna()


#  removing subjects or group
def get_values_level(df, number_or_index_name):
    return df.index.get_level_values(number_or_index_name)


def get_difference_from_list_2(list1, list2):
    return list(set(list1) - set(list2))


def incal_remove_subjects(df, number_or_index_name, subjects_to_remove):
    subjects = get_values_level(df, number_or_index_name)
    strs_to_ints = lambda l: [int(x) for x in l]
    subjects_to_remove_ints = strs_to_ints(subjects_to_remove)
    selected_subjects = get_difference_from_list_2(subjects,
                                                   subjects_to_remove_ints)
    return df.loc[:, selected_subjects, :]


def incal_remove_groups(df, number_or_index_name, groups_to_remove):
    subjects = get_values_level(df, number_or_index_name)
    selected_group = get_difference_from_list_2(subjects, groups_to_remove)
    return df.loc[:, :, selected_group]


# removing outliears
def sort_data_by_ids(df, column_name):
    return df.sort_values(column_name)


def flat_list(d_list):
    '''
    dependencies: itertools
    '''
    return list(itertools.chain.from_iterable(d_list))


def slice_df_for_floats_and_category(df, column_name):
    return df.select_dtypes(include=['float64']), df.select_dtypes(
        include=['category'])


def get_subject_ids(df, column_name):
    return df[column_name].unique()


def calc_mean_and_std_for_df_by_ids(df, ids_values):
    return df.groupby(ids_values).agg([np.mean, np.std])


def get_lims_upper_and_lower(df_means_and_stds,
                             number_of_ids,
                             number_featuers_columns,
                             by_sd_of=2):
    calcs_shape_values = df_means_and_stds.values.reshape(
        number_of_ids, number_featuers_columns, 2)
    means = calcs_shape_values[:, :, :1]
    stds = calcs_shape_values[:, :, 1:]
    upper_lims = means + stds * by_sd_of
    lower_lims = means - stds * by_sd_of
    return upper_lims, lower_lims


def reshpe_vlaues_3d_ndarray(ndarray, axis0_dimensions, axis1_columns,
                             axis2_rows):
    return ndarray.reshape(axis0_dimensions, axis1_columns, axis2_rows)


def select_and_replace_outliers(ndarry_of_features, ndarry_uppers_lims,
                                ndarry_lowers_lims):
    conditiones = [
        ndarry_of_features > ndarry_uppers_lims,
        ndarry_of_features < ndarry_lowers_lims
    ]
    choices = [np.nan, np.nan]
    return np.select(conditiones, choices, ndarry_of_features)


def back_to_2d_ndarray(ndarry_of_features, axis1, axis2):
    return ndarry_of_features.reshape(axis1, axis2)


def sort_data_by_index(df):
    return df.sort_index()


def get_categories_cals_names(df):
    return df.index.names[1:]


def incal_get_categories_col_from_multiindex(df):
    levels_names = get_categories_cals_names(df)
    get_values_values_from_index = df.reset_index(level=levels_names)
    return get_values_values_from_index[levels_names]


def remove_outliers_mixed_df(df):
    # sourcery skip: inline-immediately-returned-variable
    sorted_df = df.sort_index(level=1)
    fetuers, ids = df.values, df.index
    df_means_and_stds = calc_mean_and_std_for_df_by_ids(
        df,
        ids.get_level_values(1).astype('int32'))
    number_of_ids = len(ids.levels[1].categories.astype('int32'))
    fetuers_columns = df.columns
    number_featuers_columns = len(fetuers_columns)
    upper_lims, lower_lims = get_lims_upper_and_lower(df_means_and_stds,
                                                      number_of_ids,
                                                      number_featuers_columns)
    dimensions_by_numbers_of_ids_upper_lims = reshpe_vlaues_3d_ndarray(
        upper_lims, number_of_ids, 1, number_featuers_columns)
    dimensions_by_numbers_of_ids_lower_lims = reshpe_vlaues_3d_ndarray(
        lower_lims, number_of_ids, 1, number_featuers_columns)
    columns_of_each_id = fetuers.shape[0] // number_of_ids
    dimensions_by_numbers_of_ids_values = reshpe_vlaues_3d_ndarray(
        fetuers, number_of_ids, columns_of_each_id, number_featuers_columns)
    outliers_replaced_to_nan_values_ndarray = select_and_replace_outliers(
        dimensions_by_numbers_of_ids_values,
        dimensions_by_numbers_of_ids_upper_lims,
        dimensions_by_numbers_of_ids_lower_lims)
    combien_axis0_and_axis1 = number_of_ids * columns_of_each_id
    original_df_shape = back_to_2d_ndarray(
        outliers_replaced_to_nan_values_ndarray, combien_axis0_and_axis1,
        number_featuers_columns)
    df_fetuers_without_outliers = pd.DataFrame(original_df_shape,
                                               columns=fetuers_columns,
                                               index=ids)
    df_without_outliers = pd.concat([df_fetuers_without_outliers], axis=1)
    return df_without_outliers


# 17.1 ms ± 175 µs per loop (mean ± std. dev. of 5 runs, 100 loops each)


def incal_set_multindex(df, list_of_multi_index, drop_current_index=False):
    ids_indexed_df = df.reset_index(drop=drop_current_index)
    return ids_indexed_df.set_index(list_of_multi_index)


def create_category_column(df, categories, ordered=True):
    '''
    order_categoreis_columns make sure the group and subjects in the right order. This is for,
    the statiscal analysis. The groups and the subjects needs to be in order of the expriment design.
    In order the anova, ancova and anova with interaction to work properly
    
    '''
    return pd.Categorical(df, categories=categories, ordered=True)


def replace_ids_to_group_id(ndarray_ids, groups_names, subjects_within_group):
    conditiones = [ndarray_ids == n for n in subjects_within_group]
    choices = groups_names
    return np.select(conditiones, choices, ndarray_ids)


def incal_create_group_column_from_ids(df, ids_column_name, dict_groups):
    n_ids_multiple_name = lambda name, n: [name] * len(n)
    subjects_vlaues = incal_format[ids_column_name].values
    items = dict_groups.items()
    groups_names = flat_list(
        [n_ids_multiple_name(group, ids) for group, ids in items])
    subjects_within_groups = flat_list([ids for ids in dict_groups.values()])
    return replace_ids_to_group_id(subjects_vlaues, groups_names,
                                   subjects_within_groups)


def incal_assemble_group_column_in_df(df, ids_column_name, dict_groups,
                                      group_column_name):
    values = incal_create_group_column_from_ids(df, ids_column_name,
                                                dict_groups)
    series = pd.Series(values, copy=False, name=group_column_name)
    return concat_dfs([incal_format, series])


def get_incal_levels_properties(dict_groups):
    date_time_type = 'datetime64[ns]'
    order_subjects = flat_list(dict_groups.values())
    order_groups = list(dict_groups.keys())
    return date_time_type, order_subjects, order_groups


def design_incal_levels(idx0, idx1, idx2, date_time_type, order_subjects,
                        order_groups):
    level_0 = idx0.astype(
        date_time_type)  #level 0 convert to type of date time
    level_1 = create_category_column(
        idx1, order_subjects)  #level 0 convert to type of date time
    level_2 = create_category_column(
        idx2, order_groups)  #level 0 convert to type of date time
    return level_0, level_1, level_2


def incal_create_levels(df, dict_groups):
    # https://stackoverflow.com/questions/34417970/pandas-convert-index-type-in-multiindex-dataframe
    date_time_type, order_subjects, order_groups = get_incal_levels_properties(
        dict_groups)
    idx = df.index
    l0, l1, l2 = design_incal_levels(idx.levels[0], idx.levels[1],
                                     idx.levels[2], date_time_type,
                                     order_subjects, order_groups)
    return df.index.set_levels([l0, l1, l2])


# group column and set multiindex format for analysis
def create_category_column(df, categories, ordered=True):
    '''
    order_categoreis_columns make sure the group and subjects in the right order. This is for,
    the statiscal analysis. The groups and the subjects needs to be in order of the expriment design.
    In order the anova, ancova and anova with interaction to work properly
    
    '''
    return pd.Categorical(df, categories=categories, ordered=True)


def replace_ids_to_group_id(ndarray_ids, groups_names, subjects_within_group):
    conditiones = [ndarray_ids == str(n) for n in subjects_within_group]
    choices = groups_names
    return np.select(conditiones, choices, ndarray_ids)


def incal_create_group_column_from_ids(df, ids_column_name, dict_groups):
    n_ids_multiple_name = lambda name, n: [name] * len(n)
    subjects_vlaues = df[ids_column_name].values
    items = dict_groups.items()
    groups_names = flat_list(
        [n_ids_multiple_name(group, ids) for group, ids in items])
    subjects_within_groups = flat_list([ids for ids in dict_groups.values()])
    return replace_ids_to_group_id(subjects_vlaues, groups_names,
                                   subjects_within_groups)


def incal_assemble_multi_index_format(df, ids_column_name, dict_groups,
                                      group_column_name):

    date_time = df.index.to_frame().reset_index(drop=True)

    subjects = df[ids_column_name].reset_index(drop=True)

    subjects_order = [str(n) for n in flat_list(dict_groups.values())]
    cat_subjects = create_category_column(subjects, subjects_order)

    groups_values = incal_create_group_column_from_ids(df, ids_column_name,
                                                       dict_groups)
    groups = pd.Series(groups_values, copy=False, name=group_column_name)
    cat_groups = create_category_column(groups, dict_groups.keys())
    df = df.drop(columns='subjectID')

    frame_datetime_subjects_groups = pd.concat([
        date_time,
        pd.Series(cat_subjects, copy=False, name=ids_column_name),
        pd.Series(cat_groups, copy=False, name=group_column_name)
    ],
                                               axis=1)
    multi_index = pd.MultiIndex.from_frame(frame_datetime_subjects_groups)
    return pd.DataFrame(df.values, columns=df.columns, index=multi_index)


# removing subjects or group
def get_values_level(df, number_or_index_name):
    return df.index.get_level_values(number_or_index_name)


def get_difference_from_list_2(list1, list2):
    return list(set(list1) - set(list2))


def incal_remove_subjects(df, number_or_index_name, subjects_to_remove):
    subjects = get_values_level(df, number_or_index_name)
    strs_to_ints = lambda l: [int(x) for x in l]
    subjects_to_remove_ints = strs_to_ints(subjects_to_remove)
    selected_subjects = get_difference_from_list_2(subjects,
                                                   subjects_to_remove_ints)
    return df.loc[:, selected_subjects, :]


def incal_remove_group(df, number_or_index_name, groups_to_remove):
    subjects = get_values_level(df, number_or_index_name)
    selected_group = get_difference_from_list_2(subjects, groups_to_remove)
    return df.loc[:, :, (selected_group)]


def select_columns_by_metebolic_parm(df, param_name, exclude=False):
    if exclude == True:
        mask = ~df.columns.str.contains(pat=param_name)
        return df.loc[:, mask]
    mask = df.columns.str.contains(pat=param_name)
    return df.loc[:, mask]


def selecting_multi_column_by_part_of_name(df, list_pattern_parm):
    return df.filter(regex='|'.join(list_pattern_parm))


def multi_columns_by_metabolic_param(df, list_met_param, number):
    # https://stackoverflow.com/questions/21285380/find-column-whose-name-contains-a-specific-string
    columns_for_calc = df.columns[df.columns.astype("string").str.contains(
        pat="|".join(list_met_param))]
    df_calc = df[columns_for_calc].apply(lambda x: x * number)
    drop_old_columns = df.drop(columns_for_calc, axis=1)
    return pd.concat([drop_old_columns, df_calc], axis=1)


def loop_func_and_dfs(dfs, func, *args):
    return [func(df, *args) for df in dfs]


def get_columns_names_list(df):
    return df.columns.values.tolist()


def _make_dict_to_replace_names(columns_names_list, pattern_addition_to_parms):
    leng = len(columns_names_list)
    return {
        columns_names_list[i]:
        pattern_addition_to_parms + columns_names_list[i]
        for i in range(leng)
    }


def _get_actuals_values(df):
    df_actuals_features_calculeted = df.diff()
    first_row_df_cumuletive = df.iloc[0:1]
    return df_actuals_features_calculeted.fillna(first_row_df_cumuletive)


def incal_get_actuals_from_cumuletive(df, columns_pattern,
                                      pattern_addition_to_parms):
    # get just the cumuletive columns from the original df
    df_cumuletive_culumns = select_columns_by_metebolic_parm(
        df, columns_pattern)
    # get the columns names of the cumuletive columns
    columns_names = _get_columns_names_list(df_cumuletive_culumns)
    # dict to replace names
    dict_new_names = _make_dict_to_replace_names(columns_names,
                                                 pattern_addition_to_parms)
    # replace the columns names of the actuals culumns
    df_actuals_features = df_cumuletive_culumns.rename(columns=dict_new_names)
    df_actuals = _get_actuals_values(df_actuals_features)
    return pd.concat([df, df_actuals], axis=1).drop(columns_names, axis=1)


def incal_calc_cumuletive_values(df, columns_pattern):
    select_cols = df.columns.astype("string").str.contains(pat=columns_pattern)
    actuals = df.loc[:, select_cols]
    actuals_columns_names = actuals.columns.values.tolist()
    new_cols_names = [
        name.replace(columns_pattern, '') for name in actuals_columns_names
    ]
    langth = len(actuals_columns_names)
    cumuletive = actuals.rename(columns={
        actuals_columns_names[i]: new_cols_names[i]
        for i in range(langth)
    }).cumsum()
    return pd.concat([df, cumuletive], axis=1)


def incal_set_multindex(df, list_of_multi_index):
    ids_indexed_df = df.reset_index()
    return ids_indexed_df.set_index(list_of_multi_index)


def incal_groupby_then_agg(df, list_to_groupby, agg_func):
    groupby = df.groupby(list_to_groupby)
    return groupby.agg(agg_func)


def incal_resample(df_unstacked_subjects, role_to_resmple_by, agg_func):
    # refactoring - > make it more genric function
    # https://stackoverflow.com/questions/15799162/resampling-within-a-pandas-multiindex
    return incal_groupby_then_agg(df_unstacked_subjects, [
        pd.Grouper(level='Date_Time_1', freq=role_to_resmple_by),
        pd.Grouper(level='subjectID')
    ], agg_func)


def _multi_index_df_unstack(df_multi_indexed):
    return df_multi_indexed.unstack()


def _return_original_stacked_df(df_unstacked_subjects):
    return df_unstacked_subjects.stack().reset_index(level=1)


def incal_cumsum(df, list_of_multi_index, list_columns_names_to_cumsum):
    multi_indexed_df = incal_set_multindex(df, list_of_multi_index)
    unstacked_df = _multi_index_df_unstack(multi_indexed_df)
    cumsum_columns = unstacked_df[list_columns_names_to_cumsum].cumsum()
    cumsum_columns.columns = cumsum_columns.columns.map(
        lambda s: (s[0] + '_cumsum', s[1]))
    concat_cumsum_columns = pd.concat([unstacked_df, cumsum_columns], axis=1)
    return _return_original_stacked_df(concat_cumsum_columns)


def _right_sepert_first_underscore(string):
    return tuple(string.rsplit("_", 1))


def _assemble_multi_index_axis_1_df(df, d_list, axis_1_names=["", ""]):
    # make a multi index
    mul_i_columns = pd.MultiIndex.from_tuples(d_list, names=axis_1_names)
    # assemble new dataframe with multi index columns
    return pd.DataFrame(df.values, index=df.index, columns=mul_i_columns)
    # then stack level 1 to the columns (level 1 -> subjects names e.g. 1 2 3...)


def incal_wide_to_long_df(wide_df, col_subj_name='subjectID'):
    cols_names = _get_columns_names_list(wide_df)
    # sepert feature name from cage number and put it in a tuple together ('allmeters', '1')
    l_micolumns = [_right_sepert_first_underscore(col) for col in cols_names]
    multi_index_axis_1_df = _assemble_multi_index_axis_1_df(
        wide_df, l_micolumns, ['', col_subj_name])
    # https://pandas.pydata.org/docs/user_guide/reshaping.html
    return multi_index_axis_1_df.stack(level=1)


def flatten(lst_in_lst):
    lst = []
    for l in lst_in_lst:
        if type(l) in [list, tuple, set]:
            lst.extend(l)
        else:
            return lst_in_lst
    return lst


def order_categoreis_columns(df, **kargs):
    '''
    order_categoreis_columns make sure the group and subjects in the right order. This is for,
    the statiscal analysis. The groups and the subjects needs to be in order of the expriment design.
    In order the anova, ancova and anova with interaction to work properly
    
    '''
    for col_name, order in kargs.items():
        df[col_name] = pd.Categorical(df[col_name],
                                      ordered=True,
                                      categories=flatten(order))


def day_and_night(df, datetime_column='Date_Time_1', start=7, end=19):
    df = df.assign(time=lambda x: np.where(
        df[datetime_column].dt.hour.ge(start)
        & df[datetime_column].dt.hour.lt(end), 'Day', 'Night')).dropna()
    return df


def incal_make_averages_table(df,
                              columns_names_too_groupby=['Group', 'subjectID'],
                              column_name_for_time_of_day='time'):
    full_day = df.groupby(by=columns_names_too_groupby, sort=True,
                          dropna=True).mean().reset_index().dropna()
    full_day[column_name_for_time_of_day] = 'Full day'
    D_and_N_df = day_and_night(df).groupby(
        by=[column_name_for_time_of_day, *columns_names_too_groupby],
        sort=True,
        dropna=True).mean().reset_index().dropna()
    return pd.concat([full_day, D_and_N_df])


# day and night time this data use for the graph below
def make_lists_start_and_end_to_day_night_time(df,
                                               datetime64_column='Date_Time_1',
                                               start=7,
                                               end=19):
    array_data_list = df[datetime64_column].unique()
    Series_datetime64 = pd.Series(array_data_list, name=datetime64_column)
    mask_daylight = Series_datetime64.dt.hour.ge(
        start) & Series_datetime64.dt.hour.lt(end)
    start_end = []
    still_True = False
    for i in range(len(Series_datetime64)):
        if still_True and mask_daylight.iloc[i]:
            start_end.append(Series_datetime64.iloc[i])
            still_True = False
        elif not still_True and not mask_daylight.iloc[i]:
            start_end.append(Series_datetime64.iloc[i])
            still_True = True
    return start_end


# stats
anova_features = [
    'rq', 'locomotor_activity', 'actual_pedmeters_cumsum',
    'actual_allmeters_cumsum'
]
ancova_and_anova_with_interaction_features = [
    'Energy_Balance',
    'kcal_hr',
    'vo2',
    'vco2',
    'actual_foodupa',
    'actual_waterupa',
]


def reanem_df_by_with_list_by_index(df, indexed_new_names):
    columns_names = df.columns.values.tolist()
    new_columns_names = indexed_new_names
    zip_lists = zip(columns_names, new_columns_names)
    dict_renamed_columns = {
        column_name: new_column_name
        for column_name, new_column_name in zip_lists
    }
    return df.rename(columns=dict_renamed_columns)


def concat_dfs(list_of_series_dfs):
    return pd.concat(list_of_series_dfs, axis=1)


def anova_with_interaction(df, metabolic_var, independent, categorical):
    return ols(
        f'{metabolic_var} ~ {independent} + C({categorical}) + {independent}:C({categorical})',
        data=df).fit().pvalues


def ancova(df, metabolic_var, independent, categorical):
    return ols(f'{metabolic_var} ~ {independent} + C({categorical})',
               data=df).fit().pvalues


def anova(df, metabolic_var, categorical):
    return ols(f'{metabolic_var} ~ C({categorical})', data=df).fit().pvalues


def make_pvalues_of_anova_analysis(df, m_vars, cat_var):
    return [anova(df, m_var, cat_var) for m_var in m_vars]


def make_pvalues_of_ancova_analysis(df, m_vars, independent, cat_var):
    return [ancova(df, m_var, independent, cat_var) for m_var in m_vars]


def make_pvalues_of_anova_with_interaction_analysis(df, m_vars, independent,
                                                    cat_var):
    return [
        anova_with_interaction(df, m_var, independent, cat_var)
        for m_var in m_vars
    ]


def match_case(case, df, list_of_features, independent, category_col_name):
    cases = {
        'anova':
        make_pvalues_of_anova_analysis(df, list_of_features,
                                       category_col_name),
        'ancova':
        make_pvalues_of_ancova_analysis(df, list_of_features, independent,
                                        category_col_name),
        'anova_with_interaction':
        make_pvalues_of_anova_with_interaction_analysis(
            df, list_of_features, independent, category_col_name),
    }
    return cases[case]


def incal_create_pvalues_datafram(case, df, list_of_features, independent,
                                  category_col_name):
    results_from_anovafunction = match_case(case, df, list_of_features,
                                            independent, category_col_name)
    pvalues_dfs_concated = concat_dfs(results_from_anovafunction)
    return reanem_df_by_with_list_by_index(pvalues_dfs_concated,
                                           list_of_features)


def create_anovas_table(df):
    anova_df = incal_create_pvalues_datafram('anova', df, anova_features,
                                             'bodymass', 'Group')
    anova_with_interaction_df = incal_create_pvalues_datafram(
        'anova_with_interaction', df,
        ancova_and_anova_with_interaction_features, 'bodymass', 'Group')
    # algoritem that get each non p value in anova with interaction and replace it with anova values and fill nan where is needed
    ancova_df = incal_create_pvalues_datafram(
        'ancova', df, ancova_and_anova_with_interaction_features, 'bodymass',
        'Group')
    return concat_dfs([anova_df, anova_with_interaction_df, ancova_df]).T


dict_aggrageted_function_for_column = {
    'Energy_Balance': 'mean',
    'actual_allmeters': 'mean',
    'actual_pedmeters': 'mean',
    'bodymass': 'mean',
    'kcal_hr': 'mean',
    'locomotor_activity': 'mean',
    'rq': 'mean',
    'vco2': 'mean',
    'vo2': 'mean',
    'vh2o': 'mean',
    'xbreak': 'mean',
    'ybreak': 'mean',
    'actual_foodupa': 'sum',
    'actual_waterupa': 'sum',
}
add_feature_for_agg = {
    **dict_aggrageted_function_for_column, 'actual_allmeters_cumsum': 'mean',
    'actual_pedmeters_cumsum': 'mean'
}