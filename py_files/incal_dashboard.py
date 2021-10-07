import dash
from dash.dependencies import Input, Output, State
import dash as dcc
from dash import html
import pandas as pd
import plotly.express as px
from incal_lib import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# import design study - expriment data, subject and groups
df = pd.read_csv('csvs/data.csv')
dict_groups = OrderedDict(Control=[1, 4, 7, 10, 13],
                          Group_2=[3, 5, 9, 12, 16],
                          Group_3=[2, 6, 8, 11, 14, 15])

# assemble incal dataframe shape and properties - multiindex
df = incal_create_df_incal_format(df, dict_groups)
df_removed_outliers = remove_outliers_mixed_df(df)

# for layout
features = df.columns.values.tolist()
subjects_ids = df.index.get_level_values(1)
legand_color_order = np.sort([str(n) for n in subjects_ids.unique().values])
# select group or subject (category name)
categories_columns_names = get_categories_cals_names(df)
obj_categories_columns_names = [{
    'label': feature,
    'value': feature
} for feature in categories_columns_names]

# trim data - range slider
time_stamps = df.index.get_level_values(0)
shape_analysis_format_indexed = df.shape
end_point_index_analysis_format_indexed = shape_analysis_format_indexed[0] - 1
marks_indexed_time_stamp = {
    i: time_stamps[i]
    for i in range(shape_analysis_format_indexed[0])
}
# data for Dropdown - removing group or subjects
subjects = df.index.get_level_values(1).unique()
groups = df.index.get_level_values(2).unique()
multi_selection_subjects = [{
    'label': str(subject),
    'value': str(subject)
} for subject in subjects]
multi_selection_groups = [{
    'label': str(group),
    'value': str(group)
} for group in groups]

# remove specific point
rows_ids, rows_ind = df.index.factorize()

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(id='feature_y_axis_dropdown',
                         options=[{
                             'label': i,
                             'value': i
                         } for i in features],
                         value=features[0]),
            dcc.Dropdown(id='show_as_group_or_individual',
                         options=obj_categories_columns_names,
                         value=categories_columns_names[0]),
            dcc.Dropdown(id='remove_specific_value',
                         options=[],
                         multi=True,
                         value=[]),
            dcc.Dropdown(id='remove_group',
                         options=multi_selection_groups,
                         multi=True),
            dcc.Dropdown(id='remove_subjects',
                         options=multi_selection_subjects,
                         multi=True),
            dcc.Checklist(id='checklist_outliears',
                          options=[
                              {
                                  'label': 'Remove outliers',
                                  'value': 'True'
                              },
                          ],
                          value=[],
                          labelStyle={'display': 'inline-block'}),
            dcc.RangeSlider(id="range_slider_trim_time_series",
                            marks=marks_indexed_time_stamp,
                            value=(0, end_point_index_analysis_format_indexed),
                            allowCross=False,
                            min=0,
                            max=end_point_index_analysis_format_indexed),
        ]),
    ]),
    html.Div([
        html.Button('Save data in each graph', id='save_data', n_clicks=0),
        dcc.Graph(id='scatter_time_series', clickData={}),
        dcc.Graph(id='averages'),
        dcc.Graph(id='box', clickData={}),
        dcc.Graph(id='hist'),
        dcc.Graph(id='regression', clickData={}),
        dash_table.DataTable(id='stats_table_Pvalue')
    ]),
    html.Div([])
])


def remove_data_point(data, row_index, feature):
    # remove where keys feature is place
    data.at[row_index, feature] = np.nan


def for_loop_removeing_data_point(data, rows, feature):
    for row in rows:
        i_row = int(row.split(' ')[0])
        remove_data_point(data, rows_ind[i_row], feature)


def get_dff(df, v_feature, is_removed_points=False, **kwargs):
    dff = df.copy()
    return dff[v_feature]


def click_data_points(df, click_data, feature):
    point_info = click_data['points'][0]
    # datetime
    Timestamp = pd.Timestamp
    x_datetime = Timestamp(point_info['x'])
    # subject number
    index_legand = point_info['curveNumber']  # witch cage
    subject_number = int(legand_color_order[index_legand])
    # witch group
    group = [
        item[0] for item in list(dict_groups.items())
        if subject_number in item[1]
    ][0]
    # index number
    # tuple like ids_ind
    date_time, subject, group = x_datetime, str(
        subject_number
    ), group  # example: (Timestamp('2021-07-28 16:00:00'), 6, 'Group_3')
    # make tuple
    row_ind = (date_time, int(subject), group
               )  # if error - not found... - check int or str given on subject
    # get a list for the .index func
    rows_ind = df.index.to_list()
    # use .index func to find the index of the row in the list of rows
    index = rows_ind.index(row_ind)
    # return tuple of index datetime subject and group
    return index, date_time, subject, group


def create_scatter(dff, colors):
    x_axis = dff.index.get_level_values(0)
    color = dff.index.get_level_values(1)
    y_axis = dff.values

    fig = px.scatter(x=x_axis,
                     y=y_axis,
                     color=color,
                     color_discrete_sequence=colors,
                     template='simple_white')
    fig.update_traces(mode='lines+markers')
    fig.update_layout(legend_traceorder="normal")
    return fig


def create_bar(averages, colors):
    groups = averages.index.get_level_values(1)
    y_axis = averages.values
    return px.bar(x=groups,
                  y=y_axis,
                  color=groups,
                  color_discrete_sequence=colors,
                  template='simple_white')


def create_histogram(time_series, category_name, colors):
    x_axis = time_series.values
    color_group = time_series.index.get_level_values(category_name).values
    return px.histogram(x=x_axis,
                        color=color_group,
                        color_discrete_sequence=colors,
                        template='simple_white')


def create_regression(averages_df, colors, feature_name):
    group_color = averages_df.index.get_level_values(1)
    x_axis = averages_df['bodymass'].values
    feature_name = feature_name if feature_name != 'bodymass' else 'rq'
    y_axis = averages_df[feature_name].values
    return px.scatter(x=x_axis,
                      y=y_axis,
                      color=group_color,
                      color_discrete_sequence=colors,
                      template='simple_white',
                      trendline='ols')


def create_box(time_series, category_name, colors):
    x_axis = time_series.index.get_level_values(category_name).values
    y_axis = time_series.values
    return px.box(x=x_axis,
                  y=y_axis,
                  color=x_axis,
                  color_discrete_sequence=colors,
                  template='simple_white')


def removing_group_or_subjects(data, remove_group, remove_subjects):
    if remove_group:
        return incal_remove_group(data, 2, remove_group)
    elif remove_subjects:
        return incal_remove_subjects(data, 1, remove_subjects)
    return data


def create_average_df(data, features_calc):
    # averages df
    subjects_ids = data.index.get_level_values('subjectsID')
    groups_ids = data.index.get_level_values('Group')
    return data.groupby([subjects_ids, groups_ids]).agg(features_calc).dropna(
    )  # dropna to get rid from the 0 and nan where groupby calc on subject that dosn't belong to group


def groupby_category(data, category, features_calc):
    if category == 'subjectsID':
        return data
    datetime = data.index.get_level_values('Date_Time_1')
    groups = data.index.get_level_values(category)
    grouped_data = data.groupby([datetime, groups])
    return grouped_data.agg(features_calc).dropna()


def statstical_analysis(averages_df):
    grouped_analysis_format_df = averages_df
    analysis_format_calculeted_index_reseted = \
        grouped_analysis_format_df.reset_index()
    p_values_table = create_anovas_table(
        analysis_format_calculeted_index_reseted)
    p_values_table = p_values_table.reset_index().rename(
        columns={'index': 'Features'})
    columns = [{'id': p, 'name': p} for p in p_values_table.columns.to_list()]
    table = p_values_table.to_dict('records')
    return columns, table, p_values_table


@dash.callback(Output('remove_specific_value', 'options'),
               Input('feature_y_axis_dropdown', 'value'))
def dropdown_rows_ids_feature(feature_name):
    datetime = 0
    subject = 1
    group = 2
    return [{
        'label':
        f'{rows_ids[i]} {str(rows_ind[i][datetime])} {rows_ind[i][subject]} {rows_ind[i][group]}',
        'value':
        f'{rows_ids[i]} {str(rows_ind[i][datetime])} {rows_ind[i][subject]} {rows_ind[i][group]}'
    } for i in range(len(rows_ind))]


@dash.callback(
    Output('scatter_time_series', 'figure'),
    Output('averages', 'figure'),
    Output('box', 'figure'),
    Output('hist', 'figure'),
    Output('regression', 'figure'),
    Output('stats_table_Pvalue', 'columns'),
    Output('stats_table_Pvalue', 'data'),
    Output('remove_specific_value', 'value'),
    Input('feature_y_axis_dropdown', 'value'),
    Input('remove_subjects', 'value'),
    Input('remove_group', 'value'),
    Input('checklist_outliears', 'value'),
    Input('range_slider_trim_time_series', 'value'),
    Input('range_slider_trim_time_series', 'marks'),
    Input('scatter_time_series', 'clickData'),
    Input('show_as_group_or_individual', 'value'),  # it is there to call the 
    State('show_as_group_or_individual', 'value'),
    State('feature_y_axis_dropdown',
          'value'),  # getting current feature name from dropdown
    State('remove_specific_value', 'value'),
    Input('save_data', 'n_clicks'))
def pool_dashboard_data(value_feature, remove_subjects, remove_group,
                        checklist_outliers, tuple_start_end, dict_time_stamps,
                        click_data, input_category, category_name,
                        state_feature, strs_remove_specific_values,
                        n_clicks_save_data):
    info = dash.callback_context
    is_clickData_triggered = info.triggered[0][
        'prop_id'] == 'scatter_time_series.clickData'
    is_feature_y_axis_triggered = info.triggered[0][
        'prop_id'] == 'feature_y_axis_dropdown.value'
    # remove outliers
    data = df.copy() if not checklist_outliers else df_removed_outliers.copy()
    # remove specific points
    if is_feature_y_axis_triggered:
        strs_remove_specific_values = []
    if is_clickData_triggered:  # removing data points that been click
        i, datetime, subject, group = click_data_points(
            data, click_data,
            state_feature)  # (Timestamp('2021-08-01 13:00:00'), 7, 'Control')
        strs_remove_specific_values.append(f'{i} {datetime} {subject} {group}')
    if strs_remove_specific_values:
        for_loop_removeing_data_point(data, strs_remove_specific_values,
                                      state_feature)  # inplace
        strs_remove_specific_values = [
            str(row) for row in strs_remove_specific_values
        ]
    dropdown_ids_rows = strs_remove_specific_values
    # trim datetime from the sides
    start_time, end_time = get_start_and_end_time(tuple_start_end,
                                                  dict_time_stamps)
    data = trim_df_datetime(data, start_time, end_time)
    # removing group or subjects depnding on the selection
    data = removing_group_or_subjects(data, remove_group, remove_subjects)
    # selecting and grouping data
    features_calc = add_feature_for_agg  # dict - key (column_name): value (calc for parmeter) - this dict is for aggregetion function for each feature
    # creating an averages df
    averages_df = create_average_df(data, features_calc)
    averages_df['Energy_Balance'] = averages_df[
        'actual_foodupa'].values - averages_df['kcal_hr'].values
    # timeseries - groupby subject or groups
    time_series_df = groupby_category(data, category_name, features_calc)
    # selecting column by "state_feature" (feature state is all the parmeters of the data i.e: Energy_Balance)
    # selecting for each "_df" (averages_df, time_series_df)
    time_series = get_dff(time_series_df, state_feature)
    averages = get_dff(averages_df, state_feature)
    colors = px.colors.qualitative.Vivid
    fig_scatter = create_scatter(time_series, colors)
    fig_bar = create_bar(averages, colors)
    fig_box = create_box(time_series, category_name, colors)
    fig_histogram = create_histogram(time_series, category_name, colors)
    fig_regression = create_regression(averages_df, colors, state_feature)
    # analysis section
    columns, table, p_values_table = statstical_analysis(averages_df)
    if n_clicks_save_data:
        averages.to_csv('averages table.csv')
        time_series.to_csv('time series table.csv')
        averages_df.to_csv('averages table.csv')
        time_series_df.to_csv('time series table.csv')
        p_values_table.to_csv('p values table.csv')
    return fig_scatter, fig_bar, fig_box, fig_histogram, fig_regression, columns, table, dropdown_ids_rows


if __name__ == '__main__':
    app.run_server(debug=True)