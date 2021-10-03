import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
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

# to menage dashboard
storage_points_save = {}

# levels_as_ids = data.index
# levels_ids, levels_uniques = levels_as_ids.factorize()

app.layout = html.Div([
    html.Div([
        html.Div(id='father', children=[]),
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
        dcc.Graph(id='scatter_time_series', clickData={}),
        dcc.Graph(id='averages'),
        dcc.Graph(id='box', clickData={}),
        dcc.Graph(id='hist'),
        dcc.Graph(id='regression', clickData={}),
        dash_table.DataTable(id='stats_table_Pvalue')
    ]),
    html.Div([])
])


def remove_data_points(data):
    # remove where keys feature is place
    for feature in storage_points_save.keys():
        for row_index in storage_points_save[feature]:
            data.at[row_index, feature] = np.nan


def get_dff(df, v_feature, is_removed_points=False, **kwargs):
    dff = df.copy()
    return dff[v_feature]


def click_data_points(click_data, feature, children):
    print(click_data)
    point_info = click_data['points'][0]
    x_datetime = pd.Timestamp(point_info['x'])
    index_legand = point_info['curveNumber']
    subject_number = int(legand_color_order[index_legand])
    group = [
        item[0] for item in list(dict_groups.items())
        if subject_number in item[1]
    ][0]
    row_index = x_datetime, subject_number, group
    point = str(row_index)
    is_found_feature = feature in storage_points_save.keys()
    # get data ids and value after delete point
    # get data
    if is_found_feature:
        is_new_point = point not in storage_points_save.get(feature)
        if is_new_point:
            storage_points_save[feature].append(row_index)
            for item in children:
                if item['props']['id']['index'] == feature:
                    item['props']['options'].append({
                        'label': str(point),
                        'value': str(point)
                    })
    else:
        # the user add new feature that dosnt exist in the storage_points_save
        storage_points_save[feature] = [row_index]
        # when the user adds a feature it then creating a new pattern muching dropdown
        new_dropdown = dcc.Dropdown(id={
            'type': 'removabal_dropdown',
            'index': feature
        },
                                    options=[{
                                        'label': str(i),
                                        'value': str(i)
                                    } for i in storage_points_save[feature]],
                                    multi=True)
        children.append(new_dropdown)
    return children


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
    return columns, table


@dash.callback(
    Output('scatter_time_series', 'figure'),
    Output('averages', 'figure'),
    Output('box', 'figure'),
    Output('hist', 'figure'),
    Output('regression', 'figure'),
    Output('father', 'children'),
    Output('stats_table_Pvalue', 'columns'),
    Output('stats_table_Pvalue', 'data'),
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
    State('father', 'children'))
def pool_dashboard_data(value_feature, remove_subjects, remove_group,
                        checklist_outliers, tuple_start_end, dict_time_stamps,
                        click_data, input_category, category_name,
                        state_feature, children):

    info = dash.callback_context
    is_clickData_triggered = info.triggered[0][
        'prop_id'] == 'scatter_time_series.clickData'
    # remove outliers
    data = df.copy() if not checklist_outliers else df_removed_outliers.copy()
    # remove specific points

    if is_clickData_triggered or (state_feature in storage_points_save.keys()
                                  ):  # removing data points that been click
        children = click_data_points(click_data, state_feature, children)
        remove_data_points(data)  # inplace

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
    columns, table = statstical_analysis(averages_df)
    return fig_scatter, fig_bar, fig_box, fig_histogram, fig_regression, children, columns, table


if __name__ == '__main__':
    app.run_server(debug=True)
