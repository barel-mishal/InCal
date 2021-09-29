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
print(df_removed_outliers)

# for layout
features = df.columns.values.tolist()
subjects_ids = df.index.get_level_values(1)
legand_color_order = np.sort([str(n) for n in subjects_ids.unique().values])
# trim data - range slider
time_stamps = df.index.get_level_values(0)
shape_analysis_format_indexed = df.shape
end_point_index_analysis_format_indexed = shape_analysis_format_indexed[0] - 1
marks_indexed_time_stamp = {
    i: time_stamps[i]
    for i in range(shape_analysis_format_indexed[0])
}

# to menage dashboard
storage_points_save = {}

app.layout = html.Div([
    html.Div([
        html.Div(id='father', children=[]),
        html.Div([
            dcc.Dropdown(
                id='feature_y_axis_dropdown',
                options=[{
                    'label': i,
                    'value': i
                } for i in features],
                value=features[0],
            ),
            dcc.Checklist(id='checklist_outliears',
                          options=[
                              {
                                  'label': 'Remove outliers',
                                  'value': 'True'
                              },
                          ],
                          value=[],
                          labelStyle={'display': 'inline-block'}),
            dcc.RangeSlider(
                marks=marks_indexed_time_stamp,
                value=(0, end_point_index_analysis_format_indexed),
                id="range_slider_trim_time_series",
                allowCross=False,
                min=0,
                max=end_point_index_analysis_format_indexed,
            ),
            dcc.Dropdown(id='',
                         options=[{
                             'label': i,
                             'value': i
                         } for i in features],
                         value=features[1]),
        ]),
    ]),
    html.Div([
        dcc.Graph(id='scatter_time_series', clickData={}),
        dcc.Graph(id='averages'),
        dcc.Graph(id='box', clickData={}),
        dcc.Graph(id='hist'),
        dcc.Graph(id='regression', clickData={}),
    ]),
    html.Div([])
])


def get_dff(df, v_feature, is_removed_points=False, **kwargs):
    dff = df.copy()
    levels_as_ids = dff.index
    levels_ids, levels_uniques = levels_as_ids.factorize()
    if is_removed_points:
        for row_index in storage_points_save[v_feature]:
            dff.at[row_index, v_feature] = np.nan
    return dff[v_feature]


def click_data_points(click_data, feature, children):
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


def create_scatter(dff):
    x_axis = dff.index.get_level_values(0)
    color = dff.index.get_level_values(1)
    y_axis = dff.name
    fig = px.scatter(dff, x=x_axis, y=y_axis, color=color)
    fig.update_traces(mode='lines+markers')
    fig.update_layout(legend_traceorder="normal")
    return fig


def create_averages(dff):
    return


@dash.callback(
    Output('scatter_time_series', 'figure'),
    Output('father', 'children'),
    Input('feature_y_axis_dropdown', 'value'),
    Input('checklist_outliears', 'value'),
    Input('range_slider_trim_time_series', 'value'),
    Input('range_slider_trim_time_series', 'marks'),
    Input('scatter_time_series', 'clickData'),
    State('feature_y_axis_dropdown',
          'value'),  # getting current feature name from dropdown
    State('father', 'children'))
def create_scatter_graph(value_feature, checklist_outliers, tuple_start_end,
                         dict_time_stamps, click_data, state_feature,
                         children):
    info = dash.callback_context
    is_clickData_triggered = info.triggered[0][
        'prop_id'] == 'scatter_time_series.clickData'
    # remove outliers
    data = df.copy() if not checklist_outliers else df_removed_outliers.copy()
    # remove specific points
    if is_clickData_triggered or (state_feature in storage_points_save.keys()
                                  ):  # removing data points that been click
        children = click_data_points(click_data, state_feature, children)
        # need to make a function when saving data we get all the dot that the user removed
        # sepert deleting and get_dff (dff for dom) to sepert function and return all data that was deleted
        data = get_dff(data, state_feature, True)
    # trim datetime from the sides
    start_time, end_time = get_start_and_end_time(tuple_start_end,
                                                  dict_time_stamps)
    dff = trim_df_datetime(data, start_time, end_time)
    dff = get_dff(dff, state_feature)
    # remove groups
    # remove subjects

    return create_scatter(dff), children


if __name__ == '__main__':
    app.run_server(debug=True)
