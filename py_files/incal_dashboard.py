import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from incal_lib import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# get data
df = pd.read_csv('csvs/data.csv')
dict_groups = OrderedDict(Control=[1, 4, 7, 10, 13],
                          Group_2=[3, 5, 9, 12, 16],
                          Group_3=[2, 6, 8, 11, 14, 15])

# assemble incal dataframe shape and properties - multiindex
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

df = pd.DataFrame(df.values,
                  index=pd.MultiIndex.from_frame(multi_index_dataframe),
                  columns=df.columns.values.tolist())

# for layout
features = df.columns.values.tolist()
subjects_ids = df.index.get_level_values(1)
legand_color_order = np.sort([str(n) for n in subjects_ids.unique().values])
# to menage dashboard
storage_points_save = {}
is_removed_points = False
is_removed_outliers = False
is_trime_data = False
is_removed_group = False
is_removed_subjects = False

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
    ]),
])


def get_dff(df, v_feature, is_removed_points, **kwargs):
    dff = df.copy()
    levels_as_ids = dff.index
    levels_ids, levels_uniques = levels_as_ids.factorize()

    if is_removed_points:
        for row_index in storage_points_save[v_feature]:
            dff.at[row_index, v_feature] = np.nan
        print(dff)
    return dff[v_feature]


@dash.callback(Output('scatter_time_series', 'figure'),
               Input('feature_y_axis_dropdown', 'value'))
def create_scatter_graph(v_feature):
    dff = get_dff(df, v_feature, False)
    x_axis = dff.index.get_level_values(0)
    color = dff.index.get_level_values(1)
    y_axis = v_feature
    fig = px.scatter(dff, x=x_axis, y=y_axis, color=color)
    fig.update_traces(mode='lines+markers')
    fig.update_layout(legend_traceorder="normal")
    return fig


@dash.callback(
    Output('father', 'children'),
    Input('scatter_time_series', 'clickData'),
    State('feature_y_axis_dropdown',
          'value'),  # getting current feature name from dropdown
    State('father', 'children'),
    prevent_initial_call=True)
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

    dff = get_dff(df, feature, True, row_index=row_index)
    return children


if __name__ == '__main__':
    app.run_server(debug=True)
