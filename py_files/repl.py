import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from incal_lib import *

df = pd.read_csv('csvs/data.csv')
dict_groups = OrderedDict(Control=[1, 4, 7, 10, 13],
                          Group_2=[3, 5, 9, 12, 16],
                          Group_3=[2, 6, 8, 11, 14, 15])

# assemble incal dataframe shape and properties - multiindex
df = incal_create_df_incal_format(df, dict_groups)
df_removed_outliers = remove_outliers_mixed_df(df)

features_calc = add_feature_for_agg
# dict_aggrageted_function_for_column
datetime = df.index.get_level_values('Date_Time_1')
groups = df.index.get_level_values('Group')
grouped_df = df.groupby([datetime, groups])
print(grouped_df.mean())
print(grouped_df.agg(dict_aggrageted_function_for_column))

print(len(df.columns))
print(len(add_feature_for_agg.keys()))