import pandas as pd
import numpy as np
from incal_lib import *

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

levels_as_ids = df.index
levels_ids, levels_uniques = levels_as_ids.factorize()
df.at[levels_uniques[0], 'Energy_Balance'] = np.nan
point = '(1, "2021-07-30 05:00")'

print(10 in list(dict_groups.items())[0][1])