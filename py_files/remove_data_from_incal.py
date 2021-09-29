import pandas as pd
import numpy as np
from incal_lib import *

# import design study - expriment data, subject and groups
df = pd.read_csv('csvs/data.csv')
dict_groups = OrderedDict(Control=[1, 4, 7, 10, 13],
                          Group_2=[3, 5, 9, 12, 16],
                          Group_3=[2, 6, 8, 11, 14, 15])

df = incal_create_df_incal_format(df, dict_groups)

levels_as_ids = df.index
levels_ids, levels_uniques = levels_as_ids.factorize()
df.at[levels_uniques[0], 'Energy_Balance'] = np.nan
point = '(1, "2021-07-30 05:00")'

print(10 in list(dict_groups.items())[0][1])