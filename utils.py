import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast

credits = pd.read_csv('/Users/phuongdoan/Downloads/dataset/credits.csv')
#print(df_credits['cast'].head(10))
#print(type(df_credits['cast'][0]))

def parse_json_column(credits, column_name):
    """
    Parse a stringified JSON column into a list of 'name' values.
    """
    def parse_cell(x):
        if pd.notnull(x):
            try:
                data = ast.literal_eval(x)  # Đổi từ json.loads sang ast.literal_eval
                return [item['name'] for item in data]
            except (ValueError, TypeError, KeyError):
                return []
        else:
            return []
    
    return credits[column_name].apply(parse_cell)

credits['cast_names'] = parse_json_column(credits, 'cast')
credits['crew_names'] = parse_json_column(credits, 'crew')

print(credits[['cast_names','crew_names']].head())



