import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast

credits = pd.read_csv('/Users/phuongdoan/Downloads/dataset/credits.csv')
keywords = pd.read_csv('/Users/phuongdoan/Downloads/dataset/keywords.csv')
movies = pd.read_csv('/Users/phuongdoan/Downloads/dataset/movies_metadata.csv',low_memory = False)
ratings = pd.read_csv('/Users/phuongdoan/Downloads/dataset/ratings.csv')


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

#print(credits[['cast_names','crew_names', 'id']].head())

def parse_json_column(keywords, column_name):
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
    
    return keywords[column_name].apply(parse_cell)
keywords['keywords'] = parse_json_column(keywords,'keywords')
#print(keywords[['id','keywords']].head())

def format_currency(value):
    """Format budget value and revenue value to currency"""
    try:
        num_value = float(value)
        if num_value > 0:
            return f"${num_value:,.0f}"
        return "N/A"
    except (ValueError, TypeError):
        return "N/A"
#print(movies[['budget','revenue']].head())