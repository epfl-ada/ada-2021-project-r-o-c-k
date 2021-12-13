'''
File name: utils.py
Author: Oskar 
Date created: 13/12/2021
Date last modified: 13/12/2021
Python Version: 3.8
'''
import pycountry_convert as pc
import numpy as np

def group_nation_by_continent(df):
    '''
    Replaces nation in column of nationalities 
    by continent
    :param df: df with a column named nationality, containing country names
    :return df_new: df with a column named nationality, containing continents
    '''
    df_new = df.copy()
    for idx, nation in enumerate(df_new["nationality"]):
        try:
            df_new["nationality"].iloc[idx] = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(nation, cn_name_format="default"))
        except:
            df_new["nationality"].iloc[idx] = np.NaN
    return df_new

def normalize_min_max(df):
    '''
    Normalises each column through
    min-max normalization
    :param df: df to normalize
    :return: normalized df
    '''
    return (df - df.min())/(df.max() - df.min())