'''
File name: feature_selection.py
Author: Celinna 
Date created: 12/12/2021
Date last modified: 12/12/2021
Python Version: 3.8
'''
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets, cluster
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def merge(df_quotes, df_speaker):
    '''
    Merges quote features and speaker features
    
    param df_quotes: quotes feature df
    param df_speaker: speaker feature df
    return df: merged dataframe 
    '''
    df_quotes = df_quotes.set_index('qid')
    
    # drop duplicate rows (mostly data headers)
    df_quotes.drop_duplicates(inplace=True)
    df_quotes = df_quotes[df_quotes.quoteID.str.contains('quoteID') == False]
    
    # change value type
    for field in df_quotes.columns:
        df_quotes[field] = df_quotes[field].astype(dtypes[field], errors = 'raise')
    
    # merge df
    df = df_quotes.merge(df_speaker.set_index('id'), left_index=True, right_index=True)
    df.drop_duplicates(inplace=True)
    
    return df


def tree_feature_select(X, y):
    '''
    Does feature selection using decision trees
    
    :param X: features (n_rows, n_features)
    :param y: target (n_rows,)
    :return clf: extra trees classifiers
    :return model: model with reduced features
    '''
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    
    return clf, model


def select_predictors(df, ft_language, target):
    '''
    Extracts portion of the df and extracts the most relevant features based on a target speaker attribute
    
    :param df: merged dataframe with language and speaker features
    :param ft_language: list of language features
    :param target: target speaker attribute to predict
    :return clf: extra trees classifier
    :return model: selected model
    :return shape: shape of the data used (some target speaker features have nan which needs to be removed)
    :return X: features used to get model
    :return y: targets used to get model
    '''
    temp = df[ft_language].copy()
    temp[target] = df[target]
    temp.dropna(inplace=True)
    
    X = temp[ft_language]
    y = temp[target]
    clf, model = tree_feature_select(X, y)
    clf.feature_importances_
    
    return clf, model, temp.shape, X, y



def get_balanced_sample(target, target_dict, sample_size=50):
    # file location
    years = ['2015', '2016', '2017', '2018', '2019', '2020']
    generate_path = './../../data/merged_data/'
    file_prefix_r = 'merged_data_'
    file_suffix_r = '.csv.gzip'

    df_new = pd.DataFrame() # initialize values
    
    # load and parse data in chunks
    chunksize = 50000
    for year in years:
        print("Year: " + str(year))
        # read the file from year and filer away features that are not selected
        file_name_r = file_prefix_r + year + file_suffix_r
        chunks = pd.read_csv(generate_path + file_name_r, chunksize = chunksize, compression='gzip', low_memory=False)
        for i, chunk in enumerate(chunks):
            size = sample_size
            # check for smallest sample
            for key in target_dict.keys():
                sample = chunk.loc[chunk[target] == key]
                if sample.shape[0] < sample_size:
                    size = sample.shape[0]

            # add sample to df
            if size > 0:
                for key in target_dict.keys():
                    sample = chunk.loc[chunk[target] == key].sample(size, replace=False, random_state=1)
                    df_new = pd.concat([df_new, sample])

            if i % 50 == 0:
                print("Chunk: " + str(i), df_new.shape)
    
    return df_new


gender_dict = {
    'Male': 0,
    'Female': 1
}

religion_dict = {
    'Christian': 0,
    'Hindus': 1, 
    'Muslim': 2, 
    'Jewish': 3
}

degree_dict = {
    'Bachelor': 0, 
    'Master': 1,
    'Doctorate': 2
}

occupation_dict = {
    'Politics': 0, 
    'Arts': 1, 
    'Military': 2, 
    'Sciences': 3, 
    'Business': 4,
    'Sports': 5, 
    'Religion': 6,
    'Other': 7
}

continent_dict = {
    'NA': 0, 
    'AS': 1, 
    'EU': 2, 
    'OC': 3, 
    'SA':4, 
    'AF':5
}

generation_dict = {
    '70s': 0, 
    '50s': 1, 
    '30s': 2, 
    '90s': 3, 
    '00s': 4
}