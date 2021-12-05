'''
File name: feature_generation.py
Author: Oskar 
Date created: 04/12/2021
Date last modified: 04/12/2021
Python Version: 3.8
'''
import nltk
import mapply
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from grammar_mappings import pronoun_dict, verb_tag_dict, adj_tag_dict

def add_sentiment_cols(row):
    '''
    Adds the sentiment column to the row.
    This column is based on NLTK's polarity score
    -1 one means 100 % negative, 1 100 % positive
    :param row: dataframe row to add column to
    :return: row with specified column added
    '''
    sentiments = sid.polarity_scores(row['quotation'])
    row['sentiment'] = sentiments['pos'] - sentiments['neg']
    return row

def add_morphological_cols(row):
    '''
    Adds the following columns to each row:

    self: count of pronouns referring to the speaker (ex: me, I)
    union: count of pronouns referring to the speaker and its group (ex: we, our)
    other: count of pronouns referring to someone else than the speaker (ex: he, their)
    adj_ratio: ratio of regular adjectives in comparison to superlatives and comparatives
    {x}_count: count of x in the quote, for x in {adj, ., ,, !, ?, token, punctuation}

    -1 means only comparatives and superlatives, 1 only regular adjectives
    :param row: dataframe row to add columns to
    :return: row with specified columns added
    '''

    tokens = nltk.word_tokenize(row['quotation'])
    tokens_to_exclude = {'[',']'}
    tokens_filtered = [token for token in tokens if token not in tokens_to_exclude]
    word_tag_pairs = nltk.pos_tag(tokens_filtered)

    # prepare variables for iteration over tags

    sentence_count = 0
    token_count = 0

    sign_counts = {'.': 0, ',': 0, '!': 0, '?': 0, ':': 0, ';': 0}
    adj_counts = {'ordinal': 0, 'comparative': 0, 'superlative': 0}
    verb_counts = {'base': 0, 'pres': 0, 'past': 0}
    pronoun_counts = {'self': 0, 'union': 0, 'other': 0}
    sign_count = 0
    adj_count = 0
    verb_count = 0
    pronoun_count = 0
   
    last_punctuation_token = -1

    # Detect and count words and tags that correspond to features we look for
    for word, tag in word_tag_pairs:
        if word in sign_counts:
            sign_counts[word] += 1
            sign_count += 1
            if tag == '.':
                if last_punctuation_token + 1 != token_count: # makes cases where we have .., !!, and so on not count as two sentences
                    sentence_count += 1
                last_punctuation_token = token_count
        elif tag[:2] == 'VB':
            verb_count += 1
            verb_counts[verb_tag_dict[tag]] += 1
        elif tag[:2] == 'JJ':
            adj_count += 1
            adj_counts[adj_tag_dict[tag]] += 1
        elif tag == 'PRP' or tag == 'PRPS':
            word_lower_case = word.lower()
            if word_lower_case in pronoun_dict:
                pronoun_counts[pronoun_dict[word_lower_case]] += 1
                pronoun_count += 1
        token_count += 1

    # calculate and write values to columns
    
    if sentence_count == 0: # a quote without punctuation is interepreted as one sentence
        sentence_count = 1
    row['sentence_count'] = sentence_count

    for sign in sign_counts:
        count = sign_counts[sign]
        row[sign + '_per_sentence'] = count / sentence_count

    punctuation_count = sign_counts['.'] + sign_counts['!'] + sign_counts['?']
    
    row['sign_per_token'] = sign_count / token_count
    row['punctuation_per_sentence'] = punctuation_count / sentence_count
    
    word_count = token_count - sign_count # Will count some signs (not the ones specified) as a word, thus an approximation
    row['approx_word_count'] = word_count
    row['token_count'] = token_count

    # generate count and ratio features for specified word categories
    grammar_counts = [('adj', adj_count, adj_counts), ('verb', verb_count, verb_counts), ('pronoun', pronoun_count, pronoun_counts)]
    for (category, category_count, category_counts) in grammar_counts:
        row[category + '_per_word'] = category_count / word_count
        for sub_cat in category_counts:
            if category_count == 0:
                for verb in verb_counts:
                    row[sub_cat + '_ratio'] = 0
            else:
                for verb in verb_counts:
                    sub_count = category_counts[sub_cat]
                    row[sub_cat + '_ratio'] = 2 * ((sub_count / category_count) - (1 / 2))
            
    return row

def add_features_to_df(df):
    '''
    Adds the following features to the dataframe:
    
    self_count: count of pronouns referring to the speaker (ex: I, my)
    union_count: count of pronouns referring to the speaker and its group (ex: we, our)
    other_count: count of pronouns referring to someone else than the speaker (ex: he, her)
    {x}_count: count of x in the quote, for x in {sentence, token, approx_word (amount of words approximated by  #tokens - #signs)}
    {x}_ratio: ratio of subcategory of grammar category and grammar category in the range [-1,1] for x in {self, union, other}
    -1 means that there are only occurrences of words belonging to the other subcategories,
    0 that the occurrences of words in the specified subcategory make up half of the occurrences of the grammar category
    or that there are no occurences of words of the grammar category.
    1 that all occurences of the grammar category belong to the specified subcategory
    sentiment: based on NLTK's polarity score, -1 one means 100 % negative, 1 100 % positive
    
    :param row: dataframe row to add columns to
    :return: row with specified columns added
    :param df: dataframe to add features to
    :return new_df: new dataframe with added features
    '''
    new_df = df.copy()

    new_df = new_df.mapply(add_morphological_cols, axis = 1)
    new_df = new_df.mapply(add_sentiment_cols, axis = 1)

    return new_df

sid = SentimentIntensityAnalyzer()
mapply.init(n_workers=-1)
