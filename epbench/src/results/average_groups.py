import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def extract_groups(df, nb_events, relative_to, metric = 'f1_score_lenient'):

    # NOTE: original (jgong)
    # df_sliced = df[(df['book_nb_events'] == nb_events) & (df['book_model_name'] == 'claude-3-5-sonnet-20240620')]
    # NOTE: change to use the book_model_name from the dataframe (jgong)
    # Filter by nb_events and get the book_model_name from the dataframe (not hardcoded)
    df_filtered = df[df['book_nb_events'] == nb_events]
    if len(df_filtered) == 0:
        raise ValueError(f'No results found for book_nb_events={nb_events}')
    
    # Get the book_model_name from the first row (all rows should have the same book_model_name for same nb_events)
    book_model_name = df_filtered.iloc[0]['book_model_name']
    df_sliced = df_filtered[df_filtered['book_model_name'] == book_model_name]
    
    if len(df_sliced) == 0:
        raise ValueError(f'No results found for book_nb_events={nb_events} and book_model_name={book_model_name}')

    #print(f"(using the book with {df_sliced.iloc[0]['book_nb_events']} events)")

    # template
    i = 0
    df_res_0 = df_sliced.iloc[i]['evaluation_object'].get_pretty_summary_relative_to(relative_to, metric)
    df_results = df_res_0.iloc[:, :-1] # take all but last column

    # fill
    for i in range(len(df_sliced)):
        df_res_i = df_sliced.iloc[i]['evaluation_object'].get_pretty_summary_relative_to(relative_to, metric)
        df_results[(df_sliced.iloc[i]['answering_kind'], 
                    df_sliced.iloc[i]['answering_model_name'],
                    df_sliced.iloc[i]['answering_embedding_chunk'])] = [x for x in df_res_i.iloc[:, -1]] # average # [float(x.split('±')[0]) for x in df_res_i.iloc[:, -1]] # average

    # remove the nan
    df_results_tmp = df_results.copy()
    for col in relative_to + ['count']:
        df_results_tmp = df_results_tmp.loc[:, df_results_tmp.columns != col]
    nan_rows = [[k for i, x in enumerate(df_results_tmp.iloc[k]) if np.isnan(float(x.split('±')[0]))==True ] for k in range(len(df_results))]
    issue_rows = list(set([item for sublist in nan_rows for item in sublist]))
    df_results = df_results.drop(issue_rows)

    return df_results
