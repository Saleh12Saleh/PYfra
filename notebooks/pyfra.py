"""This module provides classes and functions available for all notebooks of the pyfra project."""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


def df_testing_info(df):
    """Returns a DataFrame that describes the given DataFrame"""
    df_dtypes_and_na_counts = pd.DataFrame({'dtypes':df.dtypes, 'n_na': df.isna().sum()})
    return pd.concat([df.describe().T, df_dtypes_and_na_counts])

def df_compare_to_description(df, description_filepath):
    '''Check whether or not the data is up-to-date 
    if DataFrame file can't be tracked on github because of it's file size)
    '''
    pd.testing.assert_frame_equal(left=(pd.read_csv(description_filepath, index_col=0)), \
                         right=df_testing_info(df),\
                         check_dtype=False, check_exact=False)

def store_metrics(model_name, model, y_test, y_pred, result_df=None):
    '''Returns a DataFrame containing the scores for the given model
    Appends the given result_df or creates a new one, if unprovided
    '''
    if result_df is None:
        result_df = pd.DataFrame(columns=['model', 'f1', 'accuracy', 'recall'])

    result_df.loc[model_name, 'model'] = model
    result_df.loc[model_name, 'f1'] = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    result_df.loc[model_name, 'accuracy'] = accuracy_score(y_true=y_test, y_pred=y_pred)
    result_df.loc[model_name, 'recall'] = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
    return result_df

def print_confusion_matrix(y_true, y_pred, model_name, filename=None, figsize=(4,4)):
    cm = pd.crosstab(y_true, y_pred, rownames=['observations'], colnames=['predictions']);
    severity_categories = ("Unscathed","Killed", "Hospitalized\nwounded", "Light injury")
    plt.figure(figsize=figsize)
    plt.title('Confusion Matrix of the '+ model_name);
    sns.heatmap(cm, cmap='RdYlGn', annot=True);
    plt.xticks(np.array(range(4))+0.5, labels=severity_categories, rotation=45);
    plt.yticks(np.array(range(4))+0.5, labels=severity_categories, rotation=0);
    if filename is not None:
        plt.savefig('../figures/'+filename+'.png', bbox_inches='tight');
        plt.savefig('../figures/'+filename+'.svg', bbox_inches='tight');