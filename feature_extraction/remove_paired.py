import pandas as pd
import numpy as np 

def encoder(df):
    prefix_map = {}
    cols = []

    for col in df.columns:
        if col.startswith('paired'):
            prefix_map[col] = col
            cols.append(col)
    
    encoded_df = pd.get_dummies(df, columns=cols, prefix=[prefix_map[col] for col in cols])
    
    return encoded_df



def update_dataframe(df, feature_list):
    """
    Takes in a processed dataframe and a list of features. Will create columns from specified paired features and then remove the original columns from the dataframe. Ignores the last feature if an odd number of features are provided.

    Parameters:
    df (pandas.DataFrame): The dataframe to be updated.
    feature_list (list): List of column names in the dataframe to be used for creating paired features.

    Returns:
    pandas.DataFrame: The updated dataframe with new paired features and without the original features used for pairing.
    """
    bin_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    new_df = df.copy()

    unwanted_list = ['SGN_Strand', 'Editing_Position', 'Acontent', 'Tcontent', 'Ccontent', 'Gcontent', 'GCcontent', 'Editing_mt']
    begin = 'SGN_mt_w_'
    
    feature_list = [feature for feature in feature_list if feature not in unwanted_list and not feature.startswith(begin)]
    
    drop_columns = []
    new_columns_data = {}
    
    for i in range(0, len(feature_list) - len(feature_list) % 2, 2):
        feature1 = feature_list[i]
        feature2 = feature_list[i + 1]
        new_column_name = f"paired_{feature1}_{feature2}"
        
        new_columns_data[new_column_name] = [bin_dict[(row[0], row[1])] for row in zip(df[feature1], df[feature2])]
        
        drop_columns.extend([feature1, feature2])

    # Create a new DataFrame for the new columns and concatenate it with the original DataFrame
    new_columns_df = pd.DataFrame(new_columns_data, index=df.index)
    new_df = pd.concat([new_df, new_columns_df], axis=1)
    
    # Drop the old columns all at once
    new_df.drop(columns=drop_columns, inplace=True)

    encoded_df = encoder(new_df)
    return encoded_df






