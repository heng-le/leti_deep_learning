import seaborn as sns
import argparse
import pandas as pd
from rf import rf_training_vis
from bkwd_elim import backward_elimination, opposite_backward_elimination
from feature_matrix_generator import generate_feature_mx
from xgb_vis import xg_training_vis
from feature_pair import find_correlation 
from remove_paired import update_dataframe
import numpy as np

import matplotlib.pyplot as plt



def main():
    # Load datasets
    fh = pd.read_csv("../FH_lenti_editing_summary_processed.csv")
    cv = pd.read_csv("../p12_editing_summary.csv")
    amp = pd.read_csv("~/internship/feature_extraction/p53_Amplicon-Seq_data.csv")
    
    # Process datasets
    fh['target_no_pam'] = fh['target'].apply(lambda x: x[:-10])
    fh['pam'] = fh['target'].apply(lambda x: x[-10:])
    fh = fh[fh['spacer_length'] == 23]
    fh_mtx = generate_feature_mx(fh)
    
    cv['target_no_pam'] = cv['target'].apply(lambda x: x[:-10])
    cv['pam'] = cv['target'].apply(lambda x: x[-10:])
    cv = cv[cv['spacer_length'] == 23]
    cv_mtx = generate_feature_mx(cv)

    amp['target_no_pam'] = amp['target'].apply(lambda x: x[:-10])
    amp['pam'] = amp['target'].apply(lambda x: x[-10:])
    amp = amp[amp['spacer_length'] == 23]
    amp_mtx = generate_feature_mx(amp)
    
    # Feature selection using backward elimination
    fh_sorted_features, fh_importance_df = backward_elimination(fh_mtx)
    cv_sorted_features, cv_importance_df = backward_elimination(cv_mtx)
    amp_sorted_features, amp_importance_df = backward_elimination(amp_mtx)
    
    # Find common features between the two datasets
    fh_features_set = set(fh_sorted_features)
    cv_features_set = set(cv_sorted_features)
    amp_features_set = set(amp_sorted_features)
    similar_features = fh_features_set.intersection(cv_features_set, amp_features_set)
    
    # Retrieve importance dataframes for similar features
    fh_similar_importances = fh_importance_df[fh_importance_df['feature'].isin(similar_features)]
    cv_similar_importances = cv_importance_df[cv_importance_df['feature'].isin(similar_features)]
    amp_similar_importances = amp_importance_df[amp_importance_df['feature'].isin(similar_features)]

    print(f"amp similar importances columns: {amp_similar_importances.columns}")
    merged_importances = fh_similar_importances.merge(cv_similar_importances,
                                                      on='feature', 
                                                      suffixes=('_fh','_cv')).merge(amp_similar_importances, 
                                                                                   on='feature', suffixes=('', '_amp'))
    print("merged importances columns")
    print(merged_importances.columns)
    merged_importances['total_importance'] = (
        merged_importances['importance_mean_fh'] + 
        merged_importances['importance_mean_cv'] + 
        merged_importances['importance_mean'] 
    )
    
    top_features = merged_importances.sort_values(by='total_importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    width = 0.25 
    r1 = range(len(top_features))
    r2 = [x + width for x in r1]
    r3 = [x + width*2 for x in r1]
    

    bars1 = ax.bar(r1, np.abs(top_features['importance_mean_fh']), width, label='FH Importance')
    bars2 = ax.bar(r2, np.abs(top_features['importance_mean_cv']), width, label='CV Importance')
    bars3 = ax.bar(r3, np.abs(top_features['importance_mean']), width, label='AMP Importance')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=top_features, x='feature', y='total_importance')
    
    # Setting labels and title with larger font sizes
    ax.set_xlabel('Features', fontsize=14)
    ax.set_ylabel('Importance', fontsize=14)
    ax.set_title('Top 10 Similar Features by Importance from FH, CV, and AMP Datasets', fontsize=16)
    
    # Setting x-ticks and their labels with larger font size
    ax.set_xticklabels(top_features['feature'], rotation=45, ha='right', fontsize=12)
    
    # Increasing the font size of y-ticks
    ax.tick_params(axis='y', labelsize=12)
    
    # Removing unnecessary spaces and adjusting layout
    plt.tight_layout()
    
    # Saving the plot
    plt.savefig('feature_importance_plot.png', format='png', dpi=300)

    # Second part: comparing the top paired features by p-value

    # CV dataset
    
    # cv_p_value_df = find_correlation(cv_mtx, cv_sorted_features)
    # cv_p_value_df = cv_p_value_df.sort_values(by=['Combined P-value', 'column'], ascending=True)

    # top_cv_paired = pd.unique(cv_p_value_df['column'].tolist())[:25]
    # print("These are the top paired features for CV dataset: ")
    # print(top_cv_paired)

    # # FH
    # fh_p_value_df = find_correlation(fh_mtx, fh_sorted_features)
    # fh_p_value_df = fh_p_value_df.sort_values(by=['Combined P-value', 'column'], ascending=True)


    # top_fh_paired = pd.unique(fh_p_value_df['column'].tolist())[:25]
    # print("These are the top paired features for FH dataset: ")
    # print(top_fh_paired)

    # # AMP
    # amp_p_value_df = find_correlation(amp_mtx, amp_sorted_features)
    # amp_p_value_df = amp_p_value_df.sort_values(by=['Combined P-value', 'column'], ascending=True)

    # top_amp_paired = pd.unique(amp_p_value_df['column'].tolist())[:25]
    # print("These are the top paired features for AMP dataset: ")
    # print(top_amp_paired)

        
    # top_cv_set = set(top_cv_paired)
    # top_fh_set = set(top_fh_paired)
    # top_amp_set = set(top_amp_paired)

    # # Find common elements 
    # common_features = list(top_cv_set.intersection(top_fh_set, top_amp_set))

    # print("Here are the common features: ")
    # print(common_features)
    

    



if __name__ == "__main__":
    main()
