import argparse
import pandas as pd
from rf import rf_training_vis
from bkwd_elim import backward_elimination, opposite_backward_elimination
from feature_matrix_generator import generate_feature_mx
from xgb_vis import xg_training_vis
from feature_pair import find_correlation 
from remove_paired import update_dataframe
import matplotlib.pyplot as plt


def split_string(input_string, possible_substrings):
    found_substrings = []
    
    possible_substrings.sort(key=len, reverse=True)
    
    for substring in possible_substrings:
        pos = input_string.find(substring)
        if pos != -1:
            found_substrings.append((pos, substring))
            input_string = input_string.replace(substring, ' ' * len(substring), 1)

    found_substrings.sort()
    sorted_substrings = [substring for pos, substring in found_substrings]

    return sorted_substrings
    


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

    unwanted_list = ['SGN_Strand', 'Editing_Position', 'Acontent', 'Tcontent', 'Ccontent', 'Gcontent', 'GCcontent', 'Editing_mt']
    begin = 'SGN_mt_w_'

    # Feature selection using backward elimination
    fh_sorted_features, fh_importance_df = backward_elimination(fh_mtx)
    cv_sorted_features, cv_importance_df = backward_elimination(cv_mtx)
    amp_sorted_features, amp_importance_df = backward_elimination(amp_mtx)

    fh_sorted_features = [feature for feature in fh_sorted_features if feature not in unwanted_list and not feature.startswith(begin)]

    cv_sorted_features = [feature for feature in cv_sorted_features if feature not in unwanted_list and not feature.startswith(begin)]
    amp_sorted_features = [feature for feature in amp_sorted_features if feature not in unwanted_list and not feature.startswith(begin)]

    # Second part: comparing the top paired features by p-value
    print('comparing features by p-value')
    # CV dataset
    
    cv_p_value_df = find_correlation(cv_mtx, cv_sorted_features)
    cv_p_value_df = cv_p_value_df.sort_values(by=['Combined P-value', 'column'], ascending=True)
    cv_p_value_df['column'] = cv_p_value_df['column'].apply(lambda x: "_".join(x.split("_")[1:]))
    cv_p_value_df['split'] = cv_p_value_df['column'].apply(lambda x: split_string(x, cv_sorted_features))
    cv_p_value_df = cv_p_value_df[~cv_p_value_df['split'].apply(lambda x: 'SGN_P2_T' in x)]
    top_cv_paired = pd.unique(cv_p_value_df['column'].tolist())


    # FH
    fh_p_value_df = find_correlation(fh_mtx, fh_sorted_features)
    fh_p_value_df = fh_p_value_df.sort_values(by=['Combined P-value', 'column'], ascending=True)
    fh_p_value_df['column'] = fh_p_value_df['column'].apply(lambda x: "_".join(x.split("_")[1:]))
    fh_p_value_df['split'] = fh_p_value_df['column'].apply(lambda x: split_string(x, fh_sorted_features))
    fh_p_value_df = fh_p_value_df[~fh_p_value_df['split'].apply(lambda x: 'SGN_P2_T' in x)]
    top_fh_paired = pd.unique(fh_p_value_df['column'].tolist())


    # AMP
    amp_p_value_df = find_correlation(amp_mtx, amp_sorted_features)
    amp_p_value_df = amp_p_value_df.sort_values(by=['Combined P-value', 'column'], ascending=True)

    top_amp_paired = pd.unique(amp_p_value_df['column'].tolist())

        
    top_cv_set = set(top_cv_paired)
    top_fh_set = set(top_fh_paired)
    top_amp_set = set(top_amp_paired)

    # Find common elements 
    common_features = list(top_cv_set.intersection(top_fh_set, top_amp_set))

    print("Here are the common paired features: ")
    print(common_features)
    

    
if __name__ == "__main__":
    main()
