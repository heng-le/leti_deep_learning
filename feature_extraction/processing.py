import argparse
import pandas as pd
from rf import rf_training_vis, lr_training_vis
from bkwd_elim import backward_elimination, opposite_backward_elimination
from feature_matrix_generator import generate_feature_mx
from xgb_vis import xg_training_vis
from feature_pair import find_correlation 
from remove_paired import update_dataframe

def load_and_preview_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        print("CSV file loaded successfully.")
        return data
    except Exception as e:
        print(f"Failed to read the CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Load and preview a CSV file.")

    parser.add_argument('csv_file', type=str, help="The path to the CSV file to be processed.")

    args = parser.parse_args()

    df_editing = load_and_preview_csv(args.csv_file)
    spacer_length = 23
    df_editing = df_editing[df_editing.spacer_length == spacer_length]
    generated_mx = generate_feature_mx(df_editing)
    spacer_length_list = [23]


    # lr_training_vis(generated_mx, spacer_length)
    
    # # list of spacer_lengths to model 
    # # random forest training 
    # # prints out r-squared, pearsonr, spearmanr, mse
    rf_training_vis(generated_mx, spacer_length)

    # xgboost 
    # prints out r-squared, pearsonr, spearmanr, mse
    # xg_training_vis(df_editing, spacer_length_list)

    # backward elimination 
    # returns a list of features, sorted by importance
    # sorted_features, importance_df = backward_elimination(generated_mx)


    # # finding paired features correlation with eff 
    # print('finding paired feature correlation')
    # p_value_df = find_correlation(generated_mx, sorted_features)
    # p_value_df = p_value_df.sort_values(by=['Combined P-value', 'column'], ascending=True)
    # p_value_df = p_value_df.dropna()
    # print(p_value_df.head(50))
    # p_value_df.to_csv('amp_p_values.csv', index=False)
    # print('\n')

    # print("updating dataframe...")
    # df_with_paired = update_dataframe(generated_mx, sorted_features)    
    # print('\n')

    # # # perform random forest on updated dataframe 
    # print("Performing random forest modeling on dataframe with most important paired features...")
    # rf_training_vis(df_with_paired, spacer_length)
    # most_important_features = backward_elimination(df_with_paired)
    # print('\n')

    
    # # # finding least important features 
    # print("Finding the least important features")
    # sorted_least_important_features = opposite_backward_elimination(generated_mx)
    # print(sorted_least_important_features)
    # df_with_paired_opposite = update_dataframe(generated_mx, sorted_least_important_features[:30])   
    # print("Performing random forest modeling on dataframe with least important paired features...")
    # rf_training_vis(df_with_paired_opposite, spacer_length)
    # least_important_features = backward_elimination(df_with_paired_opposite)


if __name__ == "__main__":
    main()
