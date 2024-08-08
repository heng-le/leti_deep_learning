import argparse
import pandas as pd
from rf import rf_training_vis
from bkwd_elim import backward_elimination, opposite_backward_elimination
from feature_matrix_generator import generate_feature_mx
from xgb_vis import xg_training_vis
from feature_pair import find_correlation, find_correlation_between_two_groups
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

    generated_mx.to_csv("./generated_mtx.csv", index=False)

    # print("Listing features (least important first)...")
    # sorted_least_important_features = opposite_backward_elimination(generated_mx, 10)
    # print(sorted_least_important_features)

    # p_value_df = find_correlation_between_two_groups(generated_mx, sorted_least_important_features[-40:-20], sorted_least_important_features[0:-20])
    # p_value_df = p_value_df.sort_values(by=['Combined P-value', 'column'], ascending=True)
    # p_value_df = p_value_df.dropna()
    # print(p_value_df.head(50))
    # p_value_df.to_csv('p_values_20_to_bottom.csv', index=False)




if __name__ == "__main__":
    main()
