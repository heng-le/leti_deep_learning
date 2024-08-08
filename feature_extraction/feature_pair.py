import pandas as pd 
import statistics 
import scipy
from scipy.stats import mannwhitneyu, combine_pvalues
from feature_matrix_generator import generate_feature_mx
import collections 
# check for correlation between encoded values and efficiency 
bin_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

# need to get rid of non-binary columns (nucleotide content, editing pos, strand)
unwanted_list = ['SGN_Strand', 'Editing_Position', 'Acontent', 'Tcontent', 'Ccontent', 'Gcontent', 'GCcontent', 'Editing_mt']
begin = 'SGN_mt_w_'

def check_eff(col_list, eff_list):
    df = pd.DataFrame({
        'Category': col_list,
        'Efficiency': eff_list
    })
    mean_efficiency = df.groupby('Category')['Efficiency'].mean().reset_index()
    return mean_efficiency

def test_significant_category(col_list, eff_list):
    df = pd.DataFrame({
        'Category': col_list,
        'Efficiency': eff_list
    })
    mean_efficiency = df.groupby('Category')['Efficiency'].mean().reset_index()
    mean_efficiency_list = list(mean_efficiency.itertuples(index=False, name=None))
    if mean_efficiency.empty:
        return None, pd.DataFrame()
    
    index_of_max = mean_efficiency['Efficiency'].idxmax()
    target_category = mean_efficiency.loc[index_of_max, 'Category']
    
    p_values = []
    comparisons = []
    
    target_data = df[df['Category'] == target_category]['Efficiency']
    is_universally_significant = True
    
    for category in mean_efficiency['Category']:
        if category != target_category:
            other_data = df[df['Category'] == category]['Efficiency']
            stat, p_value = scipy.stats.mannwhitneyu(target_data, other_data, alternative='two-sided')
            p_values.append(p_value)
            comparisons.append(f"{target_category} vs {category}")
            
            if p_value >= 0.05:
                is_universally_significant = False

    combined_pvalues = scipy.stats.combine_pvalues(p_values, method='fisher')[1]
    p_value_df = pd.DataFrame({
        'Comparison': comparisons,
        'P-value': p_values,
        'Combined P-value': [combined_pvalues] * len(comparisons),
        'Mean Efficiency': [mean_efficiency_list] * len(comparisons) 
    })
    
    if is_universally_significant:
        return target_category, p_value_df
    else:
        return None, p_value_df

def find_correlation(df, feature_list):
    bin_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    unwanted_list = ['SGN_Strand', 'Editing_Position', 'Acontent', 'Tcontent', 'Ccontent', 'Gcontent', 'GCcontent', 'Editing_mt']
    begin = 'SGN_mt_w_'
    
    feature_list = [feature for feature in feature_list if feature not in unwanted_list and not feature.startswith(begin)]
    
    overall_results = {}
    for i, feature1 in enumerate(feature_list):
        for feature2 in feature_list[i+1:]:  
            f1 = df[feature1].tolist()
            f2 = df[feature2].tolist()
            results = []
            for index, tup in enumerate(zip(f1, f2)):
                result = bin_dict.get(tup, None)
                results.append(result)
            overall_results[f"paired_{feature1}_{feature2}"] = results

    
    paired_dataframe = pd.DataFrame(overall_results)
    result_df = pd.concat([df, paired_dataframe], axis=1)
    eff_list = result_df['eff'].tolist()

    p_values = []
    for column in result_df.columns:
        if column.startswith('paired'):
            col_list = result_df[column].tolist()
            target, p_value_df = test_significant_category(col_list, eff_list)
            if target is not None:
                for index, row in p_value_df.iterrows():
                    p_values.append({
                        'column': column,
                        'Comparison': row['Comparison'],
                        'P-value': row['P-value'],
                        'Combined P-value': row['Combined P-value'],
                        'Mean Efficiency': row['Mean Efficiency']
                    })

    return pd.DataFrame(p_values)




def find_correlation_between_two_groups(df, group1, group2):
    bin_dict = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    unwanted_list = ['SGN_Strand', 'Editing_Position', 'Acontent', 'Tcontent', 'Ccontent', 'Gcontent', 'GCcontent', 'Editing_mt']
    begin = 'SGN_mt_w_'
    
    group1 = [feature for feature in group1 if feature not in unwanted_list and not feature.startswith(begin)]
    group2 = [feature for feature in group2 if feature not in unwanted_list and not feature.startswith(begin)]
    
    overall_results = {}
    for i, feature1 in enumerate(group1):
        for feature2 in group2:  
            f1 = df[feature1].tolist()
            f2 = df[feature2].tolist()
            results = []
            for index, tup in enumerate(zip(f1, f2)):
                result = bin_dict.get(tup, None)
                results.append(result)
            overall_results[f"paired_{feature1}_{feature2}"] = results

    
    paired_dataframe = pd.DataFrame(overall_results)
    result_df = pd.concat([df, paired_dataframe], axis=1)
    eff_list = result_df['eff'].tolist()

    p_values = []
    for column in result_df.columns:
        if column.startswith('paired'):
            col_list = result_df[column].tolist()
            target, p_value_df = test_significant_category(col_list, eff_list)
            if target is not None:
                for index, row in p_value_df.iterrows():
                    p_values.append({
                        'column': column,
                        'Comparison': row['Comparison'],
                        'P-value': row['P-value'],
                        'Combined P-value': row['Combined P-value'],
                        'Mean Efficiency': row['Mean Efficiency']
                    })

    return pd.DataFrame(p_values)

    

    
        
                
