from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import shap
from sklearn.inspection import permutation_importance
from feature_matrix_generator import generate_feature_mx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression




def rf_training_vis(df_stat_feat, L):
    df_ml_stat = pd.DataFrame()
    smpl = str(L) + 'bp'
    output_folder = f"{smpl}_data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    X = df_stat_feat.drop(columns=['eff'])
    y = df_stat_feat.eff.to_list()
    kf = KFold(n_splits=5, shuffle=True, random_state = 42)
    lst_y_eff = []
    lst_y_pre = []
    print("Starting training")
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Training number: {k+1} of {sum(1 for _ in kf.split(X))}")
        X_train = X.loc[train_index, :]
        X_test = X.loc[test_index, :]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        regr = RandomForestRegressor(n_estimators=200, max_depth=40, random_state=None)
        regr.fit(X_train, y_train)
        ##
        y_rf = list(regr.predict(X_test))
        lst_y_eff = lst_y_eff + y_test
        lst_y_pre = lst_y_pre + y_rf
        ##
        idx = len(df_ml_stat)
        df_ml_stat.loc[idx, 'Editor'] = smpl
        df_ml_stat.loc[idx, 'stat'] = 'r-squared'
        df_ml_stat.loc[idx, 'score'] = regr.score(X_test, y_test)
    
        ####
        idx = len(df_ml_stat)
        df_ml_stat.loc[idx, 'Editor'] = smpl
        df_ml_stat.loc[idx, 'stat'] = 'pearsonr'
        df_ml_stat.loc[idx, 'score'] = stats.pearsonr(y_rf, y_test)[0]
    
        ####
        idx = len(df_ml_stat)
        df_ml_stat.loc[idx, 'Editor'] = smpl
        df_ml_stat.loc[idx, 'stat'] = 'spearmanr'
        df_ml_stat.loc[idx, 'score'] = stats.spearmanr(y_rf, y_test)[0]

        ####
        idx = len(df_ml_stat)
        mse = mean_squared_error(y_test, y_rf)
        df_ml_stat.loc[idx, 'Editor'] = smpl
        df_ml_stat.loc[idx, 'stat'] = 'mse'
        df_ml_stat.loc[idx, 'score'] = mse
    
        ##
    feat_name = X_test.columns
    result = permutation_importance(
        regr, X, y, n_repeats=10, random_state=42, n_jobs=2
    )
    print("calculating feature importance...")

    forest_importances = pd.Series(result.importances_mean, index=feat_name)
    #
    dct_feat = dict(zip(feat_name, forest_importances))
    dct_feat = dict(sorted(dct_feat.items(), key=lambda item: item[1], reverse=True))
    print("Features by importance with full model:")
    print(dct_feat)
    
    #
    # plt.figure(figsize = (8, 10))
    # sns.barplot(y = list(dct_feat.keys()), x = list(dct_feat.values()), width=0.6)
    # plt.ylim([10.5, -0.5])
    # plt.yticks(fontsize=18)
    # plt.title(smpl, fontsize=18)
    # plt.savefig(os.path.join(output_folder, f"{smpl}_rf_xgb_importances.png"), bbox_inches='tight')
    # plt.close()
    
    ##
    explainer = shap.TreeExplainer(regr)
    plt.figure(figsize = (12, 10))
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, max_display=10, show=False)
    plt.savefig(os.path.join(output_folder, f"{smpl}_rf_xgb_shap_values.png"), bbox_inches='tight')
    plt.close()
    
    ##
    

    print(smpl)
    print(df_ml_stat)
    average_scores = df_ml_stat.groupby('stat')['score'].mean()
    # Print the average scores
    print("average_stats:")
    print(average_scores)
    print(f"ended random forest processing for spacer length = {L}")
    average_r_squared = average_scores['r-squared']

        # Set Seaborn style
    sns.set(style='whitegrid')
    
    # Create the scatter plot
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], c='k', linestyle='dashed', lw=0.75)
    
    # Scatterplot with improved aesthetics
    sns.scatterplot(x=lst_y_eff, y=lst_y_pre, s=60, color='blue', alpha=0.7)
    
    # Set axis limits
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Customize the axes
    plt.xlabel('ABE editing', fontsize=14)
    plt.ylabel('ABE prediction', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add a title
    plt.title('Scatterplot of Actual vs Predicted', fontsize=16)
    plt.text(0.05, 0.95, f'$R^2 = {average_r_squared:.2f}$', fontsize=25, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    # Save the plot
    plt.savefig(os.path.join(output_folder, f"{smpl}_rf_xgb_scatterplot.png"), bbox_inches='tight')
    plt.close()

#####


def lr_training_vis(df_stat_feat, L):
    df_ml_stat = pd.DataFrame()
    smpl = str(L) + 'bp'
    output_folder = f"{smpl}_data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    X = df_stat_feat.drop(columns=['eff'])
    y = df_stat_feat.eff.to_list()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lst_y_eff = []
    lst_y_pre = []
    print("Starting training")
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Training number: {k+1} of {sum(1 for _ in kf.split(X))}")
        X_train = X.loc[train_index, :]
        X_test = X.loc[test_index, :]
        y_train = [y[i] for i in train_index]
        y_test = [y[i] for i in test_index]
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        ##
        y_lr = list(regr.predict(X_test))
        lst_y_eff = lst_y_eff + y_test
        lst_y_pre = lst_y_pre + y_lr
        ##
        idx = len(df_ml_stat)
        df_ml_stat.loc[idx, 'Editor'] = smpl
        df_ml_stat.loc[idx, 'stat'] = 'r-squared'
        df_ml_stat.loc[idx, 'score'] = regr.score(X_test, y_test)
    
        ####
        idx = len(df_ml_stat)
        df_ml_stat.loc[idx, 'Editor'] = smpl
        df_ml_stat.loc[idx, 'stat'] = 'pearsonr'
        df_ml_stat.loc[idx, 'score'] = stats.pearsonr(y_lr, y_test)[0]
    
        ####
        idx = len(df_ml_stat)
        df_ml_stat.loc[idx, 'Editor'] = smpl
        df_ml_stat.loc[idx, 'stat'] = 'spearmanr'
        df_ml_stat.loc[idx, 'score'] = stats.spearmanr(y_lr, y_test)[0]

        ####
        idx = len(df_ml_stat)
        mse = mean_squared_error(y_test, y_lr)
        df_ml_stat.loc[idx, 'Editor'] = smpl
        df_ml_stat.loc[idx, 'stat'] = 'mse'
        df_ml_stat.loc[idx, 'score'] = mse
    
    print(smpl)
    print(df_ml_stat)
    average_scores = df_ml_stat.groupby('stat')['score'].mean()
    # Print the average scores
    print("average_stats:")
    print(average_scores)
    print(f"ended linear regression processing for spacer length = {L}")

