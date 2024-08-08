import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats
from feature_matrix_generator import generate_feature_mx
from sklearn.metrics import mean_squared_error
from rf import rf_training_vis

def xg_training_vis(df_editing, spacer_length_list):
    for L in spacer_length_list:
        print(f"starting processing for spacer length = {L}")
        df_editing_sub = df_editing[df_editing.spacer_length == L]
        print("Generating feature matrix")
        df_stat_feat = generate_feature_mx(df_editing_sub) 
        print("Complete")

        df_ml_stat = pd.DataFrame()
        smpl = str(L) + 'bp'
        output_folder = f"{smpl}_data"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        X = df_stat_feat.drop(columns=['eff'])
        y = df_stat_feat['eff'].tolist()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        lst_y_eff = []
        lst_y_pre = []

        print("Starting training")
        data_collection = []
        for k, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"Training number: {k+1} of {sum(1 for _ in kf.split(X))}")
            X_train = X.loc[train_index, :]
            X_test = X.loc[test_index, :]
            y_train = [y[i] for i in train_index]
            y_test = [y[i] for i in test_index]
            
            regr = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
            regr.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
            
            y_rf = regr.predict(X_test)
            lst_y_eff.extend(y_test)
            lst_y_pre.extend(y_rf)

            r2 = r2_score(y_test, y_rf)
            pearson_corr = stats.pearsonr(y_rf, y_test)[0]
            spearman_corr = stats.spearmanr(y_rf, y_test)[0]
            data_collection.append({'Editor': smpl, 'stat': 'r-squared', 'score': r2})
            data_collection.append({'Editor': smpl, 'stat': 'pearsonr', 'score': pearson_corr})
            data_collection.append({'Editor': smpl, 'stat': 'spearmanr', 'score': spearman_corr})
            mse = mean_squared_error(y_test, y_rf)
            data_collection.append({'Editor': smpl, 'stat': 'mse', 'score': mse})
        
        df_ml_stat = pd.DataFrame(data_collection)
        print("starting visualization output")
        feat_importances = pd.Series(regr.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        plt.figure(figsize=(8, 10))
        sns.barplot(y=feat_importances.index, x=feat_importances.values)
        plt.title(f"{smpl} Feature Importances")
        plt.savefig(os.path.join(output_folder, f"{smpl}_importances.png"), bbox_inches='tight')
        plt.close()
        
        explainer = shap.Explainer(regr)
        shap_values = explainer(X)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.savefig(os.path.join(output_folder, f"{smpl}_shap_values.png"), bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', lw=0.75)
        sns.scatterplot(x=lst_y_eff, y=lst_y_pre)
        plt.xlabel('ABE editing')
        plt.ylabel('ABE prediction')
        plt.savefig(os.path.join(output_folder, f"{smpl}_scatterplot.png"), bbox_inches='tight')
        plt.close()
        
        print(smpl)
        print(df_ml_stat)
        average_scores = df_ml_stat.groupby('stat')['score'].mean()
        # Print the average scores
        print(average_scores)
        print(f"ended processing for spacer length = {L}")

# df_editing = pd.read_csv("../p12_editing_summary.csv")
# df_editing['target_no_pam'] = df_editing['target'].apply(lambda x: x[:-10])
# df_editing['pam'] = df_editing['target'].apply(lambda x: x[-10:])
# xg_training_vis(df_editing, [23])
# rf_training_vis(df_editing, [23])
