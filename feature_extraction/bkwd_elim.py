from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd 
import joblib 


# df_editing: main dataframe 
# L: desired spacer length
def backward_elimination(df_stat_feat):
    """
    Performs backward elimination. Takes in a dataframe with maximum number of features, and identifies the most important features based on the highest average R-squared values.
    """

    X = df_stat_feat.drop(columns=['eff'])
    y = df_stat_feat['eff'].to_list()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    while len(X.columns) > 5:  
        print(f"Eliminating features. Left: {len(X.columns)}")
        scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train = [y[i] for i in train_index]
            y_test = [y[i] for i in test_index]
            regr = RandomForestRegressor(n_estimators=200, max_depth=40, random_state=42)
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
            scores.append(r2_score(y_test, y_pred))
        avg_r2 = np.mean(scores)
        results.append((list(X.columns), avg_r2))
        feature_importances = pd.Series(regr.feature_importances_, index=X.columns)
        num_to_remove = max(1, int(len(X.columns) * 0.1))  
        least_important = feature_importances.nsmallest(num_to_remove).index
        X = X.drop(columns=least_important)
    
    best_features, best_score = max(results, key=lambda x: x[1])
    print(f"Best R-squared: {best_score} with features: {best_features}")

    print("Calculating feature importance ranking...")
    X_best = df_stat_feat[best_features]
    y = df_stat_feat['eff']  
    
    X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.2, random_state=42)
    
    print("Fitting model...")
    model = RandomForestRegressor(n_estimators=200, max_depth=40, random_state=42)
    model.fit(X_train, y_train)
    print('Saving model...')
    joblib.dump(model, "best_model.pkl") 

    print("Measuring the contribution of each feature...")
    results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': results.importances_mean
    })
    
    importance_df.sort_values(by='importance_mean', ascending=False, inplace=True)

    sorted_features = importance_df['feature'].tolist()
    
    print("Features sorted by importance:", sorted_features)
    return sorted_features, importance_df


        
def opposite_backward_elimination(df_stat_feat, num_remove=1):
    """
    Performs backward elimination removing a fixed number of least important features per iteration.
    Tracks average importance scores for batch-removed features to provide some ordering.
    """

    X = df_stat_feat.drop(columns=['eff'])
    y = df_stat_feat['eff'].to_list()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    feature_importance_history = []

    while len(X.columns) > num_remove:
        scores = []
        importance_sums = pd.Series(0, index=X.columns)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train = [y[i] for i in train_index]
            y_test = [y[i] for i in test_index]
            regr = RandomForestRegressor(n_estimators=200, max_depth=40, random_state=42)
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
            scores.append(r2_score(y_test, y_pred))
            importance_sums += pd.Series(regr.feature_importances_, index=X.columns)

        average_importances = importance_sums / kf.get_n_splits()
        least_importants = average_importances.nsmallest(num_remove).index.tolist()
        for feature in least_importants:
            feature_importance_history.append((feature, average_importances[feature]))
        X = X.drop(columns=least_importants)

    for feature in X.columns:
        feature_importance_history.append((feature, average_importances[feature]))

    sorted_features = sorted(feature_importance_history, key=lambda x: x[1])

    print("Ordered least important features based on average importances during batch removal:")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance}")

    return [f[0] for f in sorted_features]