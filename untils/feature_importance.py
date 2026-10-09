import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def mean_score_decrease(X_train, y_train, model, plot=False, topk=None, n_fold=5, normalize=False, random_state=42):
    """
    Function to calculate mean score decrease. Mean score decrease is model agnostic because it perform a random
    permutation on each of the input columns then collects the score decrease caused by each columns.

    Update: jcopml 1.1
    Scikit-learn 0.22 has released permutation importance which is equivalent measure of mean score decrease.
    Hence, the code is modified using sklearn permutation importance for better performance


    == Example usage ==
    df_imp = mean_score_decrease(X_train, y_train, model, plot=True, topk=10)


    == Arguments ==
    X_train: pandas DataFrame
        training input features

    y_train: pandas Series
        training labels

    model: scikit-learn pipeline or estimator
        trained scikit-learn pipeline or estimator

    plot: bool
        show feature importance barplot

    topk: int
        number or k most important feature to show in the barplot

    n_fold: int
        number of permutation to perform before aggregating the mean score decrease

    normalize: bool
        normalize the importance so that they summed up to 1

    random_state: int
        random state for reproducible result


    == Return ==
    pandas DataFrame with the features and importance
    """
    imp = permutation_importance(model, X_train, y_train, n_repeats=n_fold, n_jobs=1, random_state=random_state)

    df_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": imp["importances_mean"],
        "stdev": imp["importances_std"]
    }).sort_values("importance", ascending=False)

    if normalize:
        df_imp[["importance", "stdev"]] = df_imp[["importance", "stdev"]] / df_imp.importance.sum()

    if topk:
        df_imp = df_imp.head(topk)

    if plot:
        plt.figure(figsize=(15, 5))
        plt.bar(range(len(df_imp)), df_imp.importance, yerr=df_imp.stdev, color='c', error_kw={"capsize": 5})
        plt.xticks(range(len(df_imp)), df_imp.feature, rotation=45, horizontalalignment='right')
        plt.ylabel('importance')
        plt.title("Mean Score Decrease", fontsize=14);
    return df_imp

