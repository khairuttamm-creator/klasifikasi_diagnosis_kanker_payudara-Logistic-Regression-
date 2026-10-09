import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, ToggleButtons
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

__all__ = [
    'plot_missing_value',
    'plot_correlation_ratio',
    'plot_correlation_matrix',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_pr_curve',
    'plot_classification_report'
]


def plot_missing_value(df, return_df=False, feature_alignment='horizontal', figsize=(15, 8)):
    plt.figure(figsize=figsize)
    if feature_alignment in ['vertical', 'v', 'column', 'c']:
        sns.heatmap(~df.isna(), cbar=False, cmap="Blues")
    elif feature_alignment in ['horizontal', 'h', 'row', 'r']:
        sns.heatmap(~df.isna().T, cbar=False, cmap="Blues")
    else:
        raise Exception("Supported alignment {'vertical', 'horizontal', 'column', 'row'} or {'v', 'h', 'c', 'r'}")
    plt.xticks(rotation=45, horizontalalignment='right')
    if return_df:
        df_miss = pd.DataFrame(df.isna().sum(), columns=['missing_value'])
        df_miss["%"] = round(df_miss / len(df) * 100, 2)
        return df_miss


def plot_correlation_ratio(df, catvar, numvar, report=False):
    df_corr = df.copy()
    eta_table = {}

    for cat in catvar:
        group_count = df_corr.groupby(cat).count()[numvar]
        group_mean = df_corr.groupby(cat).mean(numeric_only=True)[numvar]

        total_mean = (group_mean * group_count).sum()
        total_mean = total_mean.divide(group_count.sum())

        std_cat = (group_count * (group_mean - total_mean) ** 2).sum()
        std_num = ((df_corr[numvar] - total_mean) ** 2).sum()
        eta = std_cat / (std_num + 1e-8)
        eta_table[cat] = eta.values

    eta_table = pd.DataFrame(eta_table, index=numvar).T
    sns.heatmap(eta_table, cmap="bwr", vmin=-1, vmax=1, square=True, annot=True,
                fmt=".1f", cbar=False, linewidths=1, linecolor='k')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.yticks(rotation=0, verticalalignment='center')
    plt.title("Correlation Ratio", fontsize=14)
    plt.ylabel("Categorical Columns", fontsize=14)
    plt.xlabel("Numerical Columns", fontsize=14)

    if report:
        return eta_table


def plot_correlation_matrix(df, target_col, numeric_col='auto'):
    if isinstance(numeric_col, list):
        num_col = numeric_col.copy()
        if target_col not in num_col:
            num_col += [target_col]
    elif numeric_col == "auto":
        num_col = list(df.select_dtypes(include=np.number).columns)
    else:
        raise Exception("numeric_col should be a list of columns name or 'auto'")

    tmp = df[num_col].std()
    single_constant_col = list(tmp[tmp == 0].index)
    if single_constant_col:
        raise ValueError(f"Feature should not have a constant value\n"
                         f"You should remove these column from your data -> {', '.join(single_constant_col)}")

    def _simul_method(method='spearman'):
        df_corr = df[num_col].copy()

        if method == 'pearson_norm':
            power = PowerTransformer(standardize=False)
            df_corr.iloc[:, :] = power.fit_transform(df_corr)
            method_type = 'pearson'
        else:
            method_type = method

        corr = df_corr.corr(method=method_type)
        feature_corr = corr.drop(columns=target_col, index=target_col)
        target_corr = corr[[target_col]].drop(index=target_col)

        def _simul(threshold=0):
            linewidth = 1
            linecolor = 'k'
            mask_feature = None
            mask_target = None

            if threshold > 0:
                mask_feature = (feature_corr < threshold) & (feature_corr > -threshold)
                mask_target = (target_corr < threshold) & (target_corr > -threshold)

            plt.figure(figsize=(15, 7))

            plt.subplot(121)
            sns.heatmap(feature_corr, mask=mask_feature, cmap="bwr", vmin=-1, vmax=1, square=True, annot=True, fmt=".1f",
                        cbar=False, linewidths=linewidth, linecolor=linecolor)
            plt.xticks(rotation=45, horizontalalignment='right')
            plt.title(f"Feature correlation\n({method.upper()})", fontsize=14)

            plt.subplot(122)
            sns.heatmap(target_corr, mask=mask_target, cmap="bwr", vmin=-1, vmax=1, square=True, annot=True, fmt=".1f",
                        cbar=False, linewidths=linewidth, linecolor=linecolor)
            plt.xticks(rotation=45, horizontalalignment='right')
            plt.title(f"Target correlation\n({method.upper()})", fontsize=14)

        interact(_simul, threshold=(0.0, 1.0, 0.1))

    interact(_simul_method, method=ToggleButtons(description='method',
                                                  options=['spearman', 'kendall', 'pearson', 'pearson_norm']))


def plot_confusion_matrix(X_train, y_train, X_test, y_test, model):
    labels = model.classes_

    plt.figure(figsize=(11, 5))

    plt.subplot(121)
    cm = confusion_matrix(y_train, model.predict(X_train), labels=labels)
    sns.heatmap(cm, annot=True, square=True, cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels,
                fmt="d", annot_kws={"fontsize": 15})
    plt.title(f'Train score: {model.score(X_train, y_train):.3f}', fontsize=14)
    plt.xlabel('Prediction', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.yticks(rotation=0, verticalalignment='center')

    plt.subplot(122)
    cm = confusion_matrix(y_test, model.predict(X_test), labels=labels)
    sns.heatmap(cm, annot=True, square=True, cmap='Greens', cbar=False, xticklabels=labels, yticklabels=labels,
                fmt="d", annot_kws={"fontsize": 15})
    plt.title(f'Test score: {model.score(X_test, y_test):.3f}', fontsize=14)
    plt.xlabel('Prediction', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.yticks(rotation=0, verticalalignment='center')


def plot_roc_curve(X_train, y_train, X_test, y_test, model):
    plt.figure(figsize=(13, 6))

    plt.subplot(121)
    prob = model.predict_proba(X_train)[:, 1]
    fpr, tpr, _ = roc_curve(y_train, prob)
    plt.plot(fpr, tpr, 'b-')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"Train ROC_AUC: {auc(fpr, tpr):.3f}", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.subplot(122)
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, 'b-')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"Test ROC_AUC: {auc(fpr, tpr):.3f}", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("FPR")
    plt.ylabel("TPR")


def plot_pr_curve(X_train, y_train, X_test, y_test, model):
    plt.figure(figsize=(13, 6))

    plt.subplot(121)
    prob = model.predict_proba(X_train)[:, 1]
    p, r, _ = precision_recall_curve(y_train, prob)
    AP = average_precision_score(y_train, prob)
    plt.plot(r, p, 'b-')
    plt.title(f"Train PR_AUC: {auc(r, p):.3f} | AP: {AP:.3f}", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.subplot(122)
    prob = model.predict_proba(X_test)[:, 1]
    p, r, _ = precision_recall_curve(y_test, prob)
    AP = average_precision_score(y_test, prob)
    plt.plot(r, p, 'b-')
    plt.title(f"Test PR_AUC: {auc(r, p):.3f} | AP: {AP:.3f}", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")


def plot_classification_report(X_train, y_train, X_test, y_test, model, report=False, return_df=False):
    if report:
        print("Train report")
        print(classification_report(y_train, model.predict(X_train)))
        print()
        print("Test report")
        print(classification_report(y_test, model.predict(X_test)))
    else:
        plt.figure(figsize=(11, 5))
        plt.subplots_adjust(wspace=0.4)

        plt.subplot(121)
        labels = y_train.unique()
        df_train = pd.DataFrame(classification_report(y_train, model.predict(X_train), labels=labels, output_dict=True))
        df_plot = df_train.iloc[:-1, :len(labels)]
        sns.heatmap(df_plot, vmin=0, vmax=1, annot=True, square=True, cmap='Blues', cbar=False, xticklabels=labels,
                    yticklabels=df_plot.index, fmt=".2f", annot_kws={"fontsize": 15})
        plt.yticks(rotation=0, fontsize=14)
        plt.xticks(rotation=45, horizontalalignment='right', fontsize=12)
        plt.title("Train", fontsize=14)

        plt.subplot(122)
        labels = y_test.unique()
        df_test = pd.DataFrame(classification_report(y_test, model.predict(X_test), labels=labels, output_dict=True))
        df_plot = df_test.iloc[:-1, :len(labels)]
        sns.heatmap(df_plot, vmin=0, vmax=1, annot=True, square=True, cmap='Greens', cbar=False, xticklabels=labels,
                    yticklabels=df_plot.index, fmt=".2f", annot_kws={"fontsize": 15})
        plt.yticks(rotation=0, fontsize=14)
        plt.xticks(rotation=45, horizontalalignment='right', fontsize=12)
        plt.title("Test", fontsize=14)

        if return_df:
            return df_train.T, df_test.T