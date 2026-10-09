import pandas as pd
import os
from warnings import warn

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, StandardScaler, MinMaxScaler, RobustScaler, \
    OneHotEncoder, OrdinalEncoder, Normalizer, MaxAbsScaler
from sklearn.pipeline import Pipeline


def num_pipe(impute='median', poly=None, transform=None, scaling=None, memory=None, n_neighbors=5, weights="uniform"):
    """
    A scikit-learn numerical pipeline used in ColumnTransformer


    == Example usage ==
    from jcopml.pipeline import num_pipe
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('numeric', num_pipe(scaling='minmax'), numerical_columns)
    ])


    == Arguments ==
    impute: {'knn', 'mean', 'median', None}
        type of imputation

    scaling: {'standard', 'minmax', 'robust', 'maxabs', 'normalize', None}
        type of scaling

    transform: {'yeo-johnson', 'box-cox', None}
        type of power transformer

    poly: int or None
        if int is specified, it specifies the polynomial degree

    memory: None or str
        Providing path string enables scikit-learn's pipeline caching.
        See scikit-learn pipeline documentation for more info


    == Return ==
    Scikit-learn pipeline object
    """
    if impute not in ['knn', 'mean', 'median', None]:
        raise Exception("impute only supports {'knn', 'mean', 'median', None}")
    if scaling not in ['standard', 'minmax', 'robust', 'maxabs', 'normalize', None]:
        raise Exception("scaling only supports {'standard', 'minmax', 'robust', 'maxabs', 'normalize'}")
    if transform not in ['yeo-johnson', 'box-cox', None]:
        raise Exception("power_transform only supports {'yeo-johnson', 'box-cox'}")
    if (type(poly) is not int) and (poly is not None):
        raise Exception("poly should be int or None")

    if impute is None:
        steps = []
    elif impute == "knn":
        steps = [('imputer', KNNImputer(n_neighbors=n_neighbors, weights=weights))]
    else:
        steps = [('imputer', SimpleImputer(strategy=impute))]

    if poly is not None:
        steps.append(('poly', PolynomialFeatures(poly)))

    if transform is not None and scaling is not None:
        warn("Transformer has default standardization, so the scaling argument is neglected")

    if transform is not None:
        steps.append(('transformer', PowerTransformer(transform)))
    elif scaling == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaling == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaling == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif scaling == 'maxabs':
        steps.append(('scaler', MaxAbsScaler()))
    elif scaling == 'normalize':
        steps.append(('scaler', Normalizer()))

    if memory is not None:
        os.makedirs('search_cache', exist_ok=True)
        return Pipeline(steps, memory='search_cache')
    else:
        return Pipeline(steps)


def cat_pipe(impute='most_frequent', encoder='onehot', memory=False):
    """
    A scikit-learn categorical pipeline used in ColumnTransformer

    == Example usage ==
    from jcopml.pipeline import cat_pipe
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer([
        ('categoric', cat_pipe(encoder='onehot'), categorical_columns),
    ])

    == Arguments ==
    impute: {'most_frequent', None}
        type of imputation

    encoder: {'onehot', 'ordinal', None}
        type of categorical encoder

    memory: None or str
        Providing path string enables scikit-learn's pipeline caching.
        See scikit-learn pipeline documentation for more info

    == Return ==
    Scikit-learn pipeline object
    """
    if impute not in ['most_frequent', None]:
        raise Exception("impute only supports {'most_frequent', None}")
    if encoder not in ['onehot', 'ordinal', None]:
        raise Exception("encoder should be boolean {'onehot', 'ordinal', None}")

    if impute is None:
        steps = []
    else:
        steps = [('imputer', SimpleImputer(strategy=impute))]

    if encoder is not None:
        if encoder == 'onehot':
            steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
        elif encoder == 'ordinal':
            steps.append(('ordinal', OrdinalEncoder()))

    if memory:
        os.makedirs('search_cache', exist_ok=True)
        return Pipeline(steps, memory='search_cache')
    else:
        return Pipeline(steps)
