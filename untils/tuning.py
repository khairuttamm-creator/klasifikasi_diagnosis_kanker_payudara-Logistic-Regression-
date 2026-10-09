import numpy as np


class GridSearchParams:
    def __init__(self):
        self.rf_params = {
            "algo__n_estimators": [100, 150, 200],
            "algo__max_depth": [20, 50, 80],
            "algo__max_features": [0.3, 0.6, 0.8],
            "algo__min_samples_leaf": [1, 5, 10]
        }

        self.knn_params = {
            "algo__n_neighbors": np.arange(1, 31, 2),
            "algo__weights": ['uniform', 'distance'],
            "algo__p": [1, 1.5, 2]
        }

        self.svm_params = {
            "algo__gamma": np.logspace(-3, 3, 7),
            "algo__C": np.logspace(-3, 3, 7)
        }

        self.xgb_params = {
            "algo__max_depth": [3, 6, 10],
            "algo__colsample_bytree": [0.4, 0.6, 0.8],
            "algo__n_estimators": [100, 150, 200],
            "algo__subsample": [0.4, 0.6, 0.8],
            "algo__gamma": [1, 5, 10],
            "algo__learning_rate": [0.01, 0.1, 1],
            "algo__reg_alpha": [0.01, 0.1, 10],
            "algo__reg_lambda": [0.01, 0.1, 10]
        }

        self.linreg_params = {
            "algo__fit_intercept": [True, False]
        }

        self.enet_params = {
            "algo__fit_intercept": [True, False],
            "algo__alpha": np.logspace(-3, 2, 6),
            "algo__l1_ratio": [0, 0.25, 0.5, 0.75, 1]
        }

        self.logreg_params = {
            "algo__fit_intercept": [True, False],
            "algo__C": np.logspace(-3, 3, 7)
        }

        prep_poly_params = {
            "prep__numeric__poly__degree": [1, 2, 3],
            "prep__numeric__poly__interaction_only": [True, False]
        }

        self.rf_poly_params = {**prep_poly_params, **self.rf_params}

        self.knn_poly_params = {**prep_poly_params, **self.knn_params}

        self.svm_poly_params = {**prep_poly_params, **self.svm_params}

        self.xgb_poly_params = {**prep_poly_params, **self.xgb_params}

        self.linreg_poly_params = {**prep_poly_params, **self.linreg_params}

        self.enet_poly_params = {**prep_poly_params, **self.enet_params}

        self.logreg_poly_params = {**prep_poly_params, **self.logreg_params}


grid_search_params = GridSearchParams()
