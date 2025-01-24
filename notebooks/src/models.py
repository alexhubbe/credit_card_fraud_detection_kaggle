import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE

def build_model_pipeline(model, preprocessor=None, target_transformer=None):
    """
    Builds a machine learning pipeline with optional preprocessing and target transformation.
    """
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    else:
        pipeline = Pipeline([("model", model)])

    if target_transformer is not None:
        model_pipeline = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model_pipeline = pipeline
    return model_pipeline


def grid_search_cv_model(
    model,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE,
    return_train_score=False,
    scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
):
    """
    Performs grid search cross-validation for the provided model pipeline.
    """
    model_pipeline = build_model_pipeline(model, preprocessor, target_transformer)

    grid_search = GridSearchCV(
        model_pipeline,
        cv=cv,
        param_grid=param_grid,
        scoring=scoring,
        refit=scoring[0],
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search

# ##example of application

# param_grid = {
#     'model__max_depth': [3, 5, 7],
#     'model__learning_rate': [0.1, 0.01, 0.001],
#     'model__subsample': [0.5, 0.7, 1]
# }

# grid_search = grid_search_cv_model(
#     model=xgboost.XGBClassifier(random_state=RANDOM_STATE),
#     preprocessor=preprocessor_ohe_quantile, 
#     target_transformer=None, 
#     param_grid=param_grid,
#     cv=custom_split, 
#     scoring='average_precision',
# )

# grid_search

# %time grid_search.fit(X,y)

# df_scores_gridsearchcv= pd.DataFrame(grid_search.cv_results_)

# with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
#     # Display the DataFrame with full 'params' column content
#     display(df_scores_gridsearchcv[['params', 'mean_test_score', 'std_test_score']].sort_values(by='mean_test_score', ascending=False))




def randomized_search_cv_model(
    model,
    param_distributions,
    preprocessor=None,
    target_transformer=None,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE,
    return_train_score=False,
    scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    n_iter=50  # Number of parameter settings that are sampled
):
    """
    Performs randomized search cross-validation for the provided model pipeline.
    """
    model_pipeline = build_model_pipeline(model, preprocessor, target_transformer)

    randomized_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit=scoring[0],
        random_state=random_state,
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1
    )

    return randomized_search


#example of application:

# # Define the hyperparameter distributions
# param_distributions = {
#     'model__max_depth': stats.randint(3, 10),
#     'model__learning_rate': stats.uniform(0.01, 0.1),
#     'model__subsample': stats.uniform(0.5, 0.5),
#     'model__n_estimators':stats.randint(50, 200)
# # }

# randomized_search = randomized_search_cv_model(
#     model=xgboost.XGBClassifier(random_state=RANDOM_STATE),
#     preprocessor=preprocessor_ohe_quantile, 
#     target_transformer=None, 
#     param_distributions=param_distributions,
#     cv=custom_split, 
#     scoring='average_precision',
#     n_iter=1,
# )

# randomized_search

# %time randomized_search.fit(X,y)

# df_scores_randomizedsearchcv= pd.DataFrame(randomized_search.cv_results_)

# with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
#     # Display the DataFrame with full 'params' column content
#     display(df_scores_randomizedsearchcv[['params', 'mean_test_score', 'std_test_score']].sort_values(by='mean_test_score', ascending=False).iloc[:10])


def organize_results(results):
    """
    Organizes grid search results into a structured DataFrame.
    """
    for key, value in results.items():
        results[key]["time_seconds"] = (
            results[key]["fit_time"] + results[key]["score_time"]
        )

    df_results = (
        pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    )

    expanded_df_results = df_results.explode(
        df_results.columns[1:].to_list()
    ).reset_index(drop=True)

    try:
        expanded_df_results = expanded_df_results.apply(pd.to_numeric)
    except ValueError:
        pass

    return expanded_df_results


