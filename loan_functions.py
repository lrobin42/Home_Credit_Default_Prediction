import numpy as np
import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import polars.selectors as cs
from plotly.subplots import make_subplots
from polars import DataFrame
from skimpy import skim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (fbeta_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from typing import Optional

import pandas as pd
import pingouin as pg
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import polars.selectors as cs
from category_encoders import WOEEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from optuna import Study, Trial, create_study
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import RandomSampler
from optuna.trial._frozen import FrozenTrial
from pandas import DataFrame, Series
from plotly.subplots import make_subplots
from polars import DataFrame
from skimpy import skim
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble._forest import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from numpy.typing import NDArray







def lower_column_names(df) -> DataFrame:
    df: DataFrame = df.select(pl.all().name.map(lambda col_name: col_name.lower()))
    return df


def lower_column_values(df) -> DataFrame:
    """Lowers values in all columns of type string"""
    string_columns = df.select(cs.string()).columns

    for col in string_columns:
        df = df.with_columns(
            df[col]
            .map_elements(lambda x: str.lower(x), return_dtype=pl.String)
            .alias(col)
        )
    return df


def polars_read_csv(csv, ignore_errors=True) -> DataFrame:
    df: DataFrame = pl.read_csv(csv, ignore_errors=True)
    return df


def create_formatted_df(csv) -> DataFrame:
    df = pl.read_csv(csv, ignore_errors=True)  # Use pl.read_csv to use ignore_errors
    df: DataFrame = lower_column_names(df)
    df: DataFrame = lower_column_values(df)
    return df


def make_subplot(figure, df, feature, position, color_series="green") -> None:
    """Makes bar subplot for row and column of figure specified

    Args:
        figure (plotly go): Plotly graph_objects figure
        feature (string): the name of the column within pandas df to make plot of
        position (list): list of integers in [row,column] format for specifying where in figure to plot graph
        labels (list, optional): Title, xlabel, and ylabel for subplots. Defaults to ['',None,None].
    """
    try:
        df: DataFrame = df.to_pandas()

        if color_series == "green":
            color: list[str] = [
                "rgb(191,237,204)",
                "rgb(76,145,151)",
                "rgb(33,92,113)",
                "rgb(22,70,96)",
            ]
        elif color_series == "purple":
            color = [
                "rgb(224, 194, 239)",
                "rgb(168, 138, 211)",
                "rgb(108, 95, 167)",
                "rgb(108, 95, 167)",
            ]
        else:
            color = [
                "rgb(117,180,216)",
                "rgb(7, 132, 204)",
                "rgb(35, 114, 181)",
                "rgb(11, 88, 161)",
            ]

        tallies: DataFrame = df[feature].sort_values(ascending=True).value_counts()
        figure.add_trace(
            go.Bar(
                x=tallies.index,
                y=tallies.values,
                name="",
                marker=dict(color=color),
                hovertemplate="%{x} : %{y}",
                text=tallies.values,
            ),
            row=position[0],
            col=position[1],
        )
        figure.update_layout(bargap=0.2)
    except:
        print(position)


def filter_description(df, csv):
    return df.filter(pl.col("table") == csv)


def convert_to_pandas(polar_df) -> DataFrame:
    return polar_df.to_pandas()


def calculate_value_counts(df, feature):
    df: DataFrame = convert_to_pandas(df)
    value_counts = pd.DataFrame(df[feature].value_counts())
    value_counts["percentages"] = value_counts["count"].apply(
        lambda x: f"{x/len(df):.1%}"
    )
    return value_counts


def plot_histogram(df, feature, title="", text="percentages") -> None:
    # df: DataFrame = convert_to_pandas(df)
    # value_counts = pd.DataFrame(df[feature].value_counts())
    # value_counts["percentages"] = value_counts["count"].apply(lambda x: f"{x/len(df):.1%}")

    value_counts = calculate_value_counts(df, feature)

    fig: px.Figure = (
        px.bar(
            value_counts,
            x=value_counts.index,
            y="count",
            color=value_counts.index,
            color_continuous_scale="Darkmint",  #'Teal',
            text=text,
            hover_data="percentages",
        )
        .update_layout(bargap=0.2, title=title)
        .show()
    )


def column_comparison(df, training_set) -> None:
    common_columns: list = []
    for col in df.columns:
        if col in training_set.columns:
            common_columns.append(col)
    return common_columns


def column_description(description, csv, variable="*") -> str:
    if variable == "*":
        df: DataFrame = description.filter(pl.col("table") == csv)
        return df
    try:
        df: DataFrame = (
            description.filter(pl.col("table") == csv)
            .filter(pl.col("row") == variable)
            .select("description")
            .item()
        )
        return df
    except:
        print("no results found")


def clear(*args):
    for i in args:
        del i


def create_encoder_mapping(df, feature) -> dict[str, int]:
    """Creates dictionary for mapping to encode categorical features

    Args:
        df (polars dataframe): dataframe of features
        feature (string): name of feature of interest

    Returns:
        encoding_key: dictionary of feature values and numbers for encoding
    """
    df: DataFrame = (
        df.group_by(feature)
        .agg(pl.len().alias("values"))
        .sort("values", descending=True)
    )

    options: List = df[feature].to_list()

    numbers_to_encode = list(range(0, len(options)))
    encoding_key = {options[i]: numbers_to_encode[i] for i in range(len(options))}

    if df[feature].str.contains("Yes").to_list()[0] is True:
        encoding_key: dict[str, int] = {"Yes": 1, "No": 0}

    return encoding_key


def encode_feature(df, feature, encoding_key) -> DataFrame:
    """Encode features using supplied encoding key

    Args:
        df (polars): Dataframe to be modified
        feature (string): feature to be encoded
        encoding_key (dict): dictionary of values and numerical codes

    Returns:
        df: input dataframe with feature replaced by numerical values
    """
    df: DataFrame = df.with_columns(
        df.select(pl.col(feature).replace(encoding_key)).cast({feature: pl.Int64})
    )
    return df


def encode_strings_as_integers(df):
    encoding_key = dict()
    for col in df.select(pl.col(cs.String)).columns:
        key: dict[str, int] = create_encoder_mapping(df, col)
        df = encode_feature(df, col, key)
        encoding_key[col] = key
    return df, encoding_key


def condense_entry(df, column, unneeded_entry, final_entry):
    df = df.with_columns(
        df[column]
        .map_elements(lambda x: final_entry if x == unneeded_entry else x)
        .alias(column)
    )
    return df


def int_range(start, end) -> np.ndarray[int, np.dtype[int]]:
    """Generate np.linspace range for limits given, such that all inclusive consecutive integers are included

    Args:
        start (int): lower limit of range
        end (int): upper limit of range

    Returns:
        numpy array of range: integer range of values
    """
    return np.linspace(start, end, len(range(start, end + 1)), dtype=int)

def skopt_bayesian_search(classifier, x_train, y_train, params):
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
    search = BayesSearchCV(estimator=classifier, search_spaces=params, n_jobs=-1, cv=cv)
    search.fit(x_train, y_train)
    return search.best_params_

def calculate_partial_correlation(df, feature1, feature2, control):
    return pg.partial_corr(
        data=df.to_pandas(), x=feature1, y=feature2, covar=control
    )  # ['r'].values[0]


def create_encoder_mapping(df, feature) -> dict[str, int]:
    """Creates dictionary for mapping to encode categorical features

    Args:
        df (polars dataframe): dataframe of features
        feature (string): name of feature of interest

    Returns:
        encoding_key: dictionary of feature values and numbers for encoding
    """
    df: DataFrame = (
        df.group_by(feature)
        .agg(pl.len().alias("values"))
        .sort("values", descending=True)
    )

    options: List = df[feature].to_list()

    numbers_to_encode = list(range(0, len(options)))
    encoding_key = {options[i]: numbers_to_encode[i] for i in range(len(options))}

    if df[feature].str.contains("Yes").to_list()[0] == True:
        encoding_key: dict[str, int] = {"Yes": 1, "No": 0}

    return encoding_key


def encode_feature(df, feature, encoding_key) -> DataFrame:
    """Encode features using supplied encoding key

    Args:
        df (polars): Dataframe to be modified
        feature (string): feature to be encoded
        encoding_key (dict): dictionary of values and numerical codes

    Returns:
        df: input dataframe with feature replaced by numerical values
    """
    df: DataFrame = df.with_columns(
        df.select(pl.col(feature).replace(encoding_key)).cast({feature: pl.Int64})
    )
    return df


def retrieve_csv_columns(csv):
    df = pl.read_csv(csv).head(1)
    columns = set(lower_column_names(df).columns)
    return columns


def find_common_key(table1, table2) -> str:
    """For two names of csv files given as strings, return the common key column between them"""
    sk_id_set = set(
        [
            "bureau.csv",
            "previous_application.csv",
            "credit_card_balance.csv",
            "installments_payments.csv",
            "pos_cash_balance.csv",
            "application_test.csv",
            "application_train.csv",
        ]
    )
    sk_id_bureau_set = set(["bureau.csv", "bureau_balance.csv"])
    sk_id_prev_set = set(
        [
            "pos_cash_balance.csv",
            "installments_payments.csv",
            "credit_card_balance.csv",
            "previous_application.csv",
        ]
    )

    if (table1 in sk_id_set) & (table2 in sk_id_set):
        return "sk_id_curr"
    elif (table1 in sk_id_bureau_set) & (table2 in sk_id_bureau_set):
        return "sk_id_bureau"
    elif (table1 in sk_id_prev_set) & (table2 in sk_id_set):
        return "sk_id_prev"
    else:
        print("no common key found for these tables")


def tables_with_feature(description_df, feature):
    tables = set(
        description_df.filter(pl.col("row") == feature)
        .select("table")
        .unique()
        .to_series()
    )

    if len(tables) == 0:
        print("no results found.")
    else:
        return tables


def calculate_null_count(train, table, feature, mode="null_count"):
    #nulls = []
    if table == "application_{train|test}.csv":
        if mode == "null_count":
            return train[feature].null_count()
        elif mode == "series":
            return train[feature]
        # nulls.append(train[feature].null_count())
    else:
        alternate_df = create_formatted_df(table)
        key = find_common_key("application_train.csv", table)
        temp_df = train.join(alternate_df, on=key, how="inner")
    if mode == "null_count":
        return temp_df[feature].null_count()
    elif mode == "series":
        return temp_df[feature].to_series()


def null_count_comparison(train, description, feature):
    tables = tables_with_feature(description, feature)

    nulls = []
    for table in tables:
        null_count = calculate_null_count(train, table, feature, mode="null_count")
        nulls.append(null_count)
    df = pl.DataFrame(
        data={"table": list(tables), "feature": feature, "null_count": nulls},
        schema={"table": pl.String, "feature": pl.String, "null_count": pl.Int64},
    )

    return df


def replace_feature(train, null_df, feature) -> DataFrame:
    """Replace feature with fewer null values in train, else return training set unchanged"""
    null_min = null_df["null_count"].min()
    table = null_df.filter(pl.col("null_count") == null_min).select("table").item()
    if table == "application_{train|test}.csv":
        return train
    else:
        key = find_common_key("application_train.csv", table)
        alternate_df = create_formatted_df(table)[[key, feature]]
        alternate_feature = train.join(alternate_df, on=key, how="inner")[
            feature
        ].to_series()
        train = train.with_columns(alternate_feature.alias(feature))
        return train


def hypothesis_test_multiple_proportions(train_df, feature):
    """Conducts hypothesis test comparing target proportion and feature proportion"""

    # Determine the number of values in feature
    feature_default_df = pd.crosstab(
        train_df[feature].to_pandas(),
        train_df["target"].to_pandas(),
        rownames=[feature],
        colnames=["target"],
    )
    feature_default_df["default_proportion"] = feature_default_df.iloc[
        :, 1
    ] / feature_default_df.sum(axis=1)
    feature_default_df["total"] = feature_default_df.iloc[:, :-1].sum(axis=1)

    # pooled sample proportion
    p1_default_prop = feature_default_df.default_proportion.iloc[0]
    p2_default_prop = feature_default_df.default_proportion.iloc[1]
    p1_population = feature_default_df.total.iloc[0]
    p2_population = feature_default_df.total.iloc[1]

    p = (p1_default_prop * p1_population + p2_default_prop * p2_population) / (
        p1_population + p2_population
    )

    # standard error
    se = np.sqrt((p * (1 - p)) * ((1 / p1_population) + (1 / p2_population)))

    # test statistic
    z = (p1_default_prop - p2_default_prop) / se

    if np.abs(z) < 1.64485:
        print(
            f"Fail to reject the null hypothesis. We can assume the default percentage to be the same across {feature}."
        )
    else:
        print(
            f"z = {z:.3f}. Reject null hypothesis. The proportion of credit defaults across values of {feature} is not equal."
        )


def np_to_df(array, feature_list, designation="pd"):
    if designation == "pd":
        return pd.DataFrame(array, columns=feature_list)
    if designation == "pl":
        df = pl.DataFrame(array)
        df.columns = feature_list
        return df


def conduct_grid_search_tuning(
    model, grid, x_train, y_train, refit, scoring=make_scorer(fbeta_score, beta=2), cv=5
):
    """Conducts gridsearch for specified model and hyperparameter settings

    Args:
        model (string): string specifying model to test, must be 'knn', 'logistic_regression','decision_tree', or 'random_forest'
        grid (dictionary): grid of lists specifying options for hyperparameters to tune
        xy (list): x and y for model fitting, should be in [x_train,y_train] format
        scoring(string/callable): string defines scoring method to be used within grid search
    """

    grid_search = GridSearchCV(
        model, grid, cv=cv, scoring=scoring, refit=refit, n_jobs=-1
    )
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_

    return best_params  # , grid_search

def instantiate_numerical_simple_imputer(
    trial: Trial, fill_value: int = -1
) -> SimpleImputer:
    strategy = trial.suggest_categorical(
        "numerical_strategy", ["mean", "median", "most_frequent", "constant"]
    )
    return SimpleImputer(strategy=strategy, fill_value=fill_value)


def instantiate_categorical_simple_imputer(
    trial: Trial, fill_value: str = "missing"
) -> SimpleImputer:
    strategy = trial.suggest_categorical(
        "categorical_strategy", ["most_frequent", "constant"]
    )
    return SimpleImputer(strategy=strategy, fill_value=fill_value)


def instantiate_woe_encoder(trial: Trial) -> WOEEncoder:
    params = {
        "sigma": trial.suggest_float("sigma", 0.001, 5),
        "regularization": trial.suggest_float("regularization", 0, 5),
        "randomized": trial.suggest_categorical("randomized", [True, False]),
    }
    return WOEEncoder(**params)


def instantiate_robust_scaler(trial: Trial) -> RobustScaler:
    params: dict = {
        "with_centering": trial.suggest_categorical("with_centering", [True, False]),
        "with_scaling": trial.suggest_categorical("with_scaling", [True, False]),
    }
    return RobustScaler(**params)


def instantiate_extra_trees(trial: Trial, warm_start=False) -> ExtraTreesClassifier:
    params: dict = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "max_features": trial.suggest_float("max_features", 0, 1),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "n_jobs": -1,
        "random_state": 42,
    }
    return ExtraTreesClassifier(**params, warm_start=warm_start)


def instantiate_logistic_regression(trial) -> LogisticRegression:
    solver: str = trial.suggest_categorical(
        "solver", ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]
    )
    if solver == "newton-cholesky":
        penalty:str = trial.suggest_categorical("penalty", ["l2", None])
        params: dict = {
            "solver": solver,
            "penalty": penalty,
            "C": trial.suggest_float("C", 0.0, 10.0),
        }
    elif solver == "lbfgs":
        penalty = trial.suggest_categorical("penalty", ["l2", None])
        params = {
            "solver": solver,
            "penalty": penalty,
            "C": trial.suggest_float("C", 0.0, 10.0),
        }
    elif solver == "liblinear":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        params = {
            "solver": solver,
            "penalty": penalty,
            "C": trial.suggest_float("C", 0.0, 10.0),
        }
    elif solver == "newton-cg":
        penalty = trial.suggest_categorical("penalty", ["l2", None])
        params = {
            "solver": solver,
            "penalty": penalty,
            "C": trial.suggest_float("C", 0.0, 10.0),
        }
    elif solver == "sag":
        penalty = trial.suggest_categorical("penalty", ["l2", None])
        params = {
            "solver": solver,
            "penalty": penalty,
            "C": trial.suggest_float("C", 0.0, 10.0),
        }
    elif solver == "saga":
        penalty = trial.suggest_categorical("penalty", ["l2", None])
        params = {
            "solver": solver,
            "penalty": penalty,
            "C": trial.suggest_float("C", 0.0, 10.0),
        }
    return LogisticRegression(**params)


# def instantiate_lgbm_classifier(trial):
#    params = {
#        "boosting_type": trial.suggest_categorical(
#            "boosting_type", ["gbdt", "dart", "rf"]
#        ),
#        "num_leaves": trial.suggest_int("num_leaves", 2, 64),
#        "max_depth": trial.suggest_int("max_depth", -1, 20),
#        "n_estimators": trial.suggest_int("n_estimators", 35, 150),
#        "class_weight": "balanced",    }
#    return LGBMClassifier(**params)


def instantiate_lgbm_classifier(trial):
    params = {
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["gbdt", "dart"]
        ),
        "num_leaves": trial.suggest_int("num_leaves", 2, 64),
        "max_depth": trial.suggest_int("max_depth", -1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 35, 150),
        "class_weight": "balanced",
    }
    return LGBMClassifier(**params, verbose=-1)


def instantiate_xgboost(trial):
    params = {
        "objective": trial.suggest_categorical(
            "objective", ["binary:hinge", "binary:logistic"]
        ),
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart", "gblinear"]),
        "max_leaves": trial.suggest_int("max_leaves", 1, 10, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 15, 4),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise"]),
        "n_estimators": trial.suggest_int("n_estimators", 50, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1),
    }
    return XGBClassifier(**params)


def instantiate_random_forest(trial):
    params = {
        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        ),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 64),
        "max_depth": trial.suggest_int("max_depth", 2, 64),
        "n_estimators": trial.suggest_int("n_estimators", 35, 150),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample"]
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
    }
    return RandomForestClassifier(**params)


# def instantiate_random_forest(trial):

# "criterion": ["gini", "entropy", "log_loss"],
# "max_depth": list(range(1, 11)),
# "max_features": ["sqrt", "log2", None],

# n_estimators = trial.suggest_int("n_estimators", 10, 100)
# max_depth = trial.suggest_int("max_depth", 2, 32)
# criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

# clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
# return clf


def model_selector(clf_string, trial: Trial):
    if clf_string == "logistic_regression":
        model = instantiate_logistic_regression(trial)
    elif clf_string == "random_forest":
        model = instantiate_random_forest(trial)
    elif clf_string == "extra_trees":
        model = instantiate_extra_trees(trial)
    elif clf_string == "lightgbm":
        model = instantiate_lgbm_classifier(trial)
    elif clf_string == "xgboost":
        model = instantiate_xgboost(trial)

    return model


def instantiate_numerical_pipeline(trial: Trial) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", instantiate_numerical_simple_imputer(trial)),
            ("scaler", instantiate_robust_scaler(trial)),
        ]
    )
    return pipeline


def instantiate_categorical_pipeline(trial: Trial) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", instantiate_categorical_simple_imputer(trial)),
            ("encoder", instantiate_woe_encoder(trial)),
        ]
    )
    return pipeline


def instantiate_processor(
    trial: Trial, numerical_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:

    numerical_pipeline = instantiate_numerical_pipeline(trial)
    categorical_pipeline = instantiate_categorical_pipeline(trial)

    processor = ColumnTransformer(
        [
            ("numerical_pipeline", numerical_pipeline, numerical_columns),
            ("categorical_pipeline", categorical_pipeline, categorical_columns),
        ]
    )

    return processor


def instantiate_model(
    classifier,
    trial: Trial,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:

    processor = instantiate_processor(trial, numerical_columns, categorical_columns)

    clf = model_selector(classifier, trial)

    model = Pipeline([("processor", processor), ("classifier", clf)])

    return model

def calculate_model_statistics(y_true, y_predict, beta=3.0, title="statistics"):
    """Uses actual y and predicted y values to return a dataframe of accuracy, precision, recall, and f-beta values as well as false negative and false posititive rates for a given classifier

    Args:
        y_true (numpy array or data series): dependent variable values from the dataset
        y_predict (_type_): dependent variable values arising from model
        beta (float, optional): Beta value to determine weighting between precision and recall in the f-beta score.Defaults to beta value set in global scope of this notebook.
        title (str, optional): _description_. Defaults to "statistics".

    Returns:
        model_statistics: pandas dataframe of statistics
    """

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()

    # calculate statistics from confusion matrix
    # accuracy: float = accuracy_score(y_true, y_predict)
    roc_auc: float = roc_auc_score(y_true, y_predict)
    mcc: float = matthews_corrcoef(y_true, y_predict)
    f_beta: float = fbeta_score(y_true, y_predict, beta=beta)

    precision: float = precision_score(y_true, y_predict, zero_division=0)
    recall: float = recall_score(y_true, y_predict)
    balanced_accuracy = balanced_accuracy_score(y_true,y_predict)
    # false_negative_rate: float = fn / (tn + fp + fn + tp)
    # false_positive_rate: float = fp / (tn + fp + fn + tp)

    return pd.DataFrame(
        data={
            title: [
                roc_auc,
                mcc,
                f_beta,
                precision,
                recall,
                balanced_accuracy
            ]
        },
        index=[
            "roc_auc",
            "matthews_correlation",
            "f_beta",
            "precision",
            "recall",
            "balanced_accuracy"
        ],
    )

def objective_1(classifier, trial : Trial, x : DataFrame, y : np.ndarray | Series, numerical_columns : Optional[list[str]]=None, categorical_columns : Optional[list[str]]=None, random_state : int=42) -> float:
  if numerical_columns is None:
    numerical_columns = [
      *x.select_dtypes(exclude=['object', 'category']).columns
    ]

  if categorical_columns is None:
    categorical_columns = [
      *x.select_dtypes(include=['object', 'category']).columns
    ]

  model = instantiate_model(classifier,trial, numerical_columns, categorical_columns)

  kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
  roc_auc_scorer:float = make_scorer(roc_auc_score, response_method='predict')
  scores = cross_val_score(model, x, y, scoring=roc_auc_scorer, cv=kf)

  return np.min([np.mean(scores), np.median([scores])])