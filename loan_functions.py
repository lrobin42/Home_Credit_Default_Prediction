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
from sklearn.metrics import (accuracy_score, fbeta_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV


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

