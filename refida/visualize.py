from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from settings import DASHBOARD_PLOT_MIN_HEIGHT, DASHBOARD_PLOT_RATIO, memory


@memory.cache
def histogram(
    data: pd.DataFrame,
    x: Optional[str],
    y: Optional[str],
    colour: str,
    labels: dict[str, str] = {},
    height: Optional[int] = None,
) -> go.Figure:
    """
    Generate a histogram of the data.

    :param data: The data to plot.
    :param x: The column to plot on the x-axis.
    :param y: The column to plot on the y-axis.
    :param colour: The column to colour the bars by.
    :param labels: A dictionary of labels to use for the x-axis and y-axis.
    """
    if not height:
        height = get_height(data)

    return px.histogram(data, x=x, y=y, color=colour, labels=labels, height=height)


def get_height(
    data: pd.DataFrame,
    min_height: int = DASHBOARD_PLOT_MIN_HEIGHT,
    ratio: float = DASHBOARD_PLOT_RATIO,
) -> Optional[int]:
    """
    Get the height for plot.

    :param data: The data to plot.
    :param min_height: The minimum height to return.
    :param ratio: The ratio to use for the height and the length of the data.
    """
    if data is None:
        return None

    number_of_rows = len(data.index)
    height = max(
        number_of_rows * ratio if number_of_rows > 10 else min_height, min_height
    )

    return int(height)


@memory.cache
def parallel_categories(
    data: pd.DataFrame, dimensions: list[str], colour: dict[str, str]
) -> go.Figure:
    """
    Generate a parallel category plot of the data.

    :param data: The data to plot.
    :param dimensions: The dimensions to plot.
    :param colour: A dictionary of colours to use for each value.
    """
    return px.parallel_categories(
        data,
        dimensions=dimensions,
        color=colour,
        height=get_height(data, ratio=1.25),
    ).update_traces(line=dict(shape="hspline"))


@memory.cache
def scatter(data: pd.DataFrame, x: str, y: str, colour: str, size: str) -> go.Figure:
    """
    Generate a scatter plot of the data.

    :param data: The data to plot.
    :param x: The column to plot on the x-axis.
    :param y: The column to plot on the y-axis.
    :param colour: The column to colour the points by.
    :param size: The column to size the points by.
    """
    return px.scatter(data, x=x, y=y, color=colour, size=size, height=get_height(data))


@memory.cache
def bar(
    data: pd.DataFrame, x: str, y: str, colour: str, labels: dict[str, str] = {}
) -> go.Figure:
    """
    Generate a bar chart of the data.

    :param data: The data to plot.
    :param x: The column to plot on the x-axis.
    :param y: The column to plot on the y-axis.
    :param colour: The column to colour the bars by.
    :param labels: A dictionary of labels to use for the x-axis and y-axis.
    """
    return px.bar(data, x=x, y=y, color=colour, labels=labels, height=get_height(data))


@memory.cache
def scatter_mapbox(
    data: pd.DataFrame,
    field: str,
    lat: str,
    lon: str,
    facet: str,
    focus: tuple[float, float],
) -> go.Figure:
    """
    Generate a mapbox scatter plot.

    :param data: Dataframe with the data to plot.
    :param field: Field to plot.
    :param lat: Latitude field.
    :param lon: Longitude field.
    :param facet: Field used for colour and size of the bubbles.
    """
    return px.scatter_mapbox(
        data,
        hover_name=field,
        lat=lat,
        lon=lon,
        size=facet,
        mapbox_style="carto-positron",
        zoom=1,
        center=dict(lat=focus[0], lon=focus[1]),
        opacity=0.75,
        height=700,
    ).update_layout(margin=dict(r=0, t=0, l=0, b=0))
