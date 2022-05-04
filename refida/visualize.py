from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from settings import memory


@memory.cache
def histogram(
    data: pd.DataFrame,
    x: Optional[str],
    y: Optional[str],
    colour: str,
    height: int = None,
    labels: dict[str, str] = {},
) -> go.Figure:
    """
    Generate a histogram of the data.

    :param data: The data to plot.
    :param x: The column to plot on the x-axis.
    :param y: The column to plot on the y-axis.
    :param colour: The column to colour the bars by.
    :param height: The height of the plot.
    :param labels: A dictionary of labels to use for the x-axis and y-axis.
    """
    return px.histogram(data, x=x, y=y, color=colour, height=height, labels=labels)


@memory.cache
def parallel_categories(
    data: pd.DataFrame, dimensions: list[str], colour: dict[str, str], height: int
) -> go.Figure:
    """
    Generate a parallel category plot of the data.

    :param data: The data to plot.
    :param dimensions: The dimensions to plot.
    :param colour: A dictionary of colours to use for each value.
    :param height: The height of the plot.
    """
    return px.parallel_categories(
        data, dimensions=dimensions, color=colour, height=height
    )


@memory.cache
def scatter(
    data: pd.DataFrame, x: str, y: str, colour: str, size: str, height: int
) -> go.Figure:
    """
    Generate a scatter plot of the data.

    :param data: The data to plot.
    :param x: The column to plot on the x-axis.
    :param y: The column to plot on the y-axis.
    :param colour: The column to colour the points by.
    :param size: The column to size the points by.
    :param height: The height of the plot.
    """
    return px.scatter(data, x=x, y=y, color=colour, size=size, height=height)


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
    return px.bar(data, x=x, y=y, color=colour, labels=labels)


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
