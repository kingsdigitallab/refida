import geojson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from settings import memory


@memory.cache
def scatter_mapbox(
    data: pd.DataFrame,
    field: str,
    lat: str,
    lon: str,
    facet: str,
) -> go.Figure:
    """
    Create a mapbox scatter plot.

    :param data: Dataframe with the data to plot.
    :param field: Field to plot.
    :param lat: Latitude field.
    :param lon: Longitude field.
    :param facet: Field used for colour and size of the bubbles.
    """
    focus = data.iloc[0]

    return px.scatter_mapbox(
        data,
        hover_name=field,
        lat=lat,
        lon=lon,
        color=facet,
        color_continuous_scale="Viridis",
        size=facet,
        size_max=15,
        mapbox_style="carto-positron",
        zoom=3,
        center=dict(lat=focus.lat, lon=focus.lon),
        opacity=0.75,
    ).update_layout(margin=dict(r=0, t=0, l=0, b=0))
