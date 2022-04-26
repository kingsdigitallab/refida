import base64
from functools import lru_cache
from typing import Optional

import altair as alt
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from spacy_streamlit import visualize_ner
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

from cli import TopicsSection
from refida import data as dm
from settings import (
    DATA_ENTITY_SECTIONS,
    DATA_SUMMARY,
    DATA_UOA,
    FEATURE_COUNTRY,
    FEATURE_ENTITY_ENTITY,
    FEATURE_ENTITY_LABEL,
    FEATURE_ENTITY_TEXT,
    FEATURE_LAT,
    FEATURE_LON,
    FEATURE_PLACE_CATEGORY,
    FEATURE_SUMMARY,
    FEATURE_TOPIC_SCORE,
    FEATURE_TOPIC_TOPIC,
    FIELD_ID,
    PROJECT_TITLE,
    SPACY_ENTITY_TYPES,
)

STYLE_RADIO_INLINE = "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>"
UK = "United Kingdom"


def streamlit():
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")
    st.title(PROJECT_TITLE)

    with st.sidebar:
        sidebar()

    data_section()


def sidebar():
    st.title("Insights")

    st.session_state.view = st.radio(
        "Choose view",
        ("Topics", "Entities", "Locations"),
    )


def show_topics_view():
    return st.session_state.view == "Topics"


def show_entities_view():
    return st.session_state.view == "Entities"


def show_geo_view():
    return st.session_state.view == "Locations"


def data_section():
    st.header("Case studies")

    data = dm.get_etl_data()
    if data is not None:
        grid = show_data_grid(data)

        selection = grid["selected_rows"]

        if selection:
            show_data(data, pd.DataFrame(selection))
        else:
            show_data(data, grid["data"])


def show_data_grid(data: pd.DataFrame) -> dict:
    data = data.drop(columns=["names", "compressed"])
    options = GridOptionsBuilder.from_dataframe(
        data, enableRowGroup=True, enableValue=True
    )
    options.configure_selection("multiple")

    grid = AgGrid(
        data,
        data_return_mode=DataReturnMode.FILTERED,
        fit_columns_on_grid_load=False,
        gridOptions=options.build(),
        height=300,
        theme="streamlit",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=False,
    )

    return grid


def show_data(data: pd.DataFrame, selection: pd.DataFrame):
    n_rows = selection.shape[0]

    if n_rows == 0:
        st.warning("No data found")
        return

    doc = None
    doc_idx = None

    if n_rows == 1:
        doc = selection.iloc[0]
        doc_idx = data[data[FIELD_ID] == selection[FIELD_ID].iloc[0]].index[0]

        show_doc(selection)
    else:
        st.info("Multiple documents available, showing aggregate information")

    if show_topics_view():
        show_topics(selection)

    if show_entities_view():
        st.header("Entities")
        for section in DATA_ENTITY_SECTIONS:
            show_entities(selection, section)

        if doc is not None:
            st.subheader("View entities in context")
            st.write(STYLE_RADIO_INLINE, unsafe_allow_html=True)
            section = st.radio("Choose context", [None] + DATA_ENTITY_SECTIONS)
            if section:
                st.subheader(section.capitalize())
                show_entities_in_context(section, doc_idx)

    if show_geo_view():
        st.header("Locations")
        show_geo(selection)


def show_doc(data: pd.DataFrame):
    doc = data.iloc[0]

    st.subheader(doc["title"])

    with st.expander("View document", expanded=False):
        with open(doc["file"], "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    summary = get_summary(tuple([doc[FIELD_ID]]))
    if summary is not None:
        st.write(summary)
    else:
        st.write(doc[DATA_SUMMARY])


@lru_cache(maxsize=256)
def get_summary(ids: tuple[str]) -> Optional[str]:
    data = dm.get_summaries_data()

    if data is not None:
        summary = get_rows_by_id(data, ids)
        if summary is not None:
            return summary[FEATURE_SUMMARY].iloc[0]

    return None


def get_rows_by_id(data: pd.DataFrame, ids: tuple[str]) -> Optional[pd.DataFrame]:
    rows = data[data[FIELD_ID].isin(ids)]

    if rows is not None:
        return rows

    return None


def show_topics(data: pd.DataFrame):
    st.header("Topics")

    n_rows = data.shape[0]

    st.write(STYLE_RADIO_INLINE, unsafe_allow_html=True)
    source = st.radio(
        "Choose topics source",
        TopicsSection._member_names_,
    )
    aggr_function = st.radio(
        "Choose topics aggregation function",
        ("count", "mean"),
    )
    aggr = f"{aggr_function}({FEATURE_TOPIC_SCORE})"
    if n_rows == 1:
        aggr = FEATURE_TOPIC_SCORE

    threshold = st.slider("Minimum score/confidence", 0.0, 1.0, 0.75, 0.05)

    topics = get_topics(source, tuple(data[FIELD_ID].values.tolist()))
    if topics is None or topics.empty:
        st.warning("No topics found")
        return

    topics = topics.merge(data, on=FIELD_ID)

    if threshold > 0.0:
        topics = topics[topics[FEATURE_TOPIC_SCORE] >= threshold]

    topics = topics.sort_values(by=FEATURE_TOPIC_TOPIC, ascending=True)
    topics_aggr = topics.groupby(FEATURE_TOPIC_TOPIC).agg(aggr_function).reset_index()

    number_of_topics = topics[FEATURE_TOPIC_TOPIC].shape[0]
    height = number_of_topics * 2 if number_of_topics > 10 else 400

    st.subheader(
        f"Topics in {source} with confidence >= {threshold} aggregated by {aggr}"
    )
    with st.expander("View data", expanded=False):
        st.write(topics)
        st.download_button(
            label="Download data as CSV",
            data=convert_df(topics),
            file_name="topics.csv",
            mime="text/csv",
        )

    st.plotly_chart(
        px.histogram(
            topics_aggr,
            x=FEATURE_TOPIC_SCORE,
            y=FEATURE_TOPIC_TOPIC,
            color=FEATURE_TOPIC_TOPIC,
            height=height,
        ).update_layout(yaxis=dict(categoryorder="category ascending")),
        use_container_width=True,
    )

    colour_df = topics[FEATURE_TOPIC_TOPIC].copy()
    palette = px.colors.qualitative.Plotly
    colours = colour_df.map(
        {
            v: palette[i % len(px.colors.qualitative.Plotly)]
            for i, v in enumerate(colour_df.unique())
        }
    )

    st.plotly_chart(
        px.parallel_categories(
            topics,
            dimensions=[FEATURE_TOPIC_TOPIC, DATA_UOA],
            color=colours,
            height=height,
        ),
        use_container_width=True,
    )

    topics_aggr = (
        topics.groupby([FEATURE_TOPIC_TOPIC, DATA_UOA]).agg(aggr_function).reset_index()
    )
    st.plotly_chart(
        px.scatter(
            topics_aggr,
            x=DATA_UOA,
            y=FEATURE_TOPIC_TOPIC,
            color=FEATURE_TOPIC_TOPIC,
            size=FEATURE_TOPIC_SCORE,
            height=height,
        ).update_layout(
            xaxis=dict(categoryorder="category ascending"),
            yaxis=dict(
                categoryorder="category ascending", tickmode="linear", type="category"
            ),
        ),
        use_container_width=True,
    )


@lru_cache(maxsize=256)
def get_topics(label: str, ids: tuple[str]) -> Optional[pd.DataFrame]:
    data = dm.get_topics_data(label)

    if data is not None:
        return get_rows_by_id(data, ids)

    return None


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


def show_entities(data: pd.DataFrame, section: str):
    entities = get_entities(section, tuple(data[FIELD_ID].values.tolist()))
    if entities is not None:
        st.subheader(f"Entities in the {section}")
        st.plotly_chart(
            px.histogram(
                entities,
                x=FEATURE_ENTITY_ENTITY,
                color=FEATURE_ENTITY_LABEL,
            ),
            use_container_width=True,
        )


@lru_cache(maxsize=256)
def get_entities(section: str, ids: tuple[str]) -> Optional[pd.DataFrame]:
    data = dm.get_entities_data(section)

    if data is not None:
        entities = get_rows_by_id(data, ids)
        if entities is not None:
            return entities[[FEATURE_ENTITY_LABEL, FEATURE_ENTITY_ENTITY]]

    return None


def show_entities_in_context(section: str, idx: int):
    doc = dm.get_spacy_doc(section, idx)
    if doc:
        visualize_ner(
            doc,
            labels=SPACY_ENTITY_TYPES,
            show_table=False,
            title="",
        )


def show_geo(data: pd.DataFrame):
    geo_df = get_geo(tuple(data[FIELD_ID].values.tolist()))
    if geo_df is not None:
        focus = geo_df.iloc[0]

        st.subheader("National/global mentions")
        st.warning("Please note the number of mentions includes duplicate mentions.")

        countries = geo_df[[FEATURE_COUNTRY, FEATURE_PLACE_CATEGORY, "count"]]
        countries = (
            countries.groupby([FEATURE_COUNTRY, FEATURE_PLACE_CATEGORY])
            .sum()
            .reset_index()
        )
        mentions_uk = countries[countries[FEATURE_COUNTRY] == UK]["count"].count()
        mentions_global = countries[countries[FEATURE_COUNTRY] != UK]["count"].count()

        with st.expander("View data", expanded=False):
            st.write(countries)

        st.plotly_chart(
            px.histogram(
                countries,
                x=FEATURE_PLACE_CATEGORY,
                y="count",
                color=FEATURE_PLACE_CATEGORY,
                labels={"count": "number of mentions"},
            ),
            use_container_width=True,
        )
        st.info(
            "The United Kingdom, or locations within the UK, are mentioned "
            f"{mentions_uk} times. Global locations are mentioned "
            f"{mentions_global} times."
        )
        st.plotly_chart(
            px.bar(
                countries,
                x=FEATURE_COUNTRY,
                y="count",
                color=FEATURE_PLACE_CATEGORY,
                labels={"count": "number of mentions"},
            ).update_layout(
                dict(xaxis=dict(categoryorder="total descending", tickangle=-45))
            ),
            use_container_width=True,
        )

        st.subheader("Map")
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(
                    latitude=focus.lat, longitude=focus.lon, zoom=7, bearing=0, pitch=0
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        geo_df,
                        stroked=True,
                        filled=True,
                        pickable=True,
                        opacity=0.75,
                        get_position="[lon, lat]",
                        get_fill_color=[255, 140, 0],
                        get_line_color=[0, 0, 0],
                        get_radius="count * 100",
                    ),
                ],
                tooltip={"text": "{entity}\n{count}"},
            )
        )

        with st.expander("Geo data", expanded=False):
            st.write(geo_df)


@lru_cache(maxsize=256)
def get_geo(ids: tuple[str], section: Optional[str] = None) -> Optional[pd.DataFrame]:
    data = pd.DataFrame()

    if section:
        data = dm.get_geo_data(section)
    else:
        for section in DATA_ENTITY_SECTIONS:
            data = pd.concat([data, dm.get_geo_data(section)])

    if data is not None:
        geo_df = get_rows_by_id(data, ids)
        if geo_df is not None:
            geo_df = geo_df.drop(columns=[FIELD_ID, FEATURE_ENTITY_TEXT])
            geo_df["count"] = 0
            return (
                geo_df.groupby(
                    [
                        FEATURE_ENTITY_ENTITY,
                        FEATURE_COUNTRY,
                        FEATURE_PLACE_CATEGORY,
                        FEATURE_LAT,
                        FEATURE_LON,
                    ]
                )
                .count()
                .reset_index()
                .sort_values(by="count", ascending=False)
            )

    return None


if __name__ == "__main__":
    streamlit()
