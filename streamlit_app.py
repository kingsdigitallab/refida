import base64
from functools import lru_cache
from typing import Optional

import altair as alt
import pandas as pd
import spacy_streamlit
import streamlit as st
from spacy_streamlit import visualize_ner
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

from refida import data as dm
from settings import ENTITY_SECTIONS, PROJECT_TITLE, SPACY_ENTITY_TYPES


def streamlit():
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")
    st.title(PROJECT_TITLE)

    data_section()


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
        enable_enterprise_modules=True,
        fit_columns_on_grid_load=True,
        gridOptions=options.build(),
        height=300,
        theme="streamlit",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=False,
    )

    return grid


def show_data(data: pd.DataFrame, selection: pd.DataFrame):
    n_rows = selection.shape[0]
    topic_aggr = "score"

    if n_rows == 0:
        st.warning("No data found")
        return

    st.header("Insights")

    if n_rows > 1:
        st.info("Multiple documents available, showing aggregate information")
        topic_aggr = "mean(score)"

        show_topics(selection, topic_aggr)
        show_topics(selection, "count(topic)", threshold=0.15)

        for section in ENTITY_SECTIONS:
            show_entities(selection, section)
    else:
        doc_idx = data[data["id"] == selection["id"].iloc[0]].index[0]
        show_doc(selection, doc_idx, topic_aggr)


def show_doc(data: pd.DataFrame, idx: int, topic_aggr: str):
    doc = data.iloc[0]

    st.subheader(doc["title"])

    with st.expander("View document", expanded=False):
        with open(doc["file"], "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    summary = get_summary(tuple([doc["id"]]))
    if summary is not None:
        st.write(summary)
    else:
        st.write(doc["summary"])

    show_topics(data, topic_aggr)

    for section in ENTITY_SECTIONS:
        show_entities(data, section)

    st.subheader("View entities in context")
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )
    section = st.radio("Choose context", [None] + ENTITY_SECTIONS)
    if section:
        st.subheader(section.capitalize())
        show_entities_in_context(section, idx)


@lru_cache(maxsize=256)
def get_summary(ids: tuple[str]) -> Optional[str]:
    data = dm.get_summaries_data()

    if data is not None:
        summary = get_rows_by_id(data, ids)
        if summary is not None:
            return summary["summary"].iloc[0]

    return None


def get_rows_by_id(data: pd.DataFrame, ids: tuple[str]) -> Optional[pd.DataFrame]:
    rows = data[data["id"].isin(ids)]

    if rows is not None:
        return rows

    return None


def show_topics(data: pd.DataFrame, aggr: str, threshold: float = 0.0):
    topics = get_topics(tuple(data["id"].values.tolist()))
    if topics is not None:
        if threshold > 0.0:
            topics = topics[topics["score"] >= threshold]

        st.subheader(
            f"Impact categories with threshold >= {threshold} aggregated by {aggr}"
        )
        st.altair_chart(
            alt.Chart(topics)
            .mark_bar(tooltip=True)
            .encode(x=aggr, y="topic", color="topic"),
            use_container_width=True,
        )


@lru_cache(maxsize=256)
def get_topics(ids: tuple[str]) -> Optional[pd.DataFrame]:
    data = dm.get_topics_data()

    if data is not None:
        topics = get_rows_by_id(data, ids)
        if topics is not None:
            return topics[["topic", "score"]]

    return None


def show_entities(data: pd.DataFrame, section: str):
    entities = get_entities(section, tuple(data["id"].values.tolist()))
    if entities is not None:
        st.subheader(f"Entities in the {section}")
        st.altair_chart(
            alt.Chart(entities)
            .mark_bar(tooltip=True)
            .encode(x="entity", y="count(entity)", color="label"),
            use_container_width=True,
        )


@lru_cache(maxsize=256)
def get_entities(section: str, ids: tuple[str]) -> Optional[pd.DataFrame]:
    data = dm.get_entities_data(section)

    if data is not None:
        entities = get_rows_by_id(data, ids)
        if entities is not None:
            return entities[["label", "entity"]]

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


if __name__ == "__main__":
    streamlit()
