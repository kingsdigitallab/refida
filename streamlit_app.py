from functools import lru_cache
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

from refida import data as dm
from settings import PROJECT_TITLE


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
            show_data(pd.DataFrame(selection))
        else:
            show_data(grid["data"])


def show_data_grid(data: pd.DataFrame) -> dict:
    data = data.drop(columns=["names", "compressed"])
    options = GridOptionsBuilder.from_dataframe(
        data, enableRowGroup=True, enableValue=True
    )
    options.configure_selection("multiple", use_checkbox=True)

    grid = AgGrid(
        data,
        data_return_mode=DataReturnMode.FILTERED,
        gridOptions=options.build(),
        height=300,
        theme="streamlit",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=False,
    )

    return grid


def show_data(data: pd.DataFrame):
    n_rows = data.shape[0]
    topic_score = "score"

    if n_rows == 0:
        st.warning("No data found")
        return

    st.header("About the data")

    if n_rows > 1:
        st.info("Multiple documents available, showing aggregate information")
        topic_score = "mean(score)"
    else:
        doc = data.iloc[0]
        st.header(doc["title"])
        st.subheader("About the case study")
        summary = get_summary(tuple([doc["id"]]))
        if summary is not None:
            st.write(summary)
        else:
            st.write(doc["summary"])

    topics = get_topics(tuple(data["id"].values.tolist()))
    if topics is not None:
        st.subheader("Impact categories")
        st.altair_chart(
            alt.Chart(topics)
            .mark_bar(tooltip=True)
            .encode(x=topic_score, y="topic", color="topic"),
            use_container_width=True,
        )


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


@lru_cache(maxsize=256)
def get_topics(ids: tuple[str]) -> Optional[pd.DataFrame]:
    data = dm.get_topics_data()

    if data is not None:
        topics = get_rows_by_id(data, ids)
        if topics is not None:
            return topics[["topic", "score"]]

    return None


if __name__ == "__main__":
    streamlit()
