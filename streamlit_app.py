from functools import lru_cache
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
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
        selection = show_data(data)

        if selection and len(selection["selected_rows"]) > 0:
            doc = selection["selected_rows"][0]
            show_doc(doc)


def show_data(data: pd.DataFrame) -> AgGrid:
    options = GridOptionsBuilder.from_dataframe(
        data, enableRowGroup=True, enableValue=True
    )
    # options.configure_side_bar()
    options.configure_selection("single")

    selection = AgGrid(
        data,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        height=300,
        theme="streamlit",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=False,
    )

    return selection


def show_doc(doc: pd.DataFrame):
    st.header(doc["title"])

    st.subheader("Summary")
    summary = get_summary(doc["id"])
    if summary is not None:
        st.write(summary)
    else:
        st.write(doc["summary"])

    topics = get_topics(doc["id"])
    if topics is not None:
        st.subheader("Topics")
        st.altair_chart(
            alt.Chart(topics)
            .mark_bar(tooltip=True)
            .encode(x="score", y="topic", color="topic"),
            use_container_width=True,
        )


@lru_cache(maxsize=256)
def get_summary(id: str) -> Optional[str]:
    data = dm.get_summaries_data()

    if data is not None:
        summary = get_rows_by_id(data, id)
        if summary is not None:
            return summary["summary"].iloc[0]

    return None


def get_rows_by_id(data: pd.DataFrame, id: str) -> Optional[pd.DataFrame]:
    rows = data[data["id"] == id]

    if rows is not None:
        return rows

    return None


@lru_cache(maxsize=256)
def get_topics(id: str) -> Optional[pd.DataFrame]:
    data = dm.get_topics_data()

    if data is not None:
        topics = get_rows_by_id(data, id)
        if topics is not None:
            return topics[["topic", "score"]]

    return None


if __name__ == "__main__":
    streamlit()
