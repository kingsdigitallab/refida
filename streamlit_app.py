import base64
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from spacy_streamlit import visualize_ner
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

import settings as _s
from cli import TopicsSection
from refida import data as dm
from refida import visualize as vm

STYLE_RADIO_INLINE = "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>"


def streamlit():
    st.set_page_config(page_title=_s.PROJECT_TITLE, layout="wide")
    st.title(_s.PROJECT_TITLE)

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
        doc_idx = data[data[_s.FIELD_ID] == selection[_s.FIELD_ID].iloc[0]].index[0]

        show_doc(selection)
    else:
        st.info("Multiple documents available, showing aggregate information")

    if show_topics_view():
        show_topics(selection)

    if show_entities_view():
        st.header("Entities")
        for section in _s.DATA_ENTITY_SECTIONS:
            show_entities(selection, section)

        if doc is not None:
            st.subheader("View entities in context")
            st.write(STYLE_RADIO_INLINE, unsafe_allow_html=True)
            section = st.radio("Choose context", [None] + _s.DATA_ENTITY_SECTIONS)
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

    summary = get_summary(tuple([doc[_s.FIELD_ID]]))
    if summary is not None:
        st.write(summary)
    else:
        st.write(doc[_s.DATA_SUMMARY])


@st.experimental_memo
def get_summary(ids: tuple[str]) -> Optional[str]:
    data = dm.get_summaries_data()

    if data is not None:
        summary = get_rows_by_id(data, ids)
        if summary is not None:
            return summary[_s.FEATURE_SUMMARY].iloc[0]

    return None


def get_rows_by_id(data: pd.DataFrame, ids: tuple[str]) -> Optional[pd.DataFrame]:
    rows = data[data[_s.FIELD_ID].isin(ids)]

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
    aggr = f"{aggr_function}({_s.FEATURE_TOPIC_SCORE})"
    if n_rows == 1:
        aggr = _s.FEATURE_TOPIC_SCORE

    threshold = st.slider("Minimum score/confidence", 0.0, 1.0, 0.75, 0.05)

    topics = get_topics(source, tuple(data[_s.FIELD_ID].values.tolist()))
    if topics is None or topics.empty:
        st.warning("No topics found")
        return

    topics = topics.merge(data, on=_s.FIELD_ID)

    if threshold > 0.0:
        topics = topics[topics[_s.FEATURE_TOPIC_SCORE] >= threshold]

    topics = topics.sort_values(by=_s.FEATURE_TOPIC_TOPIC, ascending=True)
    topics_aggr = (
        topics.groupby(_s.FEATURE_TOPIC_TOPIC).agg(aggr_function).reset_index()
    )

    number_of_topics = topics[_s.FEATURE_TOPIC_TOPIC].shape[0]
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
        vm.histogram(
            topics_aggr,
            _s.FEATURE_TOPIC_SCORE,
            _s.FEATURE_TOPIC_TOPIC,
            _s.FEATURE_TOPIC_TOPIC,
        ).update_layout(yaxis=dict(categoryorder="category ascending")),
        use_container_width=True,
    )

    colour_df = topics[_s.FEATURE_TOPIC_TOPIC].copy()
    palette = px.colors.qualitative.Plotly
    palette_len = len(palette)
    colours = colour_df.map(
        {v: palette[i % palette_len] for i, v in enumerate(colour_df.unique())}
    )

    st.plotly_chart(
        vm.parallel_categories(
            topics, [_s.FEATURE_TOPIC_TOPIC, _s.DATA_UOA], colours, height
        ),
        use_container_width=True,
    )

    topics_aggr = (
        topics.groupby([_s.FEATURE_TOPIC_TOPIC, _s.DATA_UOA])
        .agg(aggr_function)
        .reset_index()
    )
    st.plotly_chart(
        vm.scatter(
            topics_aggr,
            _s.DATA_UOA,
            _s.FEATURE_TOPIC_TOPIC,
            _s.FEATURE_TOPIC_TOPIC,
            _s.FEATURE_TOPIC_SCORE,
            height,
        ).update_layout(
            xaxis=dict(categoryorder="category ascending"),
            yaxis=dict(
                categoryorder="category ascending", tickmode="linear", type="category"
            ),
        ),
        use_container_width=True,
    )


@st.experimental_memo
def get_topics(label: str, ids: tuple[str]) -> Optional[pd.DataFrame]:
    data = dm.get_topics_data(label)

    if data is not None:
        return get_rows_by_id(data, ids)

    return None


@st.experimental_memo
def convert_df(df):
    return df.to_csv().encode("utf-8")


def show_entities(data: pd.DataFrame, section: str):
    entities = get_entities(section, tuple(data[_s.FIELD_ID].values.tolist()))
    if entities is not None:
        st.subheader(f"Entities in the {section}")
        st.plotly_chart(
            vm.histogram(
                entities, _s.FEATURE_ENTITY_ENTITY, None, _s.FEATURE_ENTITY_LABEL
            ).update_layout(dict(xaxis=dict(tickangle=-45))),
            use_container_width=True,
        )


@st.experimental_memo
def get_entities(section: str, ids: tuple[str]) -> Optional[pd.DataFrame]:
    data = dm.get_entities_data(section)

    if data is not None:
        entities = get_rows_by_id(data, ids)
        if entities is not None:
            return entities[[_s.FEATURE_ENTITY_LABEL, _s.FEATURE_ENTITY_ENTITY]]

    return None


def show_entities_in_context(section: str, idx: int):
    doc = dm.get_spacy_doc(section, idx)
    if doc:
        visualize_ner(
            doc,
            labels=_s.SPACY_ENTITY_TYPES,
            show_table=False,
            title="",
        )


def show_geo(data: pd.DataFrame):
    places = get_places(tuple(data[_s.FIELD_ID].values.tolist()))
    if places is None:
        st.warning("No places data found")
        return

    st.subheader("Local/national/global mentions")

    places = places.sort_values(by=_s.FEATURE_GEO_CATEGORY)
    with st.expander("View data", expanded=False):
        st.write(places)

    min_mentions = st.slider("Minimum number of mentions", 1, 20, 1, 1)
    places = places[places["count"] >= min_mentions]

    st.plotly_chart(
        vm.histogram(
            places,
            _s.FEATURE_GEO_CATEGORY,
            "count",
            _s.FEATURE_GEO_CATEGORY,
            {"count": "number of mentions"},
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        vm.bar(
            places,
            _s.FEATURE_GEO_PLACE,
            "count",
            _s.FEATURE_GEO_CATEGORY,
            {"count": "number of mentions"},
        ).update_layout(
            dict(xaxis=dict(categoryorder="total descending", tickangle=-45))
        ),
        use_container_width=True,
    )

    st.subheader("Map")
    places = places.drop(columns=[_s.FEATURE_GEO_CATEGORY])
    places = (
        places.groupby(places.columns[:-1].values.tolist())
        .sum()
        .reset_index()
        .sort_values(by="count", ascending=False)
    )

    focus = places.iloc[0]
    st.plotly_chart(
        vm.scatter_mapbox(
            places,
            _s.FEATURE_GEO_PLACE,
            _s.FEATURE_GEO_PLACE_LAT,
            _s.FEATURE_GEO_PLACE_LON,
            "count",
            (focus[_s.FEATURE_GEO_PLACE_LAT], focus[_s.FEATURE_GEO_PLACE_LON]),
        ),
        use_container_width=True,
    )

    with st.expander("Geo data", expanded=False):
        st.write(places)


@st.experimental_memo
def get_places(
    ids: tuple[str], section: Optional[str] = None
) -> Optional[pd.DataFrame]:
    data = pd.DataFrame()

    if section:
        data = dm.get_geo_data(section)
    else:
        for section in _s.DATA_ENTITY_SECTIONS:
            section_data = dm.get_geo_data(section)
            if section_data is not None:
                data = pd.concat([data, section_data])

        data = data.drop_duplicates()

    if data is not None:
        places = get_rows_by_id(data, ids)
        if places is not None:
            columns = [
                _s.FEATURE_GEO_CATEGORY,
                _s.FEATURE_GEO_PLACE,
                _s.FEATURE_GEO_PLACE_LAT,
                _s.FEATURE_GEO_PLACE_LON,
            ]

            places = places[columns]
            places["count"] = 0
            return (
                places.groupby(places.columns[:-1].values.tolist())
                .count()
                .reset_index()
            )

    return None


if __name__ == "__main__":
    streamlit()
