import base64
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from spacy_streamlit import visualize_ner
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

import settings as _s
from refida import data as dm
from refida import visualize as vm
from refida.__init__ import __version__

STYLE_RADIO_INLINE = ""


def streamlit():
    st.set_page_config(page_title=_s.PROJECT_TITLE, layout="wide")
    st.error(_s.DASHBOARD_DEVELOPMENT)
    st.title(_s.PROJECT_TITLE)
    st.write(f"v{__version__}")
    st.warning(_s.DASHBOARD_DISCLAIMER)

    with st.sidebar:
        sidebar()

    data_section()


def sidebar():
    st.title("Insights")

    st.session_state.view = st.radio(
        "Choose insights",
        (
            "About the data",
            "Impact categories",
            "Types of impact",
            "Fields of research",
            "Partners",
            "Beneficiaries",
            "Locations",
        ),
    )

    if (
        show_impact_categories_view()
        or show_types_of_impact_view()
        or show_fields_of_research_view()
    ):
        topics_sidebar()

    if show_partners_view() or show_beneficiaries_view():
        entities_sidebar()

    if show_geo_view():
        geo_sidebar()

    filters_sidebar()


def show_about_data_view():
    return get_session_view() == "About the data"


def get_session_view() -> str:
    return st.session_state.view


def show_impact_categories_view():
    return get_session_view() == "Impact categories"


def show_types_of_impact_view():
    return get_session_view() == "Types of impact"


def show_fields_of_research_view():
    return get_session_view() == "Fields of research"


def topics_sidebar():
    view = get_session_view()

    st.subheader(f"{view} options")
    st.session_state.topics_aggr_function = st.radio(
        f"Aggregate {view.lower()} by",
        ("count", "mean"),
        help=_s.DASHBOARD_HELP_TOPICS_AGGR_FUNCTION,
    )


def show_partners_view():
    return get_session_view() == "Partners"


def show_beneficiaries_view():
    return get_session_view() == "Beneficiaries"


def entities_sidebar():
    view = get_session_view()
    st.subheader(f"{view} options")

    st.session_state.entity_types = st.multiselect(
        f"Choose {view.lower()} entity types",
        _s.SPACY_ENTITY_TYPES,
        default=_s.SPACY_ENTITY_TYPES,
    )


def show_geo_view():
    return get_session_view() == "Locations"


def geo_sidebar():
    view = get_session_view()
    st.subheader(f"{view} options")

    st.session_state.geo_min_mentions = st.slider(
        "Minimum number of mentions", 1, 20, 1, 1
    )
    st.session_state.entity_types_geo = st.multiselect(
        f"Choose {view.lower()} entity types",
        _s.SPACY_LOCATION_ENTITY_TYPES,
        default=_s.SPACY_LOCATION_ENTITY_TYPES,
    )


def filters_sidebar():
    st.subheader("Filter the data")
    with st.expander("Filter by unit of assessment", expanded=True):
        st.session_state.filter_uoa = st.multiselect(
            "Unit of assessment", _s.UOA.values()
        )

    st.session_state.filter_topics_score_threshold = st.slider(
        "Topics classification minimum score/confidence",
        0.0,
        1.0,
        0.5,
        0.05,
        help=_s.DASHBOARD_HELP_TOPICS_SCORE_THRESHOLD,
    )
    impact_categories = sorted(_s.TOPIC_CLASSIFICATION_TOPICS)
    with st.expander("Filter by impact categories", expanded=True):
        st.session_state.filter_impact_categories = st.multiselect(
            "Impact categories", impact_categories
        )

    types_of_impact = sorted(_s.TOPIC_CLASSIFICATION_IMPACTS)
    with st.expander("Filter by types of impact", expanded=True):
        st.session_state.filter_types_of_impact = st.multiselect(
            "Types of impact", types_of_impact
        )

    fields_of_research = sorted(_s.get_fields_of_research())
    with st.expander("Filter by fields of research", expanded=True):
        st.session_state.filter_fields_of_research = st.multiselect(
            "Fields of research", fields_of_research
        )


def data_section():
    st.header("Data")

    data = dm.get_etl_data()
    if data is not None:
        data = filter_data(data)

        with st.expander("Using the data grid"):
            st.markdown(_s.DASHBOARD_HELP_DATA_GRID)
        grid = get_data_grid(data)

        selection = grid["selected_rows"]

        if selection:
            selection = pd.DataFrame(selection)
        else:
            selection = grid["data"]

        show_data(data, selection)


def filter_data(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    uoas = get_session_filter_uoa()
    if uoas:
        data = data[data[_s.DATA_UOA].isin(uoas)]

    impact_categories = get_topics(
        [_s.DATA_TEXT],
        get_session_filter_topics_score_threshold(),
        topics=get_session_filter_impact_categories(),
    )
    if impact_categories is not None:
        data = data[data[_s.FIELD_ID].isin(impact_categories[_s.FIELD_ID])]

    types_of_impact = get_topics(
        [_s.DATA_SUMMARY, _s.DATA_DETAILS],
        get_session_filter_topics_score_threshold(),
        topics=get_session_filter_types_of_impact(),
    )
    if types_of_impact is not None:
        data = data[data[_s.FIELD_ID].isin(types_of_impact[_s.FIELD_ID])]

    fields_of_research = get_topics(
        [_s.DATA_RESEARCH],
        get_session_filter_topics_score_threshold(),
        topics=get_session_filter_fields_of_research(),
    )
    if fields_of_research is not None:
        data = data[
            data[_s.DATA_RESEARCH].isnull()
            | data[_s.FIELD_ID].isin(fields_of_research[_s.FIELD_ID])
        ]

    return data


def get_session_filter_uoa() -> list[str]:
    return st.session_state.filter_uoa


def get_session_filter_topics_score_threshold() -> float:
    return st.session_state.filter_topics_score_threshold


def get_session_filter_impact_categories() -> list[str]:
    return st.session_state.filter_impact_categories


def get_session_filter_types_of_impact() -> list[str]:
    return st.session_state.filter_types_of_impact


def get_session_filter_fields_of_research() -> list[str]:
    return st.session_state.filter_fields_of_research


def get_data_grid(data: pd.DataFrame) -> dict:
    data = data[_s.DASHBOARD_COLUMNS_FOR_DATA_GRID]
    options = GridOptionsBuilder.from_dataframe(
        data, enableRowGroup=True, enableValue=True
    )
    options.configure_selection(
        "multiple", use_checkbox=False, rowMultiSelectWithClick=True
    )

    grid = AgGrid(
        data,
        data_return_mode=DataReturnMode.FILTERED,
        fit_columns_on_grid_load=False,
        gridOptions=options.build(),
        height=300,
        theme="streamlit",
        update_mode=GridUpdateMode.SELECTION_CHANGED,
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

    if show_about_data_view():
        st.header("About the data")
        show_about_data(data)
    else:
        with st.expander("About the data", expanded=False):
            show_about_data(data)

    if n_rows == 1:
        doc = selection.iloc[0]
        doc_idx = data[data[_s.FIELD_ID] == selection[_s.FIELD_ID].iloc[0]].index[0]

        show_doc(selection)

    if show_impact_categories_view():
        show_topics("Impact categories", selection)
        return

    if show_types_of_impact_view():
        show_topics("Types of impact", selection, [_s.DATA_SUMMARY, _s.DATA_DETAILS])
        return

    if show_fields_of_research_view():
        show_topics("Fields of research", selection, [_s.DATA_RESEARCH])
        return

    if show_partners_view():
        show_entities(
            "Partners", selection, [_s.DATA_SUMMARY, _s.DATA_SOURCES], doc, doc_idx
        )
        return

    if show_beneficiaries_view():
        show_entities("Beneficiaries", selection, [_s.DATA_DETAILS], doc, doc_idx)
        return

    if show_geo_view():
        show_geo(selection)
        return


def show_about_data(data: pd.DataFrame):
    n_rows = data.shape[0]
    if "metric_n_rows" not in st.session_state:
        st.session_state.metric_n_rows = n_rows

    n_impact_case_studies = data[data[_s.DATA_TYPE].str.startswith("Impact")].shape[0]
    if "metric_n_impact_case_studies" not in st.session_state:
        st.session_state.metric_n_impact_case_studies = n_impact_case_studies

    n_env_statements = data[data[_s.DATA_TYPE].str.startswith("Unit")].shape[0]
    if "metric_n_env_statements" not in st.session_state:
        st.session_state.metric_n_env_statements = n_env_statements

    research = data[
        [
            _s.FIELD_ID,
            _s.DATA_RESEARCH_START,
            _s.DATA_RESEARCH_END,
            _s.DATA_IMPACT_START,
            _s.DATA_IMPACT_END,
        ]
    ].dropna()
    research["research_duration"] = (
        research[_s.DATA_RESEARCH_END] - research[_s.DATA_RESEARCH_START]
    )
    research["impact_duration"] = (
        research[_s.DATA_IMPACT_END] - research[_s.DATA_IMPACT_START]
    )
    research["research_to_impact"] = (
        research[_s.DATA_IMPACT_START] - research[_s.DATA_RESEARCH_START]
    )

    research_duration_avg = research["research_duration"].mean().round(decimals=1)
    if "metric_research_duration_avg" not in st.session_state:
        st.session_state.metric_research_duration_avg = research_duration_avg

    impact_duration_avg = research["impact_duration"].mean().round(decimals=1)
    if "metric_impact_duration_avg" not in st.session_state:
        st.session_state.metric_impact_duration_avg = impact_duration_avg

    research_to_impact_avg = research["research_to_impact"].mean().round(decimals=1)
    if "metric_research_to_impact_avg" not in st.session_state:
        st.session_state.metric_research_to_impact_avg = research_to_impact_avg

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", n_rows, delta=n_rows - st.session_state.metric_n_rows)
    st.session_state.metric_n_rows = n_rows
    col2.metric(
        "Impact case studies",
        n_impact_case_studies,
        delta=n_impact_case_studies - st.session_state.metric_n_impact_case_studies,
    )
    st.session_state.metric_n_impact_case_studies = n_impact_case_studies
    col3.metric(
        "Environment statements",
        n_env_statements,
        delta=n_env_statements - st.session_state.metric_n_env_statements,
    )
    st.session_state.metric_n_env_statements = n_env_statements

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Research duration in years (avg)",
        research_duration_avg,
        delta=research_duration_avg - st.session_state.metric_research_duration_avg,
    )
    st.session_state.metric_research_duration_avg = research_duration_avg
    col2.metric(
        "Impact duration in years (avg)",
        impact_duration_avg,
        delta=impact_duration_avg - st.session_state.metric_impact_duration_avg,
    )
    st.session_state.metric_impact_duration_avg = impact_duration_avg
    col3.metric(
        "Years to impact (avg)",
        research_to_impact_avg,
        delta=research_to_impact_avg - st.session_state.metric_research_to_impact_avg,
    )
    st.session_state.metric_research_to_impact_avg = research_to_impact_avg

    st.subheader("Units of assessment distribution")
    st.plotly_chart(
        vm.histogram(data, None, _s.DATA_UOA, _s.DATA_TYPE), use_container_width=True
    )

    st.subheader("Research/impact timeline")
    research[_s.DATA_RESEARCH_START] = pd.to_datetime(
        research[_s.DATA_RESEARCH_START].apply(lambda x: str(int(x))), yearfirst=True
    )
    research[_s.DATA_RESEARCH_END] = pd.to_datetime(
        research[_s.DATA_RESEARCH_END].apply(lambda x: str(int(x))), yearfirst=True
    )
    research["type"] = "Research period"
    impact = research[[_s.FIELD_ID, _s.DATA_IMPACT_START, _s.DATA_IMPACT_END]]
    impact[_s.DATA_RESEARCH_START] = pd.to_datetime(
        impact[_s.DATA_IMPACT_START].apply(lambda x: str(int(x))), yearfirst=True
    )
    impact[_s.DATA_RESEARCH_END] = pd.to_datetime(
        impact[_s.DATA_IMPACT_END].apply(lambda x: str(int(x))), yearfirst=True
    )
    impact["type"] = "Impact period"
    research = pd.concat([research, impact])
    research["title"] = research[_s.FIELD_ID].apply(
        lambda x: " ".join(x.split("_")[3:])
    )
    st.plotly_chart(
        px.timeline(
            research,
            x_start=_s.DATA_RESEARCH_START,
            x_end=_s.DATA_RESEARCH_END,
            y="title",
            color="type",
            opacity=0.7,
        ),
        use_container_width=True,
    )


def show_doc(data: pd.DataFrame):
    doc = data.iloc[0]

    st.subheader(doc["title"])

    with st.expander("View document", expanded=False):
        with open(doc["file"], "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = (
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                'width="100%" height="500" type="application/pdf"></iframe>'
            )
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


def show_topics(
    title: str, data: pd.DataFrame, sources: list[str] = [_s.DATA_TEXT]
) -> None:
    st.header(title)
    with st.expander("About topic classification", expanded=False):
        st.markdown(_s.DASHBOARD_HELP_TOPICS)

    n_rows = data.shape[0]

    st.write(STYLE_RADIO_INLINE, unsafe_allow_html=True)
    aggr_function = get_session_topics_aggr_function()
    aggr = f"{aggr_function}({_s.FEATURE_TOPIC_SCORE})"
    if n_rows == 1:
        aggr = _s.FEATURE_TOPIC_SCORE

    threshold = get_session_filter_topics_score_threshold()
    topics = get_topics(sources, threshold, tuple(data[_s.FIELD_ID].values.tolist()))
    if topics is None or topics.empty:
        st.warning("No topics found")
        return

    topics = topics.merge(data, on=_s.FIELD_ID)

    topics = topics.sort_values(by=_s.FEATURE_TOPIC_TOPIC, ascending=True)
    topics_aggr = (
        topics.groupby(_s.FEATURE_TOPIC_TOPIC).agg(aggr_function).reset_index()
    )

    st.subheader(f"{title} with confidence >= {threshold} aggregated by {aggr}")
    with st.expander("View data", expanded=False):
        # https://stackoverflow.com/questions/69578431/how-to-fix-streamlitapiexception-expected-bytes-got-a-int-object-conver
        # https://issues.apache.org/jira/browse/ARROW-14087
        st.write(topics.astype(str))
        st.download_button(
            label="Download data as CSV",
            data=convert_df(topics),
            file_name="topics.csv",
            mime="text/csv",
        )

    st.subheader(f"{title} distribution")
    st.plotly_chart(
        vm.histogram(
            topics_aggr,
            _s.FEATURE_TOPIC_SCORE,
            _s.FEATURE_TOPIC_TOPIC,
            _s.FEATURE_TOPIC_TOPIC,
        ).update_layout(yaxis=dict(categoryorder="category ascending")),
        use_container_width=True,
    )

    colour_df = topics[_s.DATA_UOA].copy()
    palette = px.colors.qualitative.Plotly
    palette_len = len(palette)
    colours = colour_df.map(
        {v: palette[i % palette_len] for i, v in enumerate(colour_df.unique())}
    )

    st.subheader(f"Connections between {title.lower()} and unit of assessment")
    st.plotly_chart(
        vm.parallel_categories(topics, [_s.FEATURE_TOPIC_TOPIC, _s.DATA_UOA], colours),
        use_container_width=True,
    )

    st.subheader(f"Correlation between {title.lower()} and unit of assessment")
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
        ).update_layout(
            xaxis=dict(categoryorder="category ascending", tickangle=-45),
            yaxis=dict(
                categoryorder="category ascending",
                tickmode="linear",
                type="category",
            ),
        ),
        use_container_width=True,
    )


def get_session_topics_aggr_function() -> str:
    return st.session_state.topics_aggr_function


@st.experimental_memo
def get_topics(
    sections: list[str],
    threshold: float = 0.0,
    ids: tuple[str] = None,
    topics: list[str] = None,
) -> Optional[pd.DataFrame]:
    data = pd.DataFrame()

    for section in sections:
        section_df = dm.get_topics_data(section)
        if section_df is not None:
            data = pd.concat([data, section_df], ignore_index=True)

    if data is not None:
        data = data.drop_duplicates()
        if data is not None:
            data = data[data[_s.FEATURE_TOPIC_SCORE] >= threshold]

            if topics:
                data = data[data[_s.FEATURE_TOPIC_TOPIC].isin(topics)]

            if ids:
                return get_rows_by_id(data, ids)

            return data

    return None


@st.experimental_memo
def convert_df(df):
    return df.to_csv().encode("utf-8")


def show_entities(
    title: str,
    data: pd.DataFrame,
    sections: list[str],
    doc: Optional[pd.Series],
    doc_idx: Optional[int],
):
    st.header(title)
    with st.expander("About entity extraction", expanded=False):
        st.markdown(_s.DASHBOARD_HELP_ENTITIES)

    entities = get_entities(
        sections, tuple(data[_s.FIELD_ID].values.tolist()), get_session_entity_types()
    )
    if entities is not None:
        st.subheader(f"{title} distribution")
        st.plotly_chart(
            vm.histogram(
                entities,
                None,
                _s.FEATURE_ENTITY_ENTITY,
                _s.FEATURE_ENTITY_LABEL,
                height=_s.DASHBOARD_PLOT_MIN_HEIGHT,
            ),
            use_container_width=True,
        )

    if doc is not None and doc_idx:
        st.subheader(f"View {title.lower()} in context")
        st.write(STYLE_RADIO_INLINE, unsafe_allow_html=True)
        section = sections[0]
        if len(sections) > 1:
            section = st.radio("Choose context", sections)
        st.subheader(section.capitalize())
        show_entities_in_context(section, doc_idx)


def get_session_entity_types() -> list[str]:
    return st.session_state.entity_types


@st.experimental_memo
def get_entities(
    sections: list[str], ids: tuple[str], entity_types: list[str]
) -> Optional[pd.DataFrame]:
    data = pd.DataFrame()

    for section in sections:
        section_df = dm.get_entities_data(section)
        if section_df is not None:
            data = pd.concat([data, section_df], ignore_index=True)

    if data is not None:
        entities = get_rows_by_id(data, ids)
        if entities is not None:
            if entity_types:
                entities = entities[
                    entities[_s.FEATURE_ENTITY_LABEL].isin(entity_types)
                ]
            return entities[
                [_s.FEATURE_ENTITY_LABEL, _s.FEATURE_ENTITY_ENTITY]
            ].sort_values(by=_s.FEATURE_ENTITY_ENTITY)

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
    st.header("Locations")
    with st.expander("About locations", expanded=False):
        st.markdown(_s.DASHBOARD_HELP_LOCATIONS)

    places = get_places(
        tuple(data[_s.FIELD_ID].values.tolist()), get_session_geo_entity_types()
    )
    if places is None:
        st.warning("No places data found")
        return

    st.subheader("Local/national/global mentions")

    places = places.sort_values(by=_s.FEATURE_GEO_CATEGORY)
    with st.expander("View data", expanded=False):
        st.write(places)

    min_mentions = get_session_geo_min_mentions()
    places = places[places["count"] >= min_mentions]

    st.plotly_chart(
        vm.histogram(
            places,
            "count",
            _s.FEATURE_GEO_CATEGORY,
            _s.FEATURE_GEO_CATEGORY,
            labels={"count": "number of mentions"},
        ),
        use_container_width=True,
    )

    st.subheader("Places distribution")
    st.plotly_chart(
        vm.bar(
            places,
            "count",
            _s.FEATURE_GEO_PLACE,
            _s.FEATURE_GEO_CATEGORY,
            labels={"count": "number of mentions"},
        ),
        use_container_width=True,
    )

    st.subheader("Map")
    places = places[places[_s.FEATURE_GEO_CATEGORY] != "Local"]
    places = places.drop(columns=[_s.FEATURE_ENTITY_LABEL, _s.FEATURE_GEO_CATEGORY])
    places = places.groupby(places.columns[:-1].values.tolist()).sum().reset_index()
    places = places.sort_values(by="count", ascending=False)
    with st.expander("Map data", expanded=False):
        st.write(places)

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


def get_session_geo_entity_types() -> list[str]:
    return st.session_state.entity_types_geo


def get_session_geo_min_mentions() -> int:
    return st.session_state.geo_min_mentions


@st.experimental_memo
def get_places(
    ids: tuple[str], entity_types: list[str], section: Optional[str] = None
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
            if entity_types:
                places = places[places[_s.FEATURE_ENTITY_LABEL].isin(entity_types)]
            columns = [
                _s.FIELD_ID,
                _s.FEATURE_ENTITY_LABEL,
                _s.FEATURE_GEO_CATEGORY,
                _s.FEATURE_GEO_PLACE,
                _s.FEATURE_GEO_PLACE_LAT,
                _s.FEATURE_GEO_PLACE_LON,
            ]

            places = places[columns]
            places = places.drop_duplicates()
            places = places.drop(columns=[_s.FIELD_ID])
            places["count"] = 0
            return (
                places.groupby(places.columns[:-1].values.tolist())
                .count()
                .reset_index()
            )

    return None


if __name__ == "__main__":
    streamlit()
