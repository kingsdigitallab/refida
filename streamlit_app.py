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
from refida.search_index import LexicalIndexDoc, SemIndexDoc

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

    st.markdown(_s.DASHBOARD_FOOTER, unsafe_allow_html=True)


def sidebar():
    st.title("Insights")

    st.session_state.view = st.radio(
        "Choose insights",
        (
            "About the data",
            "Impact categories",
            "Outputs",
            "Fields of research",
            "Partners",
            "Beneficiaries",
            "Locations",
            "Search results",
        ),
    )

    if (
        show_impact_categories_view()
        or show_outputs_view()
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
    return st.session_state.get("view", "About the data")


def show_impact_categories_view():
    return get_session_view() == "Impact categories"


def show_outputs_view():
    return get_session_view() == "Outputs"


def show_fields_of_research_view():
    return get_session_view() == "Fields of research"


def show_partners_view():
    return get_session_view() == "Partners"


def show_beneficiaries_view():
    return get_session_view() == "Beneficiaries"


def show_geo_view():
    return get_session_view() == "Locations"


def show_text_search_view():
    return get_session_view() == "Search results"


def topics_sidebar():
    view = get_session_view()

    st.subheader(f"{view} options")
    st.session_state.topics_aggr_function = st.radio(
        f"Aggregate {view.lower()} by",
        ("count", "mean"),
        help=_s.DASHBOARD_HELP_TOPICS_AGGR_FUNCTION,
    )


def entities_sidebar():
    view = get_session_view()
    st.subheader(f"{view} options")

    st.session_state.entity_types = st.multiselect(
        f"Choose {view.lower()} entity types",
        _s.SPACY_ENTITY_TYPES,
        default=_s.SPACY_ENTITY_TYPES,
    )


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

    with st.expander("Text search", expanded=bool(get_search_phrase())):
        st.session_state.search_phrase = st.text_input("Search phrase")
        st.session_state.is_search_semantic = st.checkbox(
            "Semantic search",
            True,
            help=_s.DASHBOARD_HELP_SEARCH_MODE,
        )
        st.session_state.search_limit = st.selectbox(
            "Maximum number of results",
            [10, 20, 50, 100, 500],
            _s.SEARCH_LIMIT_INDEX,
            help=_s.DASHBOARD_HELP_SEARCH_LIMIT,
        )

    with st.expander("Filter by panel", expanded=True):
        st.session_state.filter_panel = st.multiselect("Panel", _s.PANELS)

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

    outputs = sorted(_s.get_outputs()[1])
    with st.expander("Filter by outputs", expanded=True):
        st.session_state.filter_outputs = st.multiselect("Outputs", outputs)

    fields_of_research = sorted(_s.get_fields_of_research()[1])
    with st.expander("Filter by fields of research", expanded=True):
        st.session_state.filter_fields_of_research = st.multiselect(
            "Fields of research", fields_of_research
        )

    entities = get_entities([_s.DATA_SUMMARY, _s.DATA_DETAILS, _s.DATA_SOURCES])
    if entities is not None:
        entities = entities[_s.FEATURE_ENTITY_ENTITY]
        entities = entities.drop_duplicates()
    with st.expander("Filter by entities", expanded=True):
        st.session_state.filter_entities = st.multiselect("Entities", entities)


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
    panels = get_session_filter_panel()
    if panels:
        data = data[data[_s.DATA_PANEL].isin(panels)]

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

    outputs = get_topics(
        [_s.DATA_SUMMARY, _s.DATA_DETAILS],
        get_session_filter_topics_score_threshold(),
        topics=get_session_filter_outputs(),
    )
    if outputs is not None:
        data = data[data[_s.FIELD_ID].isin(outputs[_s.FIELD_ID])]

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

    entities = get_entities(
        [_s.DATA_SUMMARY, _s.DATA_DETAILS, _s.DATA_SOURCES],
        entities=get_session_filter_entities(),
    )
    if entities is not None:
        data = data[data[_s.FIELD_ID].isin(entities[_s.FIELD_ID])]

    text_search(data)
    data = filter_data_by_text_search(data)

    return data


def get_session_filter_panel() -> list[str]:
    return st.session_state.get("filter_panel", [])


def get_session_filter_uoa() -> list[str]:
    return st.session_state.get("filter_uoa", [])


def get_session_filter_topics_score_threshold() -> float:
    return st.session_state.get(
        "filter_topics_score_threshold", _s.DEFAULT_FILTER_TOPICS_SCORE_THRESHOLD
    )


def get_session_filter_impact_categories() -> list[str]:
    return st.session_state.get("filter_impact_categories", [])


def get_session_filter_outputs() -> list[str]:
    return st.session_state.get("filter_outputs", [])


def get_session_filter_fields_of_research() -> list[str]:
    return st.session_state.get("filter_fields_of_research", [])


def get_session_filter_entities() -> list[str]:
    return st.session_state.get("filter_entities", [])


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


def show_data(data: pd.DataFrame, selection: pd.DataFrame):  # noqa
    n_rows = selection.shape[0]

    if n_rows == 0:
        st.warning("No data found")
        return

    doc = None
    doc_idx = None

    if show_about_data_view():
        st.header("About the data")
        show_about_data(data)

    if n_rows == 1 and not show_text_search_view():
        doc = selection.iloc[0]
        doc_idx = data[data[_s.FIELD_ID] == selection[_s.FIELD_ID].iloc[0]].index[0]

        show_first_doc(selection)

    if show_impact_categories_view():
        show_topics("Impact categories", selection)
        return

    if show_outputs_view():
        show_topics("Outputs", selection, [_s.DATA_SUMMARY, _s.DATA_DETAILS])
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

    if show_text_search_view():
        show_search_results(data)
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
    view_and_download_data("Units of assessment", data)
    st.plotly_chart(
        vm.histogram(data, None, _s.DATA_UOA, _s.DATA_TYPE), use_container_width=True
    )

    st.subheader("Research/impact timeline")
    research[_s.DATA_RESEARCH_START] = pd.to_datetime(
        research[_s.DATA_RESEARCH_START].apply(lambda x: str(int(x))), yearfirst=True
    )
    research[_s.DATA_RESEARCH_END] = pd.to_datetime(
        research[_s.DATA_RESEARCH_END].apply(lambda x: f"{str(int(x))}-12"),
        yearfirst=True,
    )
    research["type"] = "Research period"
    impact = research[[_s.FIELD_ID, _s.DATA_IMPACT_START, _s.DATA_IMPACT_END]].copy()
    impact[_s.DATA_RESEARCH_START] = pd.to_datetime(
        impact[_s.DATA_IMPACT_START].apply(lambda x: str(int(x))), yearfirst=True
    )
    impact[_s.DATA_RESEARCH_END] = pd.to_datetime(
        impact[_s.DATA_IMPACT_END].apply(lambda x: f"{str(int(x))}-12"),
        yearfirst=True,
    )
    impact["type"] = "Impact period"
    research = pd.concat([research, impact])
    research["title"] = research[_s.FIELD_ID].apply(
        lambda x: " ".join(x.split("_")[3:])
    )
    research["start"] = research[_s.DATA_RESEARCH_START]
    research["end"] = research[_s.DATA_RESEARCH_END]
    research = research[[_s.FIELD_ID, "title", "start", "end", "type"]]
    view_and_download_data("Timeline", research)
    st.plotly_chart(
        px.timeline(
            research,
            x_start="start",
            x_end="end",
            y="title",
            color="type",
            opacity=0.7,
        ),
        use_container_width=True,
    )


def view_and_download_data(title: str, data: pd.DataFrame):
    if data is not None:
        with st.expander("View data", expanded=False):
            # https://stackoverflow.com/questions/69578431/how-to-fix-streamlitapiexception-expected-bytes-got-a-int-object-conver
            # https://issues.apache.org/jira/browse/ARROW-14087
            st.write(data.astype(str))
            st.download_button(
                label="Download data as CSV",
                data=convert_df(data),
                file_name=f"{title.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )


@st.experimental_memo
def convert_df(df):
    return df.to_csv().encode("utf-8")


def show_first_doc(data: pd.DataFrame):
    if len(data):
        doc = data.iloc[0]

        st.subheader(doc["title"])
        show_doc(doc)


def show_doc(doc: pd.Series, hide_summary=False):
    with st.expander("View document", expanded=False):
        with open(doc["file"], "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = (
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                'width="100%" height="500" type="application/pdf"></iframe>'
            )
            st.markdown(pdf_display, unsafe_allow_html=True)

    if not hide_summary:
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

    topics[_s.DATA_UOA] = topics.apply(
        lambda x: f"{str(x[_s.DATA_UOA_N]).zfill(2)}: {x[_s.DATA_UOA]}", axis=1
    )
    topics = topics.sort_values(
        [_s.FEATURE_TOPIC_GROUP, _s.FEATURE_TOPIC_TOPIC, _s.DATA_UOA], ascending=False
    )
    topics_aggr = (
        topics.groupby([_s.FEATURE_TOPIC_GROUP, _s.FEATURE_TOPIC_TOPIC])
        .agg(aggr_function)
        .reset_index()
    )

    st.subheader(f"{title} with confidence >= {threshold} aggregated by {aggr}")

    st.subheader(f"{title} distribution")
    view_and_download_data(
        f"{title} distribution",
        topics_aggr[
            [_s.FEATURE_TOPIC_GROUP, _s.FEATURE_TOPIC_TOPIC, _s.FEATURE_TOPIC_SCORE]
        ],
    )
    st.plotly_chart(
        vm.histogram(
            topics_aggr,
            _s.FEATURE_TOPIC_SCORE,
            _s.FEATURE_TOPIC_TOPIC,
            _s.FEATURE_TOPIC_GROUP,
        ),
        use_container_width=True,
    )

    colour_df = topics[_s.DATA_PANEL].copy()
    palette = px.colors.qualitative.Plotly
    palette_len = len(palette)
    colours = colour_df.map(
        {v: palette[i % palette_len] for i, v in enumerate(colour_df.unique())}
    )

    st.subheader(f"Connections between {title.lower()} and unit of assessment")
    view_and_download_data(f"{title} alluvial", topics)
    st.plotly_chart(
        vm.parallel_categories(
            topics, [_s.FEATURE_TOPIC_TOPIC, _s.DATA_PANEL, _s.DATA_UOA], colours
        ),
        use_container_width=True,
    )

    st.subheader(f"Correlation between {title.lower()} and unit of assessment")
    topics_aggr = (
        topics.groupby([_s.FEATURE_TOPIC_GROUP, _s.FEATURE_TOPIC_TOPIC, _s.DATA_UOA])
        .agg(aggr_function)
        .reset_index()
    )
    view_and_download_data(
        f"{title} correlation",
        topics_aggr[
            [
                _s.FEATURE_TOPIC_GROUP,
                _s.FEATURE_TOPIC_TOPIC,
                _s.DATA_UOA,
                _s.FEATURE_TOPIC_SCORE,
            ]
        ],
    )
    st.plotly_chart(
        vm.scatter(
            topics_aggr,
            _s.DATA_UOA,
            _s.FEATURE_TOPIC_TOPIC,
            _s.FEATURE_TOPIC_GROUP,
            _s.FEATURE_TOPIC_SCORE,
        ).update_layout(
            xaxis=dict(categoryorder="category ascending", tickangle=-45),
            # yaxis=dict(
            # categoryorder="category ascending",
            # tickmode="linear",
            # type="category",
            # ),
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
        if len(data):
            data = data[data[_s.FEATURE_TOPIC_SCORE] >= threshold]

            if topics:
                data = data[data[_s.FEATURE_TOPIC_TOPIC].isin(topics)]

            if ids:
                return get_rows_by_id(data, ids)

            return data

    return None


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
        view_and_download_data(title, entities)
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
    sections: list[str],
    ids: tuple[str] = None,
    entity_types: list[str] = None,
    entities: list[str] = None,
) -> Optional[pd.DataFrame]:
    data = pd.DataFrame()

    for section in sections:
        section_df = dm.get_entities_data(section)
        if section_df is not None:
            data = pd.concat([data, section_df], ignore_index=True)

    if data is not None:
        if entity_types:
            data = data[data[_s.FEATURE_ENTITY_LABEL].isin(entity_types)]

        if entities:
            data = data[data[_s.FEATURE_ENTITY_ENTITY].isin(entities)]

        if ids:
            data = get_rows_by_id(data, ids)

        return data[
            [_s.FIELD_ID, _s.FEATURE_ENTITY_LABEL, _s.FEATURE_ENTITY_ENTITY]
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

    places = places.sort_values(by=_s.FEATURE_GEO_CATEGORY)
    min_mentions = get_session_geo_min_mentions()
    places = places[places["count"] >= min_mentions]

    st.subheader("Local/national/global mentions")
    view_and_download_data("Places mentions", places)
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
    view_and_download_data("Places distribution", places)
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
    focus = places.iloc[0]
    view_and_download_data("Places map", places)
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


def text_search(data: pd.DataFrame):
    """Run a text search
    and sets st.session_state.search_hits = [
        {id:, text:, score: },
    ]
    or None if no search phrase provided by user
    """
    hits = None
    phrase = st.session_state.search_phrase.strip()

    if phrase:
        index = get_search_index()
        hits = index.search_phrase(phrase, limit=st.session_state.search_limit)

    st.session_state.search_hits = hits


def is_search_semantic():
    return st.session_state.is_search_semantic


def filter_data_by_text_search(data):
    hits = st.session_state.search_hits
    if hits is not None:
        data = data[data.id.isin([hit["id"] for hit in hits])]
    return data


def get_search_index():
    if is_search_semantic():
        ret = SemIndexDoc()
    else:
        ret = LexicalIndexDoc()
    ret.set_highlight_format(
        '<span style="background-color: rgb(255, 255, 128);">', "</span>"
    )
    return ret


def show_search_results(data: pd.DataFrame):
    # EXPLAIN_STRATEGY = temporary flag, for experimentation
    # 0: show summary, 1: use txtai explain, 2: use sentence similarity
    # 3: sentence similarity using semindex_sents
    # --
    # 1: is unacceptably slow without a GPU (~1min to explain just two paras)
    # 2: also quite slow (few seconds per result)
    # 3: fast but inexplicable results: some docs with no sentences
    # others not relevant

    hits = st.session_state.search_hits

    if hits is None:
        st.warning("Use the 'Text search' in the side bar to start a search.")
        return

    index = get_search_index()

    phrase = st.session_state.search_phrase.strip()

    st.header(f"Search results ({len(hits)})")

    multiple_terms_without_and = len(phrase.split()) > 1 and "OR" not in phrase
    if not is_search_semantic() and multiple_terms_without_and:
        st.info(
            "Tip: by default only documents that contain all the terms"
            " in your query will be returned by the lexical search."
            " `health OR medical` will return documents"
            " that contain any of those words. "
        )

    for hit_idx, hit in enumerate(hits):
        rows = data[data["id"] == hit["id"]]
        title = hit["id"]
        if len(rows):
            row = rows.iloc[0]
            title = row["title"]

        st.subheader(f"{hit_idx+1}. {title}")

        indicator_width = min(1.0, hit["score"]) * 100
        st.write(
            "<div style='border-bottom:1px solid black; "
            f"width:{indicator_width}%'></div>"
            "",
            unsafe_allow_html=True,
        )

        if len(rows):
            show_doc(row, True)

            message = ""

            explanation = index.get_highlighted_text_from_hit(hit, phrase)

            st.write(explanation, unsafe_allow_html=True)

        st.write(f"(score: {hit['score']:.2f}, id: {repr(hit['id'])}) {message}")


def get_search_phrase():
    ret = st.session_state.get("search_phrase", "").strip()
    return ret


if __name__ == "__main__":
    streamlit()
