# REF Impact Data Analysis

For the list of changes to the project see the [Changelog](CHANGELOG.md).

## Workflow

The nodes with round edges represent actions available as commands via the [cli](#run-the-cli).

```mermaid
flowchart LR
    data_raw[/Raw data/] --> etl(etl)
    data_raw -.- comment_data_raw[PDF files provided by the impact team]
    class comment_data_raw comment

    comment_data_etl[CSV with data extracted/transformed\nfrom the PDF files] -.- data_etl
    class comment_data_etl comment

    etl --> data_etl[/ETL data/]
    etl -.- comment_etl[Extract, transform and load the raw data\ninto a data frame]
    class comment_etl comment

    data_etl --> entities(entities)
    data_etl --> summaries(summaries)
    data_etl --> topics(topics)

    entities --> data_entities[/Entities data/]
    entities --> data_doc_entities[/spaCy entities docs/]
    entities -.- comment_entities[Entity extraction, can be applied to the\nsummary, details and sources sections]
    class comment_entities comment

    data_entities -.- comment_data_entities[CSVs with the entities extracted for each section]
    data_entities --> geolocate(geolocate)
    class comment_data_entities comment

    data_doc_entities -.- comment_data_doc_entities[Serialized spaCy docs for reuse]
    class comment_data_doc_entities comment

    geolocate -.- comment_geolocate[Geolocation, using OpenStreetMap's Nominatin service,\n can be applied to the location entities]
    geolocate --> data_geolocate[/Location entities data/]
    geolocate --> data_geojson[/Location entities geometry data/]
    class comment_geolocate comment

    data_geolocate -.- comment_data_geolocate[CSVs with location entities with lat and lon coordinates]
    class comment_data_geolocate comment
    data_geojson -.- comment_data_geojson[GeoJSON files with geometry data for location entities]
    class comment_data_geojson comment

    summaries --> data_summaries[/Summarised data/]
    summaries -.- comment_summaries[Abstractive text summarisation]
    class comment_summaries comment

    data_summaries -.- comment_data_summaries[CSV with summaries of each text]
    class comment_data_summaries comment

    data_impact_categories[/Impact categories/] --> topics(topics)
    comment_data_impact_categories[List of categories provided by the impact team] -.- data_impact_categories
    class comment_data_impact_categories comment

    data_for[/Fields of research/] --> topics(topics)
    comment_data_for[ANZSRC authority list of fields of research] -.- data_for
    class comment_data_for comment

    topics --> data_topics[/Topics data/]
    topics -.- comment_topics[Topic classification using\nimpact categories or fields of research\n as potential topics]
    class comment_topics comment

    data_topics -.- comment_data_topics[CSVs with topic and confidence value]
    class comment_data_topics comment

    classDef comment fill:lightyellow,stroke-width:0px;
```

## Set up

Install [poetry](https://python-poetry.org/docs/#installation) and the requirements:

    poetry install

## Run the cli

    poetry run python cli.py

## Run the gui

    poetry run streamlit run streamlit_app.py

## Development

    poetry install --dev
    poetry shell
