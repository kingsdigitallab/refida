# REF 2021 impact data analysis

[![Python application](https://github.com/kingsdigitallab/refida/actions/workflows/test.yml/badge.svg)](https://github.com/kingsdigitallab/refida/actions/workflows/test.yml)

For the list of changes to the project see the [Changelog](CHANGELOG.md).

The REF 2021 Impact Data Analysis was a small project, between
[King's Digital Lab (KDL)](https://kdl.kcl.ac.uk/) and the
[King's College London (KCL)](https://kcl.ac.uk/) impact team, to produce analytical
work of the college's [REF 2021](https://results2021.ref.ac.uk/) impact case studies and
environment statements.

The data includes 158 impact case studies and environment statements, in PDF
(5-10 pages of text each), which follow standard templates but are expressed
with heterogenous descriptions and language.

The project was set up to help the impact team's address the questions:

> - What are the main types of impact KCL has delivered? Which pathways have been used
>   to deliver those impacts?
> - Who are our key partners and beneficiaries of our impacts?
> - Where are they - local (London), National or Global?
> - Is there a correlation between discipline and types of impact or pathways to impact
>   used?
> - What are the areas identified as strengths, areas for development and future plans?

## Architecture

![Architecture](docs/architecture.jpg)

The project has two main components, a Python command line tool to do the data
processing and to run the machine learning processes, and a web-dashboard to present
the results of the data processing.

## Workflow

![Workflow](docs/workflow.jpg)

- The process starts with extracting data from relevant sections of the documents into
  a single CSV file, which is then used by the different machine learning processes;
- [Zero shot](https://en.wikipedia.org/wiki/Zero-shot_learning) topic classification
  is applied to extract impact categories, fields of research and impact outputs;
- A [transformers](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>)-based
  language model is used to extract entities (mainly organisations and locations) from
  the data. The location data is further enriched by applying geocoding to gather
  coordinates and place geometries;
- Abstractive text summarisation is used to create summaries of the documents;
- And an indexing process indexes all the text both to perform keyword and semantic
  searches.

## Dashboard

The results of the machine learning processes can be explored and visualised via a
[web-based dashboard](docs/dashboard.png). Via the dashboard it is also possible to
search and filter the data to get more specific insights.
The [landing page](docs/dashboard.png) of the dashboard displays a table with all the
data and also overview information about the data.

## Set up

Install [poetry](https://python-poetry.org/docs/#installation) and the requirements:

    poetry install

Configure the settings by editing the file `settings.py`.

## Run the cli

    poetry run python cli.py

### Cli workflow

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

## Run the dashboard

To run the dashboard directly on the system:

    poetry run streamlit run streamlit_app.py

To run the dashboard with [Docker](https://www.docker.com/), first copy the
`docker-compose.override.yaml.example` into `docker-compose.override.yaml` and edit as
needed. The dashboard can then be run with the command:

    scripts/docker.sh

## Development

    poetry install --dev
    poetry shell
