# REF Impact Data Analysis

For the list of changes to the project see the [Changelog](CHANGELOG.md).

## Workflow

The nodes with round edges represent actions available as commands via the [cli](#run-the-cli).

```mermaid
flowchart LR
    data_raw[/Raw data/] --> etl(etl)
    data_raw -.- comment_data_raw[PDF files provided by the impact team]
    class comment_data_raw comment

    etl --> data_etl[/ETL data/]
    etl -.- comment_etl[Extract, transform and load the raw data into a data frame]
    class comment_etl comment

    data_etl -.- comment_data_etl[CSV with data extracted/transformed\nfrom the PDF files]
    data_etl --> topics(topics)
    class comment_data_etl comment

    data_impact_categories[/Impact categories/] --> topics(topics)
    data_impact_categories -.- comment_data_impact_categories[List of categories provided by the impact team]
    class comment_data_impact_categories comment

    topics --> data_topics[/Topics data/]
    topics -.- comment_topics[Topic classification using\nimpact categories as potential topics]
    class comment_topics comment

    data_topics -.- comment_data_topics[List of tuples with topic and confidence value]
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
