# Narrative Atoms

For the list of changes to the project see the [Changelog](CHANGELOG.md).

## Workflow

```mermaid
flowchart LR
    data_raw[/Raw data/] --> etl(Extract/transform/load)
    data_raw -.- comment_data_raw[PDF files provided by the impact team]
    class comment_data_raw comment

    etl --> data_extracted[/Extracted data/]

    data_extracted -.- comment_data_extracted[CSV with data extracted from the PDF files]
    class comment_data_extracted comment

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
