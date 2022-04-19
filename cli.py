import pickle
from enum import Enum

import typer

from refida import data as dm
from refida import etl as em
from refida import features
from settings import (
    DATA_DETAILS,
    DATA_DIR,
    DATA_RESEARCH,
    DATA_SOURCES,
    DATA_SUMMARY,
    DATA_TEXT,
    TOPIC_CLASSIFICATION_TOPICS,
    get_fields_of_research,
)

app = typer.Typer()


@app.command()
def etl(datadir: str = DATA_DIR.name):
    """
    Extract, transform and load data.

    :param datadir: Path to the data directory.
    """
    with typer.progressbar(length=2, label="ETL data...") as progress:
        files = dm.get_raw_data(datadir)
        progress.update(1)

        if not files:
            error("No PDF files found.")

        data = em.extract(files)
        data.to_csv(dm.get_etl_data_path(datadir), index=False)
        progress.update(1)


class TopicsSection(str, Enum):
    """
    Enum for the sections of the topics data.
    """

    text = DATA_TEXT
    research = DATA_RESEARCH


@app.command()
def topics(datadir: str = DATA_DIR.name, column: str = TopicsSection.text):
    """
    Apply topic classification to the data.

    :param datadir: Path to the data directory.
    :param column: Column to use for topic classification.
    """
    with typer.progressbar(length=2, label="Topic classification...") as progress:
        data = dm.get_etl_data(datadir)
        progress.update(1)

        if data is None:
            error("No data found. Run the `etl` command first.")

        labels = TOPIC_CLASSIFICATION_TOPICS
        if column == TopicsSection.research:
            labels = get_fields_of_research()

        topics = features.topic_classification(data, column, labels)
        topics.to_csv(dm.get_topics_data_path(column, datadir), index=False)

        progress.update(1)


def error(msg: str):
    """
    Print an error message and exit.

    :param msg: Error message.
    """
    typer.echo()
    typer.secho(f"Error: {msg}", fg=typer.colors.RED)
    raise typer.Abort()


@app.command()
def summaries(datadir: str = DATA_DIR.name):
    """
    Summarise the text of in the data.

    :param datadir: Path to the data directory.
    """
    with typer.progressbar(length=2, label="Summarising text...") as progress:
        data = dm.get_etl_data(datadir)
        progress.update(1)

        if data is None:
            error("No data found. Run the `etl` command first.")

        summaries = features.summarise(data)
        summaries.to_csv(dm.get_summaries_data_path(datadir), index=False)

        progress.update(1)


class EntitySection(str, Enum):
    """
    Enum for the sections of the entity data.
    """

    summary = DATA_SUMMARY
    details = DATA_DETAILS
    sources = DATA_SOURCES


@app.command()
def entities(
    datadir: str = DATA_DIR.name, column: EntitySection = EntitySection.summary
):
    """
    Extract entities from the data of the text of the given column.

    :param datadir: Path to the data directory.
    :param column: Name of the column to extract entities from.
    """
    with typer.progressbar(
        length=2, label=f"Extracting {column} entities..."
    ) as progress:
        data = dm.get_etl_data(datadir)
        progress.update(1)

        if data is None:
            error("No data found. Run the `etl` command first.")

        if column not in data.columns:
            error(f"Column {column} not found in data.")

        docs, entities = features.entity_extraction(data, column)
        entities.to_csv(dm.get_entities_data_path(column, datadir), index=False)

        with open(dm.get_spacy_docs_path(column, datadir), "wb") as f:
            pickle.dump(docs, f)

        progress.update(1)


@app.command()
def geolocate(
    datadir: str = DATA_DIR.name, column: EntitySection = EntitySection.summary
):
    """
    Geolocate the location entities in the data.

    :param datadir: Path to the data directory.
    :param column: Name of the column to geolocate entities from.
    """
    with typer.progressbar(
        length=2, label="Geolocating location entities..."
    ) as progress:
        data = dm.get_entities_data(column, datadir)
        progress.update(1)

        if data is None:
            error("No data found. Run the `entities` command first.")

        geolocations = features.geolocate(data)
        geolocations.to_csv(dm.get_geo_data_path(column, datadir), index=False)

        progress.update(1)


if __name__ == "__main__":
    app()
