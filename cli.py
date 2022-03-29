import typer

import settings
from refida import data as dm
from refida import etl as em
from refida import features

app = typer.Typer()


@app.command()
def etl(datadir: str = settings.DATA_DIR.name):
    """
    Extract, transform and load data.

    :param datadir: Path to the data directory.
    """
    with typer.progressbar(length=2, label="ETL data...") as progress:
        files = dm.get_raw_data(datadir)
        progress.update(1)

        data = em.extract(files)
        data.to_csv(dm.get_etl_data_path(datadir), index=False)
        progress.update(1)


@app.command()
def topics(datadir: str = settings.DATA_DIR.name):
    """
    Apply topic classification to the data.

    :param datadir: Path to the data directory.
    """
    with typer.progressbar(length=2, label="Topic classification...") as progress:
        data = dm.get_etl_data(datadir)
        progress.update(1)

        if data is None:
            error("No data found. Run the `etl` command first.")

        topics = features.topics_classification(data)
        topics.to_csv(dm.get_topics_data_path(datadir), index=False)

        progress.update(1)


def error(msg: str):
    typer.echo()
    typer.secho(f"Error: {msg}", fg=typer.colors.RED)
    raise typer.Abort()
