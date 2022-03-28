import typer

import settings
from refida import data as dm
from refida import etl as em

app = typer.Typer()


@app.command()
def etl(datadir: str = settings.DATA_DIR.name):
    """
    Extract, transform and load data.

    :param datadir: Path to the data directory
    """
    with typer.progressbar(length=2, label="Preparing data...") as progress:
        files = dm.get_raw_data(datadir)
        progress.update(1)

        extracted = em.extract(files)
        extracted.to_csv(dm.get_extracted_data_path(datadir), index=False)
        progress.update(1)


if __name__ == "__main__":
    app()
