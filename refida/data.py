from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


def get_raw_data(datadir: str) -> Iterator[Path]:
    return get_raw_data_path(datadir).glob("**/*.pdf")


def get_raw_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "0_raw")


def get_data_path(
    dir: str,
    stage: Optional[str] = None,
    filename: Optional[str] = None,
    file_exists: bool = False,
) -> Path:
    path = Path(dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {path}")

    if not stage:
        return path

    path = path.joinpath(Path(stage))
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not filename:
        return path

    path = path.joinpath(Path(filename))
    if file_exists and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def get_etl_data(datadir: str) -> Optional[pd.DataFrame]:
    return get_data(get_etl_data_path(datadir))


def get_data(filename: Path, kwargs: dict = {}) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(filename, **kwargs)  # type: ignore
    except FileNotFoundError:
        return None


def get_etl_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "etl.csv")


def get_topics_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "topics.csv")


def get_summaries_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "summaries.csv")
