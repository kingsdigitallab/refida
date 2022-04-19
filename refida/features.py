from functools import lru_cache
from typing import Optional

import geopy
import pandas as pd
import spacy
from txtai.pipeline import Labels, Summary

from settings import (
    DATA_TEXT,
    FEATURE_ENTITY_ENTITY,
    FEATURE_ENTITY_LABEL,
    FEATURE_ENTITY_TEXT,
    FEATURE_LAT,
    FEATURE_LON,
    FEATURE_SUMMARY,
    FEATURE_TOPIC_SCORE,
    FEATURE_TOPIC_TOPIC,
    FIELD_ID,
    SPACY_ENTITY_TYPES,
    SPACY_LANGUAGE_MODEL,
    SPACY_LOCATION_ENTITY_TYPES,
    SUMMARISATION_MODEL,
    TOPIC_CLASSIFICATION_MODEL,
    TOPIC_CLASSIFICATION_TOPICS,
    geocode,
)


def topic_classification(
    data: pd.DataFrame,
    column: str,
    topics: list[str] = TOPIC_CLASSIFICATION_TOPICS,
    model: str = TOPIC_CLASSIFICATION_MODEL,
) -> pd.DataFrame:
    """
    Topic classification using txtai.Labels.

    :param data: DataFrame with text to classify.
    :param column: Column with text to classify.
    :param model: Model to use.
    :param topics: Topics to classify.
    """
    classifier = Labels(model)

    topics_df = data[[FIELD_ID, column]].copy()
    topics_df = topics_df.dropna(subset=[column])

    topics_df["topics"] = classifier(
        topics_df[column].values.tolist(), topics, multilabel=True
    )
    topics_df["topics"] = topics_df["topics"].apply(
        lambda predictions: [[topics[p[0]], p[1]] for p in predictions]
    )
    topics_df = topics_df.explode("topics")
    topics_df[[FEATURE_TOPIC_TOPIC, FEATURE_TOPIC_SCORE]] = pd.DataFrame(
        topics_df["topics"].tolist(), index=topics_df.index
    )
    topics_df = topics_df.drop(columns=[column, "topics"])

    return topics_df


def summarise(data: pd.DataFrame, model: str = SUMMARISATION_MODEL) -> pd.DataFrame:
    """
    Summarise the text.

    :param data: DataFrame with text to summarise.
    :param name: The name of the summary model to use.
    """
    summary = Summary(model)

    summary_df = data[[FIELD_ID]].copy()
    summary_df[FEATURE_SUMMARY] = summary(data[DATA_TEXT].values.tolist())

    return summary_df


def entity_extraction(
    data: pd.DataFrame,
    column: str,
    model: str = SPACY_LANGUAGE_MODEL,
    entity_types: list[str] = SPACY_ENTITY_TYPES,
) -> tuple[list, pd.DataFrame]:
    """
    Extract entities using spaCy.

    :param data: DataFrame with text to extract entities from.
    :param column: Column to extract entities from.
    :param model: spaCy language model to use.
    :param entity_types: Entity types to extract.
    """
    if column not in data.columns:
        raise ValueError(f"Column {column} not in data.")

    nlp = spacy.load(model)

    entities_df = data[[FIELD_ID]].copy()
    entities_df["doc"] = data[column].fillna("").apply(nlp)

    docs = entities_df["doc"].tolist()

    entities_df["entities"] = entities_df["doc"].apply(
        lambda doc: [
            [ent.label_, ent.text, f"{ent.label_}: {ent.text}"]
            for ent in doc.ents
            if ent.label_ in entity_types
        ]
    )
    entities_df = entities_df.drop(columns=["doc"])
    entities_df = entities_df.explode("entities")
    entities_df = entities_df.dropna(subset=["entities"])
    entities_df[
        [FEATURE_ENTITY_LABEL, FEATURE_ENTITY_TEXT, FEATURE_ENTITY_ENTITY]
    ] = pd.DataFrame(entities_df["entities"].tolist(), index=entities_df.index)
    entities_df = entities_df.drop(columns=["entities"])

    return docs, entities_df


def geolocate(
    data: pd.DataFrame,
    entity_types: list[str] = SPACY_LOCATION_ENTITY_TYPES,
) -> pd.DataFrame:
    """
    Geolocate data using OpenStreetMap Nominatim service.

    :param data: DataFrame with text to geocode.
    :param entity_types: Entity types to geocode.
    """
    geo_df = data[data[FEATURE_ENTITY_LABEL].isin(entity_types)].copy()
    geo_df[[FEATURE_LAT, FEATURE_LON]] = geo_df.apply(
        lambda x: get_coordinates(x[FEATURE_ENTITY_TEXT]), axis=1, result_type="expand"
    )
    geo_df = geo_df.dropna(subset=[FEATURE_LAT, FEATURE_LON])

    return geo_df


@lru_cache(maxsize=1024)
def get_coordinates(name: str) -> Optional[list[float, float]]:
    """
    Geolocate a place name using OpenStreetMap Nominatim service.

    :param name: The name of the location to geolocate.
    """
    try:
        location = geocode(name)
        if location:
            return [location.latitude, location.longitude]
    except (geopy.exc.GeocoderTimedOut, geopy.exc.GeocoderServiceError):
        return None
    return None
