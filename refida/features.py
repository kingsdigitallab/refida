from functools import lru_cache
from typing import Optional

import geojson
import geopy
import pandas as pd
import spacy
from geopy.location import Location
from txtai.pipeline import Labels, Summary

import settings as _s


def topic_classification(
    data: pd.DataFrame,
    column: str,
    topics: list[str] = _s.TOPIC_CLASSIFICATION_TOPICS,
    model: str = _s.TOPIC_CLASSIFICATION_MODEL,
) -> pd.DataFrame:
    """
    Topic classification using txtai.Labels.

    :param data: DataFrame with text to classify.
    :param column: Column with text to classify.
    :param model: Model to use.
    :param topics: Topics to classify.
    """
    classifier = Labels(model)

    topics_df = data[[_s.FIELD_ID, column]].copy()
    topics_df = topics_df.dropna(subset=[column])

    topics_df["topics"] = classifier(
        topics_df[column].values.tolist(), topics, multilabel=True
    )
    topics_df["topics"] = topics_df["topics"].apply(
        lambda predictions: [[topics[p[0]], p[1]] for p in predictions]
    )
    topics_df = topics_df.explode("topics")
    topics_df[[_s.FEATURE_TOPIC_TOPIC, _s.FEATURE_TOPIC_SCORE]] = pd.DataFrame(
        topics_df["topics"].tolist(), index=topics_df.index
    )
    topics_df = topics_df.drop(columns=[column, "topics"])

    return topics_df


def summarise(data: pd.DataFrame, model: str = _s.SUMMARISATION_MODEL) -> pd.DataFrame:
    """
    Summarise the text.

    :param data: DataFrame with text to summarise.
    :param name: The name of the summary model to use.
    """
    summary = Summary(model)

    summary_df = data[[_s.FIELD_ID]].copy()
    summary_df[_s.FEATURE_SUMMARY] = summary(data[_s.DATA_TEXT].values.tolist())

    return summary_df


def entity_extraction(
    data: pd.DataFrame,
    column: str,
    model: str = _s.SPACY_LANGUAGE_MODEL,
    entity_types: list[str] = _s.SPACY_ENTITY_TYPES,
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

    entities_df = data[[_s.FIELD_ID]].copy()
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
        [_s.FEATURE_ENTITY_LABEL, _s.FEATURE_ENTITY_TEXT, _s.FEATURE_ENTITY_ENTITY]
    ] = pd.DataFrame(entities_df["entities"].tolist(), index=entities_df.index)
    entities_df = entities_df.drop(columns=["entities"])

    return docs, entities_df


def geolocate(
    data: pd.DataFrame,
    entity_types: list[str] = _s.SPACY_LOCATION_ENTITY_TYPES,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Geolocate data using OpenStreetMap Nominatim service.

    :param data: DataFrame with text to geocode.
    :param entity_types: Entity types to geocode.
    """
    place_data_columns = [
        _s.FEATURE_GEO_DISPLAY_NAME,
        _s.FEATURE_GEO_LAT,
        _s.FEATURE_GEO_LON,
        _s.FEATURE_GEO_CATEGORY,
        _s.FEATURE_GEO_PLACE,
        _s.FEATURE_GEO_PLACE_LAT,
        _s.FEATURE_GEO_PLACE_LON,
        _s.FEATURE_GEO_GEOJSON,
    ]

    geo_df = data[data[_s.FEATURE_ENTITY_LABEL].isin(entity_types)].copy()
    geo_df[place_data_columns] = geo_df.apply(
        lambda x: get_place_data(x[_s.FEATURE_ENTITY_LABEL], x[_s.FEATURE_ENTITY_TEXT]),
        axis=1,
        result_type="expand",
    )
    geo_df = geo_df.dropna(subset=[_s.FEATURE_GEO_LAT, _s.FEATURE_GEO_LON])
    geo_df[_s.FEATURE_GEO_GEOJSON] = geo_df.apply(
        lambda x: geojson.Feature(
            id=x[_s.FEATURE_ENTITY_TEXT],
            geometry=x[_s.FEATURE_GEO_GEOJSON],
            properties={"name": x[_s.FEATURE_ENTITY_TEXT]},
        ),
        axis=1,
    )

    geojson_features = geo_df[_s.FEATURE_GEO_GEOJSON].drop_duplicates().values.tolist()

    geo_df = geo_df.drop(columns=[_s.FEATURE_GEO_GEOJSON])
    geo_df = geo_df.drop_duplicates()

    return geo_df, geojson_features


@lru_cache
def get_place_data(
    label: str,
    name: str,
) -> Optional[
    tuple[
        str,
        float,
        float,
        Optional[str],
        Optional[str],
        Optional[float],
        Optional[float],
        dict,
    ]
]:
    """
    Get place data using OpenStreetMap Nominatim service.

    :param label: Entity label.
    :param name: Name of the place to get data for.
    """
    if not name:
        return None

    location = geocode(name)
    if not location:
        return None

    raw = location.raw

    city = name
    place = display_name = raw["display_name"]
    place_location = None

    # continents
    if label == "LOC":
        place_location = location

    if "address" in raw:
        if "city" in raw["address"]:
            city = raw["address"]["city"]
        if "country" in raw["address"]:
            place = raw["address"]["country"]

            place_location = geocode(place)

    return (
        display_name,
        location.latitude,
        location.longitude,
        _s.get_place_category(city, place),
        place,
        place_location.latitude if place_location else None,
        place_location.longitude if place_location else None,
        location.raw["geojson"],
    )


@_s.memory.cache
def geocode(name: str) -> Optional[Location]:
    """
    Geolocate a place name using OpenStreetMap Nominatim service.

    :param name: The name of the location to geolocate.
    """
    try:
        return _s.geolocator(name, language="en", addressdetails=1, geometry="geojson")
    except (geopy.exc.GeocoderTimedOut, geopy.exc.GeocoderServiceError):
        return None
