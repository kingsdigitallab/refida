import pandas as pd
import spacy
from txtai.pipeline import Labels, Summary

from settings import (
    SPACY_ENTITY_TYPES,
    SPACY_LANGUAGE_MODEL,
    SUMMARISATION_MODEL,
    TOPIC_CLASSIFICATION_MODEL,
    TOPIC_CLASSIFICATION_TOPICS,
)


def topic_classification(
    data: pd.DataFrame,
    model: str = TOPIC_CLASSIFICATION_MODEL,
    topics: list[str] = TOPIC_CLASSIFICATION_TOPICS,
) -> pd.DataFrame:
    """
    Topic classification using txtai.Labels.

    :param data: DataFrame with text to classify.
    :param model: Model to use.
    :param topics: Topics to classify.
    """
    classifier = Labels(model)

    topics_df = data[["id"]].copy()
    topics_df["topics"] = classifier(data["text"].values.tolist(), topics)
    topics_df["topics"] = topics_df["topics"].apply(
        lambda predictions: [[topics[p[0]], p[1]] for p in predictions]
    )
    topics_df = topics_df.explode("topics")
    topics_df[["topic", "score"]] = pd.DataFrame(
        topics_df["topics"].tolist(), index=topics_df.index
    )
    topics_df = topics_df.drop(columns=["topics"])

    return topics_df


def summarise(data: pd.DataFrame, model: str = SUMMARISATION_MODEL) -> pd.DataFrame:
    """
    Summarise the text.

    :param data: DataFrame with text to summarise.
    :param name: The name of the summary model to use.
    """
    summary = Summary(model)

    summary_df = data[["id"]].copy()
    summary_df["summary"] = summary(data["text"].values.tolist())

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

    entities_df = data[["id"]].copy()
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
    entities_df[["label", "text", "entity"]] = pd.DataFrame(
        entities_df["entities"].tolist(), index=entities_df.index
    )
    entities_df = entities_df.drop(columns=["entities"])

    return docs, entities_df
