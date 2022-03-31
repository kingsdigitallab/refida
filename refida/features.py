import pandas as pd
from txtai.pipeline import Labels, Summary

from settings import (
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


def summarise(data: pd.DataFrame, model: str = SUMMARISATION_MODEL) -> str:
    """
    Summarise the text.

    :param data: DataFrame with text to summarise.
    :param name: The name of the summary model to use.
    """
    summary = Summary(model)

    summary_df = data[["id"]].copy()
    summary_df["summary"] = summary(data["text"].values.tolist())
    return summary_df
