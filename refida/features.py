import pandas as pd
from txtai.pipeline import Labels

from settings import TOPIC_CLASSIFICATION_MODEL, TOPIC_CLASSIFICATION_TOPICS


def topics_classification(
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
    topics_df["topics"] = data["text"].apply(classify, args=(classifier, topics))

    return topics_df


def classify(
    text: str, classifier: Labels, topics: list[str], workers: int = 4
) -> list[tuple[str, float]]:
    """
    Get topics from text.

    :param text: Text to classify.
    :param classifier: Topic classifier to use.
    :param topics: Topics to classify.
    :param workers: Number of workers to use.
    """
    predictions = classifier(text, topics, workers=workers)
    return [(topics[p[0]], p[1]) for p in predictions]
