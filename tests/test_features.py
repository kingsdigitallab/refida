import pandas as pd
import pytest

from refida import features
from settings import TOPIC_CLASSIFICATION_AREAS


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        data=dict(
            id=[1, 2, 3],
            text=[
                "That which does not kill us makes us stronger.",
                "The weak can never forgive. Forgiveness is the attribute of the strong.",  # noqa
                "In the depth of winter, I finally learned that within me there lay an invincible summer.",  # noqa
            ],
        )
    )


def test_topic_classification(data):
    topics = features.topic_classification(
        data, "text", topics=TOPIC_CLASSIFICATION_AREAS
    )
    assert len(topics) == len(data) * len(TOPIC_CLASSIFICATION_AREAS)


def test_topic_classification_sentences(data):
    number_of_sentences = 4
    topics = features.topic_classification(
        data, "text", topics=TOPIC_CLASSIFICATION_AREAS, sentences=True
    )

    assert len(topics) == number_of_sentences * len(TOPIC_CLASSIFICATION_AREAS)


def test_topic_classification_threshold(data):
    topics = features.topic_classification(
        data, "text", topics=TOPIC_CLASSIFICATION_AREAS, threshold=0.5
    )
    assert len(topics) < len(data) * len(TOPIC_CLASSIFICATION_AREAS)
