import shap
from transformers import ZeroShotClassificationPipeline
from txtai.pipeline import Labels

import settings as _s
from settings import memory


class ExplainableZeroShotClassificationPipeline(ZeroShotClassificationPipeline):
    """
    Explainable ZeroShotClassificationPipeline to workaround shap not currently
    supporting zero shot classification because it doesn't pass the extra parameters
    (labels) needed by zero shot classification.

    https://stackoverflow.com/questions/69628487
    """

    def __call__(self, *args):
        """
        Call the ZeroShotClassificationPipeline with the workaround labels.
        """
        output = super().__call__(args[0], self.explain_labels, multi_label=True)[0]

        return [
            [
                {"label": x[0], "score": x[1]}
                for x in zip(output["labels"], output["scores"])
            ]
        ]

    def set_explain_labels(self, labels):
        self.explain_labels = labels


@memory.cache
def topic_classification(
    text: str,
    topics: list[str] = _s.TOPIC_CLASSIFICATION_TOPICS,
    model_name: str = _s.TOPIC_CLASSIFICATION_MODEL,
):
    classifier = Labels(model_name)

    model = classifier.pipeline.model
    tokenizer = classifier.pipeline.tokenizer

    model.config.label2id.update({v: k for k, v in enumerate(topics)})
    model.config.id2label.update({k: v for k, v in enumerate(topics)})

    pipeline = ExplainableZeroShotClassificationPipeline(
        model=model, tokenizer=tokenizer
    )
    pipeline.set_explain_labels(topics)

    explainer = shap.Explainer(pipeline)
    shap_values = explainer([text])

    return shap_values
