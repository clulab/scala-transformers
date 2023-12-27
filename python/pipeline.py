from transformers import Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForTokenClassification
from torch import Tensor

# see https://huggingface.co/docs/transformers/add_new_pipeline

# import os
# model_dir = os.path.expanduser("~/repos/clu-ling/clu-processors/avg")

class CluPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class

PIPELINE_REGISTRY.register_pipeline(
    "annotate",
    pipeline_class=CluPipeline,
    # FIXME: is this the right PT model for a cascade of token-based tasks?
    pt_model=AutoModelForTokenClassification,
)