import torch

from parameters import parameters
from transformers import DataCollatorForTokenClassification

# A custom data collator that creates correct batches for all tasks by including the cf.HEAD_POSITIONS column as well
class DualDataCollator(DataCollatorForTokenClassification):
    def __init__(self) -> None:
        self.input_ids = "input_ids"

    def make_head_features(self, features) -> list[dict[str, "feature"]]:
        return [{self.input_ids: feature[parameters.HEAD_POSITIONS]} for feature in features]

    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None
        )
        # HF does not batch head_positions, so we have to fake it by masquerading them as input_ids
        heads = self.make_head_features(features)
        batch_heads = self.tokenizer.pad(
            heads,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None
        )
        #print(f"batched head positions: {batch_heads[input_ids]}")
        batch[parameters.HEAD_POSITIONS] = batch_heads[self.input_ids]

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch[self.input_ids]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        return batch
