import torch

from names import names
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from typing import Any, Dict, List

# A custom data collator that creates correct batches for all tasks by including the names.HEAD_POSITIONS column as well
class DualDataCollator(DataCollatorForTokenClassification):
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        super().__init__(tokenizer)

    def make_head_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = [{names.INPUT_IDS: feature[names.HEAD_POSITIONS]} for feature in features]
        return result
        # return [{names.INPUT_IDS: feature[names.HEAD_POSITIONS]} for feature in features]

    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        label_name: str = "label" if "label" in features[0].keys() else names.LABELS
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
        #print(f"batched head positions: {batch_heads[names.INPUT_IDS]}")
        batch[names.HEAD_POSITIONS] = batch_heads[names.INPUT_IDS]

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch[names.INPUT_IDS]).shape[1]
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
