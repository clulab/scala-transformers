# These are names that need to be coordinated between components.

__all__ = ["Names"]

class Names:
    ACCURACY = "accuracy"
    INPUT_IDS = "input_ids"
    LABELS = "labels"
    # for dependency parsing, this dataset column indicates where the positions of the heads are stored
    HEAD_POSITIONS = "head_positions"
    TASK_IDS = "task_ids"
    TOKENIZER_NAMES = [
        "bert-base-cased",
        "distilbert-base-cased",
        "roberta-base",
        "xlm-roberta-base",
        "google/bert_uncased_L-4_H-512_A-8",
        "google/electra-small-discriminator",
        "microsoft/deberta-v3-base"
    ]
