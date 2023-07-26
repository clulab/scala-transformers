# These are names that need to be coordinated between components.

class Names:
    def __init__(self) -> None:
        self.ACCURACY = "accuracy"
        self.INPUT_IDS = "input_ids"
        self.LABELS = "labels"
        # for dependency parsing, this dataset column indicates where the positions of the heads are stored
        self.HEAD_POSITIONS = "head_positions"
        self.TASK_IDS = "task_ids"
        self.tokenizer_names = [
            "bert-base-cased",
            "distilbert-base-cased",
            "roberta-base",
            "xlm-roberta-base",
            "google/bert_uncased_L-4_H-512_A-8",
            "google/electra-small-discriminator",
            "microsoft/deberta-v3-base"
        ]

names = Names()
