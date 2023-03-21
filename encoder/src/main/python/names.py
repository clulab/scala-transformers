# These are names that need to be coordinated between components.

class Names:
    def __init__(self) -> None:
        self.INPUT_IDS = "input_ids"
        # for dependency parsing, this dataset column indicates where the positions of the heads are stored
        self.HEAD_POSITIONS = "head_positions"

names = Names()
