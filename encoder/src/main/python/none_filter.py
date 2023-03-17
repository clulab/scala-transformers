
class NoneFilter():
    def __init__(self, key) -> None:
        self.key = key

    def at(self, collection):
        None if collection is None else collection[self.key]
