from processors.tokenizers import CluTokenizer
from processors.trainers import CluTrainer
from processors.core import ShortTaskDef, Task
import pathlib

def test_true() -> None:
    assert True == True

def test_false() -> None:
    assert False != True

def test_clu_trainer() -> None:
    tokenizer = CluTokenizer.from_pretrained(use_fast=True)
    tasks = Task.mk_tasks(
      (pathlib.Path(__file__).parents[0] / "data").resolve(), 
      tokenizer, 
      [
        ShortTaskDef(
          task_name="NER",
          local_base_dir="conll-ner", 
          train_file_name="train_sample.txt",    
          dev_file_name="train_sample.txt",  
          test_file_name="train_sample.txt"
        ),
        ShortTaskDef(
          task_name="POS",
          local_base_dir="pos", 
          train_file_name="train_sample.txt",
          dev_file_name="train_sample.txt",
          test_file_name="train_sample.txt"
        ),
        ShortTaskDef(
          task_name="Chunking",
          local_base_dir="chunking",
          train_file_name="train_sample.txt",
          dev_file_name="test_sample.txt",
          test_file_name="test_sample.txt"
        ),
        # ShortTaskDef("Deps Head",  "deps-wsj/", "train_small.heads",  "dev.heads",        "test.heads"),
        # ShortTaskDef("Deps Label", "deps-wsj/", "train_small.labels", "dev.labels",       "test.labels", dual_mode = True)
      ]
    )
    # FIXME: this test should assert no exception is thrown by the call to .train()
    CluTrainer(tokenizer).train(tasks, epochs=1)
    assert 42 == 42


if __name__ == "__main__":
    test_clu_trainer()
