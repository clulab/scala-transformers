import sys

sys.stdin.reconfigure(encoding="utf-8") # just in case!

from transformers import AutoTokenizer

name = sys.argv[1]
add_prefix_space = sys.argv[2] == "true"
use_fast = sys.argv[3] == "true"

# name = "microsoft/deberta-v3-base"
# add_prefix_space = True
# use_fast = True
#
# print(name)
# print(add_prefix_space)
# print(use_fast)

tokenizer = AutoTokenizer.from_pretrained(name, add_prefix_space=add_prefix_space, use_fast=use_fast)

def printTokenization(tokenization):
  input_ids = tokenization["input_ids"]
  word_ids = [word_id if word_id != None else -1 for word_id in tokenization.word_ids()] if use_fast else [-1 for _ in input_ids]
  tokens = tokenizer.convert_ids_to_tokens(input_ids)
  print("4")
  print(" ".join([str(input_id) for input_id in input_ids]))
  print(" ".join([str(word_id) for word_id in word_ids]))
  print(" ".join(tokens))
  print("", flush=True)

while (True):
  line = sys.stdin.readline().strip()
  words = line.split()
  tokenization = tokenizer(line)
  printTokenization(tokenization)
