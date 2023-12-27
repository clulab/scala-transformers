```python
conda create -y -n "cluproc" python=3.11 ipython
conda activate cluproc
pip install -e ".[all]"
```


```python
from processors.tokenizers import CluTokenizer

tokenizer = CluTokenizer.from_pretrained()
toks = tokenizer.tokenizer("My name is Inigo Montoya.")
```