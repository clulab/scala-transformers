# encoder

The encoder is dependent on a large number of libraries, which probably should not be installed into the global Python environment of your computer.  Instead, use conda or venv to start a local environment for the libraries.

```sh
conda create --name env
conda activate env
```

or

```sh
/bin/python3.9 -m venv env
source env/bin/activate
```

For the Python part of this encoder subproject, a `requirements.txt` file is provided.  Run something like
```sh
pip install -r requirements.txt
```
from the subproject directory to ensure that you have all the necessary Python modules installed and that their versions match expectations.

If you add a library, perform
```sh
pip freeze > requirements.txt
```

To run the tests, use
```sh
pytest
```
