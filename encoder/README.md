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
