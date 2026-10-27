PY ?= python
test: ; PYTHONPATH=. $(PY) -m pytest tests -q
