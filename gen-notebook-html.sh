#!/bin/bash
source .venv/bin/activate
ipynb-py-convert eda.py artifacts/eda.ipynb
papermill artifacts/eda.ipynb artifacts/eda.ipynb
jupyter nbconvert artifacts/eda.ipynb --to=html --TemplateExporter.exclude_input=True
