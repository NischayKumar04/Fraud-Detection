PYTHON ?= python

.PHONY: help install preprocess train train-cost-10 train-cost-20 train-cost-25 evaluate explain app clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make preprocess     - Run preprocessing"
	@echo "  make train          - Train all models (default cost mode FN=25 FP=1)"
	@echo "  make train-cost-10  - Train with FN=10 FP=1"
	@echo "  make train-cost-20  - Train with FN=20 FP=1"
	@echo "  make train-cost-25  - Train with FN=25 FP=1"
	@echo "  make evaluate       - Evaluate best model"
	@echo "  make explain        - Run SHAP explainability"
	@echo "  make app            - Launch Streamlit app"
	@echo "  make clean          - Remove Python cache files"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) -m src.preprocess

train:
	$(PYTHON) -m src.train --model all --max_rows 0 --threshold_mode cost --fn_cost 25 --fp_cost 1

train-cost-10:
	$(PYTHON) -m src.train --model all --max_rows 0 --threshold_mode cost --fn_cost 10 --fp_cost 1

train-cost-20:
	$(PYTHON) -m src.train --model all --max_rows 0 --threshold_mode cost --fn_cost 20 --fp_cost 1

train-cost-25:
	$(PYTHON) -m src.train --model all --max_rows 0 --threshold_mode cost --fn_cost 25 --fp_cost 1

evaluate:
	$(PYTHON) -m src.evaluate

explain:
	$(PYTHON) -m src.explain --sample_size 12000 --top_n 15

app:
	streamlit run app/streamlit_app.py

clean:
	@echo "Cleaning __pycache__ and .pyc files..."
	-find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	-find . -type f -name "*.pyc" -delete 2>/dev/null || true