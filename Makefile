PY ?= python
DEMO_OUT = outputs/demo

.PHONY: install demo full test clean

install:
	$(PY) -m pip install -e .

demo:
	$(PY) scripts/prepare_data.py --data synthetic --n-scenes 24
	$(PY) scripts/train.py --config configs/demo.yaml
	$(PY) scripts/evaluate.py --config configs/demo.yaml
	$(PY) scripts/make_tables.py --run-dir $(DEMO_OUT)
	$(PY) scripts/plot_fig1_architecture.py --out $(DEMO_OUT)/fig1_architecture.png
	$(PY) scripts/plot_fig2_attention.py --config configs/demo.yaml --out $(DEMO_OUT)/fig2_attention.png
	$(PY) scripts/plot_fig3_training.py --run-dir $(DEMO_OUT) --out $(DEMO_OUT)/fig3_training.png
	$(PY) scripts/plot_fig4_robustness.py --run-dir $(DEMO_OUT) --out $(DEMO_OUT)/fig4_robustness.png
	$(PY) scripts/plot_fig5_qualitative.py --config configs/demo.yaml --out $(DEMO_OUT)/fig5_qualitative.png

full:
	$(PY) scripts/prepare_data.py --data synthetic --n-scenes 100
	$(PY) scripts/train.py --config configs/main.yaml
	$(PY) scripts/evaluate.py --config configs/main.yaml
	$(PY) scripts/make_tables.py --run-dir outputs/main
	$(PY) scripts/plot_fig1_architecture.py --out outputs/main/fig1_architecture.png
	$(PY) scripts/plot_fig2_attention.py --config configs/main.yaml --out outputs/main/fig2_attention.png
	$(PY) scripts/plot_fig3_training.py --run-dir outputs/main --out outputs/main/fig3_training.png
	$(PY) scripts/plot_fig4_robustness.py --run-dir outputs/main --out outputs/main/fig4_robustness.png
	$(PY) scripts/plot_fig5_qualitative.py --config configs/main.yaml --out outputs/main/fig5_qualitative.png

test:
	$(PY) -m pytest -q

clean:
	rm -rf outputs/demo outputs/main outputs/real checkpoints
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
