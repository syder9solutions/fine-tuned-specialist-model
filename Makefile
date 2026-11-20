.PHONY: data train eval gate test serve clean

PY = .venv/bin/python
export PYTHONPATH = .

data:
	$(PY) cli.py data

train:
	$(PY) cli.py train --run-name lora-r16

eval:
	$(PY) cli.py eval --run-name lora-r16

gate:
	$(PY) scripts/ci_gate.py --run-name lora-r16

test:
	$(PY) -m pytest tests/ -q

serve:
	$(PY) cli.py serve --run-name lora-r16 --port 8098

clean:
	rm -rf runs reports data/train.jsonl data/test.jsonl data/curation_report.json
