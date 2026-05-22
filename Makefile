.PHONY: install run validate clean

install:
	pip install -r requirements.txt

run:
	python main.py

validate:
	python validate.py

pipeline: install run validate

clean:
	rm -f artifacts/chunks.json artifacts/retrieval.json artifacts/answers.json \
	      artifacts/eval.json artifacts/grounding_check.json artifacts/chunking_comparison.json
