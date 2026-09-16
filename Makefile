PY = python3

UV = uv

MAIN = main.py

FLAGS = --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

run :
	$(UV) run $(PY) -m src

install :
	-pip install uv
	@$(UV) sync

debug :
	@$(UV) run $(PY) -m pdb $(MAIN)

clean :
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint :
	@$(UV) run flake8 .
	@$(UV) run mypy . $(FLAGS)
