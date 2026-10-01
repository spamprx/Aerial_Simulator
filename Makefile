PYTHON ?= python
.PHONY: test slow bench figures dashboard
test:
	$(PYTHON) -m pytest
slow:
	$(PYTHON) -m pytest -m slow
bench:
	$(PYTHON) -m pytest -m bench
figures:
	@for f in scripts/fig_*.py; do echo $$f; $(PYTHON) $$f $(if $(QUICK),--quick) || exit 1; done
dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py
