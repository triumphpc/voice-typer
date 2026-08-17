# Builds and installs VoiceTyper.app. See README.md for details.

.PHONY: install run clean help

help:  ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## Build VoiceTyper.app into /Applications
	./build_app.sh

run:  ## Run from the terminal with console output (debug mode)
	./run.sh

clean:  ## Remove the virtualenv and caches
	rm -rf .venv __pycache__
