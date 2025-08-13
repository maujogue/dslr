# Makefile for DSLR project

CONFIG_DIR := config
SRC_DIR := src
VENV := $(CONFIG_DIR)/.venv_dslr
VENV_ACTIVATE := $(VENV)/bin/activate
RM := rm -rf
PYTHON := python3
PIP := pip3

# Project variables
PROJECT_NAME := dslr
REQUIREMENTS_FILE := $(CONFIG_DIR)/requirements.txt

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@echo "=================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dependencies in virtual environment
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created successfully!"
	@echo "Installing dependencies..."
	. $(VENV_ACTIVATE) && $(PIP) install --upgrade pip
	. $(VENV_ACTIVATE) && $(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "Dependencies installed successfully!"
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_ACTIVATE)"

.PHONY: clean
clean: ## Remove cache files
	@echo "Cleaning up..."
	$(RM) .coverage
	$(RM) .pytest_cache
	$(RM) __pycache__
	$(RM) weights.txt
	$(RM) datasets/houses.csv
	$(RM) datasets/Training_houses.csv
	$(RM) datasets/Validation_houses.csv
	find . -type d -name "__pycache__" -exec $(RM) {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup completed!"

.PHONY: clean-venv
clean-venv: clean## Remove virtual environment
	@echo "Cleaning up..."
	$(RM) $(VENV)
	@echo "Cleanup completed!"

.PHONY: lint
lint: ## Run linting checks
	@echo "Running linting checks..."
	. $(VENV_ACTIVATE) && python -m flake8 $(SRC_DIR) --max-line-length=79
	. $(VENV_ACTIVATE) && python -m black --check --line-length=79 $(SRC_DIR)

.PHONY: format
format: ## Format code with black
	@echo "Formatting code..."
	. $(VENV_ACTIVATE) && python -m black --line-length=79 $(SRC_DIR)


.PHONY: requirements
requirements: ## Generate requirements.txt from current environment
	@echo "Generating requirements.txt..."
	. $(VENV_ACTIVATE) && $(PIP) freeze > $(REQUIREMENTS_FILE)
	@echo "Requirements file generated!"

.PHONY: update
update: ## Update all dependencies to latest versions
	@echo "Updating dependencies..."
	. $(VENV_ACTIVATE) && $(PIP) install --upgrade -r $(REQUIREMENTS_FILE)
	@echo "Dependencies updated!"

shell:
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_ACTIVATE)"

.PHONY: programs
programs: ## List all available programs
	@echo "Available programs in the project:"
	. $(VENV_ACTIVATE) && python $(SRC_DIR)/main.py list