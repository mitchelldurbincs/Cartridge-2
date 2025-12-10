# Simple helpers to run the Cartridge2 loop locally

ENV_ID           ?= tictactoe
ACTOR_EPISODES   ?= 50
ACTOR_LOG_INTERVAL ?= 10
TRAIN_STEPS      ?= 50
TRAIN_BATCH      ?= 32
TRAIN_DEVICE     ?= cpu

PYTHON           ?= python3
CARGO            ?= cargo
NPM              ?= npm
# Use Linux filesystem venv for fast imports (avoid WSL2 /mnt/c slowness)
VENV             := $(HOME)/venvs/cartridge2
VENV_PIP         := $(VENV)/bin/pip
VENV_PYTHON      := $(VENV)/bin/python

# Number of AlphaZero iterations (actor + trainer cycles)
ITERATIONS       ?= 5

.PHONY: data actor trainer web-backend frontend-dev full-loop train-loop trainer-install venv clean-data clean-models clean-all \
        actor-tictactoe actor-connect4 train-loop-tictactoe train-loop-connect4

data:
	mkdir -p data data/models

# Clean targets
clean-data:
	rm -f data/replay.db data/stats.json

clean-models:
	rm -f data/models/*.onnx data/models/*.onnx.data data/models/*.pt

clean-all: clean-data clean-models
	rm -rf data/

# Create virtual environment for trainer
$(VENV):
	$(PYTHON) -m venv $(VENV)

# Run a short self-play job to populate replay.db
actor: data
	cd actor && $(CARGO) run -- --env-id $(ENV_ID) --max-episodes $(ACTOR_EPISODES) --log-interval $(ACTOR_LOG_INTERVAL) --replay-db-path ../data/replay.db --data-dir ../data

# Game-specific actor shortcuts
actor-tictactoe: data
	$(MAKE) actor ENV_ID=tictactoe

actor-connect4: data
	$(MAKE) actor ENV_ID=connect4

# Install trainer dependencies once (editable mode) - creates venv if needed
# don't do this shit on wsl, hella slow
trainer-install: $(VENV)
	$(VENV_PIP) install -e trainer/

# Train a small model and write stats/model artifacts
trainer: data
	$(VENV_PYTHON) -m trainer --db data/replay.db --model-dir data/models --stats data/stats.json --steps $(TRAIN_STEPS) --batch-size $(TRAIN_BATCH) --device $(TRAIN_DEVICE) --env-id $(ENV_ID)

# Start the Rust backend (Axum). Blocks until stopped.
web-backend: data
	cd web && DATA_DIR=../data $(CARGO) run

# Start the Svelte frontend dev server. Blocks until stopped.
frontend-dev:
	cd web/frontend && $(NPM) install && $(NPM) run dev

# Convenience: run actor then trainer with small defaults
full-loop: actor trainer

# AlphaZero training loop: run multiple iterations of actor + trainer
# Usage: make train-loop ITERATIONS=5 ACTOR_EPISODES=500 TRAIN_STEPS=1000
train-loop: data
	@for i in $$(seq 1 $(ITERATIONS)); do \
		echo ""; \
		echo "=== Iteration $$i/$(ITERATIONS) ==="; \
		echo ""; \
		$(MAKE) actor ACTOR_EPISODES=$(ACTOR_EPISODES) ACTOR_LOG_INTERVAL=$(ACTOR_LOG_INTERVAL); \
		$(MAKE) trainer TRAIN_STEPS=$(TRAIN_STEPS) TRAIN_BATCH=$(TRAIN_BATCH) TRAIN_DEVICE=$(TRAIN_DEVICE); \
	done
	@echo ""
	@echo "=== Training complete: $(ITERATIONS) iterations ==="

# Game-specific training loop shortcuts
train-loop-tictactoe: data
	$(MAKE) train-loop ENV_ID=tictactoe

train-loop-connect4: data
	$(MAKE) train-loop ENV_ID=connect4
