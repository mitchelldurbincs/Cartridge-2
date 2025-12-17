# Plan: Refactor Trainer Config CLI Wiring

## Problem

`__main__.py` defines ~18 argparse arguments and then manually mirrors each one into `TrainerConfig` (lines 163-182). This causes:

1. **Duplicate defaults** - both argparse and dataclass define `batch_size=64`, `lr=1e-3`, etc.
2. **Manual mapping** - every argument must be explicitly mapped (`args.lr` → `config.learning_rate`)
3. **Drift risk** - easy to change a default in one place but not the other
4. **Name mismatches** - CLI names differ from config fields (`--db` → `db_path`, `--lr` → `learning_rate`)

## Solution

Add argument metadata to `TrainerConfig` fields and provide two class methods:
- `configure_parser(parser)` - adds arguments to parser from field metadata
- `from_args(args)` - constructs config from parsed args

### Approach: Field Metadata + Class Methods

Use dataclass `field(metadata={...})` to store CLI configuration alongside each field:

```python
@dataclass
class TrainerConfig:
    db_path: str = field(
        default="./data/replay.db",
        metadata={"cli": "--db", "help": "Path to SQLite replay database"}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"cli": "--lr", "help": "Learning rate"}
    )
    # Fields without metadata are not exposed to CLI
    value_loss_weight: float = 1.0
```

## Implementation Steps

### Step 1: Create CLI metadata helper in `trainer.py`

Add a small helper class to define CLI argument metadata:

```python
from dataclasses import dataclass, field, fields
from typing import Any

def cli_field(
    default: Any,
    *,
    cli: str | None = None,  # CLI flag name (e.g., "--lr")
    help: str = "",
    choices: list | None = None,
    action: str | None = None,  # e.g., "store_true"
):
    """Create a dataclass field with CLI metadata."""
    metadata = {}
    if cli is not None:
        metadata["cli"] = cli
        metadata["help"] = help
        if choices:
            metadata["choices"] = choices
        if action:
            metadata["action"] = action
    return field(default=default, metadata=metadata)
```

### Step 2: Update `TrainerConfig` fields to use `cli_field()`

Convert existing field definitions:

```python
@dataclass
class TrainerConfig:
    # Paths
    db_path: str = cli_field("./data/replay.db", cli="--db", help="Path to SQLite replay database")
    model_dir: str = cli_field("./data/models", cli="--model-dir", help="Directory for ONNX checkpoints")
    stats_path: str = cli_field("./data/stats.json", cli="--stats", help="Path to write stats.json")

    # Training hyperparameters
    batch_size: int = cli_field(64, cli="--batch-size", help="Batch size for training")
    learning_rate: float = cli_field(1e-3, cli="--lr", help="Learning rate")
    weight_decay: float = cli_field(1e-4, cli="--weight-decay", help="Weight decay")
    grad_clip_norm: float = cli_field(1.0, cli="--grad-clip", help="Gradient clip max norm (0=disable)")

    # Loss weights (no CLI - internal only)
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0

    # LR scheduler
    use_lr_scheduler: bool = cli_field(True, cli="--no-lr-schedule", action="store_false", help="Disable LR scheduler")
    lr_min_ratio: float = cli_field(0.1, cli="--lr-min-ratio", help="Final LR as ratio of initial")

    # Training schedule
    total_steps: int = cli_field(1000, cli="--steps", help="Total training steps")
    checkpoint_interval: int = cli_field(100, cli="--checkpoint-interval", help="Steps between checkpoints")
    stats_interval: int = cli_field(10, cli="--stats-interval", help="Steps between stats updates")
    log_interval: int = cli_field(10, cli="--log-interval", help="Steps between log messages")
    max_checkpoints: int = cli_field(10, cli="--max-checkpoints", help="Max checkpoints to keep")

    # Wait/timeout
    wait_interval: float = cli_field(2.0, cli="--wait-interval", help="Seconds between data checks")
    max_wait: float = cli_field(300.0, cli="--max-wait", help="Max wait for data (0=forever)")

    # Stats (no CLI)
    max_history_length: int = 100

    # Environment
    env_id: str = cli_field("tictactoe", cli="--env-id", help="Game environment ID")
    device: str = cli_field("cpu", cli="--device", choices=["cpu", "cuda", "mps"], help="Training device")
```

### Step 3: Add `configure_parser()` class method

```python
@classmethod
def configure_parser(cls, parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments to parser based on field metadata."""
    for f in fields(cls):
        meta = f.metadata.get("cli_meta", {})
        if not meta.get("cli"):
            continue

        kwargs = {
            "default": f.default if f.default is not dataclasses.MISSING else None,
            "help": meta.get("help", ""),
        }

        # Handle store_true/store_false actions
        action = meta.get("action")
        if action:
            kwargs["action"] = action
            kwargs.pop("default", None)  # argparse handles default for actions
        else:
            # Infer type from field annotation
            kwargs["type"] = f.type if f.type in (int, float, str) else str

        if meta.get("choices"):
            kwargs["choices"] = meta["choices"]

        parser.add_argument(meta["cli"], **kwargs)
```

### Step 4: Add `from_args()` class method

```python
@classmethod
def from_args(cls, args: argparse.Namespace) -> "TrainerConfig":
    """Construct TrainerConfig from parsed CLI arguments."""
    kwargs = {}
    for f in fields(cls):
        meta = f.metadata.get("cli_meta", {})
        cli_name = meta.get("cli")

        if cli_name:
            # Convert --foo-bar to foo_bar for namespace lookup
            attr_name = cli_name.lstrip("-").replace("-", "_")
            if hasattr(args, attr_name):
                kwargs[f.name] = getattr(args, attr_name)

    return cls(**kwargs)
```

### Step 5: Simplify `__main__.py`

```python
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cartridge2 AlphaZero-style Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add all config arguments from metadata
    TrainerConfig.configure_parser(parser)

    # Add logging (not part of TrainerConfig)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(...)

    # Create config from args (single line!)
    config = TrainerConfig.from_args(args)

    # ... rest unchanged
```

## Files to Modify

1. **`trainer/src/trainer/trainer.py`**
   - Add `cli_field()` helper function
   - Update `TrainerConfig` fields to use `cli_field()`
   - Add `configure_parser()` class method
   - Add `from_args()` class method

2. **`trainer/src/trainer/__main__.py`**
   - Remove all individual `parser.add_argument()` calls for config fields
   - Replace with single `TrainerConfig.configure_parser(parser)` call
   - Replace manual config construction with `TrainerConfig.from_args(args)`
   - Keep `--log-level` as standalone (not part of config)

## Benefits

1. **Single source of truth** - defaults live only in `TrainerConfig`
2. **Declarative** - field metadata makes CLI mapping explicit and scannable
3. **Type-safe** - field types drive argparse type inference
4. **Extensible** - adding a new config field with CLI is one line
5. **No external deps** - pure stdlib + dataclasses

## Testing

- Run existing trainer tests to verify no regression
- Manual test: `python -m trainer --help` should show same arguments
- Manual test: `python -m trainer --db foo.db --steps 10` should work
