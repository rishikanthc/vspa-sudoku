## Run pretraining

```bash
uv sync
uv run python main.py
```

## Start MLflow

The default Hydra config points training at `http://127.0.0.1:5000`, so start the tracking server before launching training:

```bash
bash scripts/start_mlflow.sh
```

This helper starts MLflow on `0.0.0.0:5000` and enables:

- `--cors-allowed-origins "*"`
- `--allowed-hosts "*"`

These settings are intentionally permissive for LAN/dev use.
