# deep-learning-pipeline

Small **publisher → in-memory bus → subscriber** layout for dataset batching and training-side hooks. Same structural idea as a pub/sub data pipeline without requiring Kafka or Redis for local work.

## Layout

- `src/dl_pipeline/bus/` — message types and synchronous bus
- `src/dl_pipeline/publishers/` — dataset / batch producers
- `src/dl_pipeline/subscribers/` — sinks (logging, metrics, trainers)
- `src/dl_pipeline/pipeline/` — wiring and orchestration
- `datasets/` — manifests or references to real data locations

## Quick start

```bash
cd deep-learning-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
