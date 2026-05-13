# deep-learning-pipeline

> Small **publisher → in-memory bus → subscriber** layout for dataset
> batching and training-side hooks. Same structural idea as a pub/sub
> data pipeline without requiring Kafka or Redis for local work.

```mermaid
flowchart LR
    DS[("📦<br/>datasets/")]
    PUB["📤 publishers<br/>dataset_batch"]
    BUS{{"🚌 in-memory bus<br/><i>InMemoryBus</i>"}}
    SUB["📥 subscribers<br/>batch_collector"]
    RUN["🎯 pipeline.runner"]
    OUT[/"🧠 training hooks<br/>logs · metrics"/]

    DS --> PUB --> BUS --> SUB --> OUT
    RUN -. wires .-> PUB
    RUN -. wires .-> BUS
    RUN -. wires .-> SUB

    classDef io fill:#0e1116,stroke:#2f81f7,stroke-width:1.5px,color:#e6edf3;
    classDef tool fill:#161b22,stroke:#3fb950,stroke-width:1.5px,color:#e6edf3;
    classDef brain fill:#161b22,stroke:#d29922,stroke-width:1.5px,color:#e6edf3;
    classDef out fill:#0e1116,stroke:#a371f7,stroke-width:1.5px,color:#e6edf3;
    class DS io;
    class PUB,SUB tool;
    class BUS,RUN brain;
    class OUT out;
```

## Table of contents

- [Layout](#layout)
- [Architecture at a glance](#architecture-at-a-glance)
- [Quick start](#quick-start)
- [License](#license)

## Layout

- `src/dl_pipeline/bus/` — message types and synchronous bus
- `src/dl_pipeline/publishers/` — dataset / batch producers
- `src/dl_pipeline/subscribers/` — sinks (logging, metrics, trainers)
- `src/dl_pipeline/pipeline/` — wiring and orchestration
- `datasets/` — manifests or references to real data locations

### Architecture at a glance

```mermaid
flowchart TB
    subgraph PIPE["🎯 pipeline · runner"]
        WIRE["wire publishers → bus → subscribers"]
    end
    subgraph BUSBOX["🚌 bus"]
        PROTO["protocol.py · Message"]
        MEM["in_memory.py · InMemoryBus"]
    end
    subgraph PUBS["📤 publishers"]
        DB["dataset_batch.py"]
    end
    subgraph SUBS["📥 subscribers"]
        BC["batch_collector.py"]
    end
    WIRE --> DB --> MEM
    MEM --> BC
    PROTO -.types.-> MEM
    PROTO -.types.-> DB
    PROTO -.types.-> BC
```

## Quick start

```bash
cd deep-learning-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
