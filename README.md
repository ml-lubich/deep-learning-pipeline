# deep-learning-pipeline

In-memory **datasets**, a tiny **linear model** with manual SGD, and the **`dlpipe` CLI**. Layout matches [`pub-sub-pipeline`](https://github.com/ml-lubich/pub-sub-pipeline): library + binary + `examples/` + integration tests + optional **llvm-cov** / **nextest**.

## Architecture

```mermaid
flowchart LR
  D[Dataset synthetic_linear] --> S[split train / val]
  S --> T[fit_sgd loop]
  T --> M[mean_squared_error]
```

| Piece | Role |
|--------|------|
| **`Dataset`** | Synthetic regression rows (`y ≈ 2·x₀ + 3·x₁`) |
| **`LinearModel`** | Forward + single-example SGD |
| **`fit_sgd` / `mean_squared_error`** | Training and evaluation helpers |
| **`dlpipe`** | Binary: `demo`, `fit` |

## Requirements

- **Rust 1.85+** (edition 2024; see `rust-version` in `Cargo.toml`).
- **Coverage** (optional): `cargo install cargo-llvm-cov --locked`
- **Nextest** (optional): `cargo install cargo-nextest --locked`

## Quick start

```bash
cargo test
cargo run --bin dlpipe -- --help
cargo run --bin dlpipe -- demo
cargo run --bin dlpipe -- fit --rows 48 --steps 250 --lr 0.07
cargo run --example demo
```

## Testing

| Workflow | Command |
|----------|---------|
| Default | `cargo test` |
| Nextest | `cargo nextest run` |
| Coverage | `cargo cov` (alias in `.cargo/config.toml`, ≥ 80% lines) |

CLI integration tests use **`assert_cmd`** (`tests/cli_e2e.rs`).

## Verify locally

```bash
cargo fmt --check
cargo test
cargo nextest run   # optional
cargo cov           # optional
cargo clippy --all-targets --locked -- -D warnings
```

## License

MIT — see `Cargo.toml`.
