//! ASCII banner and help copy for the CLI.

/// Minimal “network” motif (pure ASCII).
pub const BANNER: &str = r"
   o     o     o
  /|\   /|\   /|\      D E E P   L E A R N I N G
   |     |     |      datasets + tiny linear core
  / \   / \   / \
";

/// Short tagline for `--help` headers.
pub const TAGLINE: &str = "In-memory datasets, minimal linear model, SGD loop, `dlpipe` CLI.";

/// Long-form body for `clap` (`long_about`): banner + tagline + tips.
pub const HELP_LONG: &str = r"

   o     o     o
  /|\   /|\   /|\      D E E P   L E A R N I N G
   |     |     |      datasets + tiny linear core
  / \   / \   / \

In-memory datasets, minimal linear model, SGD loop, `dlpipe` CLI.

Tips
  RUST_LOG=info (or debug, trace) works with `dlpipe --verbose …`.

Scope
  Educational scaffold: no external tensor stack. Swap `model`/`train` for your DL framework later.
";
