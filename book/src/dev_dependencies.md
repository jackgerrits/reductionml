# Dev dependencies

- [Rust](https://www.rust-lang.org/tools/install) (required)
- [`just`](https://github.com/casey/just) (optional)
  - Used for helpful shortcuts. See the [justfile](https://github.com/jackgerrits/reductionml/blob/main/justfile).
- [`mdbook`](https://rust-lang.github.io/mdBook/) (optional)
  - To build this documentation book
- [`maturin`](https://www.maturin.rs/) (optional)
  - To build the python bindings
- [`cargo-nextest](https://nexte.st/)
  - A nicer experience for running tests. `just test` uses this if available.