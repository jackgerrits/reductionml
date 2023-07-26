py-develop:
  maturin develop --manifest-path ./reductionml-python/Cargo.toml

py-install:
  maturin build --manifest-path ./reductionml-python/Cargo.toml && pip install --find-links ./target/wheels --force-reinstall reductionml

py-install-docs-extension:
  maturin build --manifest-path ./utils/reductionml-docs-extension/Cargo.toml && pip install --find-links ./target/wheels --force-reinstall reductionml-docs-extension

py-test:
  PYTHONPATH=$(pwd)/reductionml-python/python python -m pytest ./reductionml-python/tests

py-docs: py-install py-install-docs-extension
  make -C docs/ html && python -m http.server --directory docs/build/html/

update-schema:
  cargo run --bin reml -- gen-schema > ./schemas/config/latest/schema.json

@test:
  cargo nextest run --run-ignored all || cargo test -- --include-ignored
