build-site:
  mdbook build book
  mkdir -p site-dist/book
  rm -r site-dist/
  mkdir -p site-dist/book
  cp -r book/book site-dist/
  cp site/index.html site-dist/index.html

serve-site:
  python -m http.server --directory site-dist/

py-develop:
  maturin develop --manifest-path ./reductionml-python/Cargo.toml

py-install:
  maturin build --manifest-path ./reductionml-python/Cargo.toml && pip install --find-links ./target/wheels --force-reinstall reductionml

py-test:
  PYTHONPATH=$(pwd)/reductionml-python/python python -m pytest ./reductionml-python/tests

py-docs: py-install
  make -C reductionml-python/docs/ html && python -m http.server --directory reductionml-python/docs/build/html/

update-schema:
  cargo run --bin reml -- gen-schema > ./schemas/config/latest/schema.json

@test:
  cargo nextest run --run-ignored all || cargo test -- --include-ignored
