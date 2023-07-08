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

py-test:
  PYTHONPATH=$(pwd)/reductionml-python/python python -m pytest ./reductionml-python/tests

update-schema:
  cargo run --bin reml -- gen-schema > ./schemas/config/latest/schema.json
