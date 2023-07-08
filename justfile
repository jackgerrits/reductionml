build-site:
  mdbook build book
  mkdir -p site-dist/book
  rm -r site-dist/
  mkdir -p site-dist/book
  cp -r book/book site-dist/
  cp site/index.html site-dist/index.html

serve-site:
  python -m http.server --directory site-dist/

update-schema:
  cargo run --bin reml -- gen-schema > ./schemas/config/latest/schema.json