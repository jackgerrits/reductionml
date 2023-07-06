build-site:
  mdbook build book
  mkdir -p site-dist/
  cp -r book/book site-dist/book
  cp site/index.html site-dist/index.html

serve-site:
  python -m http.server --directory site-dist/
