.. div:: sd-text-center sd-fs-1 sd-font-weight-bold sd-mt-5 sd-mb-5

    ReductionML

.. div:: sd-text-center sd-fs-3 sd-mb-2 sd-mb-5

    Reduction-based machine learning framework

.. role:: raw-html(raw)
    :format: html

.. grid:: 1 1 2 2

    .. grid-item::

        .. card::
            :class-header: card-emoji

            🐍 :raw-html:`<br />` Python package
            ^^^

            Install from `PyPi <https://pypi.org/project/reductionml/>`_:

            ::

                pip install reductionml

            .. raw:: html

                <div style="display:flex;">

            .. button-ref:: getting_started
                :color: primary
                :shadow:
                :class: sd-mr-1

                Getting started

            .. button-ref:: reference
                :color: primary
                :shadow:

                API reference

            .. raw:: html

                </div>


    .. grid-item::

        .. card::
            :class-header: card-emoji

            📦 :raw-html:`<br />` Rust package
            ^^^

            Install from `crates.io <https://crates.io/crates/reductionml-core>`__:

            ::

                cargo add reductionml-core

            .. button-link:: https://docs.rs/reductionml-core/latest/reductionml_core/
                :color: primary
                :shadow:

                API reference :octicon:`link-external`

    .. grid-item::

        .. card::
            :class-header: card-emoji

            ⌨️ :raw-html:`<br />` Command line tool
            ^^^

            Install from `crates.io <https://crates.io/crates/reductionml-cli>`__:

            ::

                cargo install reductionml-cli

            Binary name: ``reml``

            .. button-link:: https://github.com/jackgerrits/reductionml#first-steps
                :color: primary
                :shadow:

                Getting started :octicon:`link-external`

.. toctree::
    :hidden:

    GitHub <https://github.com/jackgerrits/reductionml>
    getting_started
    configuration
    input_formats
    model_serialization
    reductions/index

.. toctree::
    :caption: API Reference
    :hidden:

    Python <reference>

    Rust <https://docs.rs/reductionml-core/latest/reductionml_core/>
