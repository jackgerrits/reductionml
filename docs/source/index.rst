.. div:: sd-text-center sd-fs-1 sd-font-weight-bold sd-mt-5 sd-mb-5

    ReductionML

.. div:: sd-text-center sd-fs-3 sd-mb-5

    Reduction-based machine learning framework

.. div:: sd-mb-5

    ReductionML is a machine learning framework with solutions to a range of
    problems. It revolves around the concept of simplifying problems by breaking
    them down into more manageable components that already have solutions. This
    process is done by `reductions <reductions>`__ as they reduce one problem to
    another. This approach draws inspiration from the `VowpalWabbit
    <https://github.com/VowpalWabbit/vowpal_wabbit>`__, a project I hold in high
    regard and deeply value. In fact, if you are familiar with VowpalWabbit then you
    should be able to pick up ReductionML with ease.


.. role:: raw-html(raw)
    :format: html

Packages
^^^^^^^^

.. grid:: 1 1 2 2

    .. grid-item::

        .. card::
            :class-header: card-header

            :raw-html:`<span class="card-emoji">🐍</span><br />` Python package
            ^^^

            Install from `PyPi <https://pypi.org/project/reductionml/>`__:

            ::

                pip install reductionml

            .. raw:: html

                <div style="display:flex;">

            .. button-ref:: getting_started
                :color: info
                :shadow:
                :class: sd-mr-1

                Getting started

            .. button-ref:: reference
                :color: info
                :shadow:

                API reference

            .. raw:: html

                </div>


    .. grid-item::

        .. card::
            :class-header: card-header

            :raw-html:`<span class="card-emoji">📦</span><br />` Rust package
            ^^^

            Install from `crates.io <https://crates.io/crates/reductionml-core>`__:

            ::

                cargo add reductionml-core

            .. button-link:: https://docs.rs/reductionml-core/latest/reductionml_core/
                :color: info
                :shadow:

                API reference :octicon:`link-external`

    .. grid-item::

        .. card::
            :class-header: card-header

            :raw-html:`<span class="card-emoji">⌨️</span><br />` Command line tool
            ^^^

            Install from `crates.io <https://crates.io/crates/reductionml-cli>`__:

            ::

                cargo install reductionml-cli

            Binary name: ``reml``

            .. button-ref:: getting_started_cli
                :color: info
                :shadow:

                Getting started

.. toctree::
    :hidden:

    GitHub <https://github.com/jackgerrits/reductionml>
    getting_started
    getting_started_cli
    configuration
    input_formats
    model_serialization
    reductions/index

.. toctree::
    :caption: API Reference
    :hidden:

    Python <reference>

    Rust <https://docs.rs/reductionml-core/latest/reductionml_core/>
