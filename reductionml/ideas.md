- enums to pass generic types between reductions

- Function to map config to reduction type
- reduction type as enum
- json5 for format

TODO:
- Work out how to represent the graph structure


- passing the incorrect type to a reduction function is considered a bug and so panic is used.
- types can and must be checked ahead of time.

use a supertrait above Any for config to allow a generic any type for config. then downcast in the relvant methods which use it.


missing features
- interactions
- metrics  (loss, cb estimate etc)