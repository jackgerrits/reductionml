# Model serialization

There are two kinds of serialization, binary and json.

```{caution}
Serialized models should be loaded with the same version of the library that was used to create them. There are currently no guarantees that models will be compatible across versions.
```

## Binary

### Saving

`````{tab-set}

````{tab-item} Python
:sync: python

[`reductionml.Workspace.serialize`](reductionml.Workspace.serialize)


```python
from reductionml import Workspace
config = {"entryReduction": {"typename": "Coin"}, "globalConfig": {}}
workspace = Workspace.create_from_config(config)

model_data = workspace.serialize()
```

````

````{tab-item} CLI
:sync: cli

Binary models can be created at the end of a training run using the `--output-model` option:

```
-o, --output-model <OUTPUT_MODEL>
```

````

`````

### Loading

`````{tab-set}

````{tab-item} Python
:sync: python

[`reductionml.Workspace.create_from_model`](reductionml.Workspace.create_from_model)


```Python
from reductionml import Workspace

model_data = ... # load from file or other source
workspace = Workspace.create_from_model(config)
```
````

````{tab-item} CLI
:sync: cli


Binary models can be loading for a training run using the `--input-model` option:

```
-i, --input-model <INPUT_MODEL>
```


````

`````

## Json

```{caution}
While a very powerful and helpful tool, extreme care must be taken if the json model is edited by hand. No guarantees are made that any changes will load back in correctly. If you do edit the model, you should understand all the implications of the changes you are making.
```

### Saving

`````{tab-set}

````{tab-item} Python
:sync: python

[`reductionml.Workspace.serialize_to_json`](reductionml.Workspace.serialize_to_json)

```Python
from reductionml import Workspace
config = {"entryReduction": {"typename": "Coin"}, "globalConfig": {}}
workspace = Workspace.create_from_config(config)

model_data = workspace.serialize_to_json()
```

````

````{tab-item} CLI
:sync: cli

To create a json model file using the command line you must have an existing binary model and then it can be converted to json with:

```
reml export-model <INPUT_MODEL>
```

````

`````

### Loading

`````{tab-set}

````{tab-item} Python
:sync: python

[`reductionml.Workspace.create_from_json_model`](reductionml.Workspace.create_from_json_model)

```Python
from reductionml import Workspace

model_data = ... # load from file or other source
workspace = Workspace.create_from_json_model(config)
```

````

````{tab-item} CLI
:sync: cli

To convert a json model back into a binary model the following command can be used:

```
reml import-model <INPUT_FILE> --output-model <OUTPUT_MODEL>
```

````

`````

