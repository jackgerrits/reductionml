# Configuration

## Interactions

Interactions are specified in the global config. For example, the following is a single quadratic interaction between the default namespace and "namespace":

```json
{
    "globalConfig": {
        "interactions": [
            ["Default", {"Named": "namespace"}]
        ]
    }
}
