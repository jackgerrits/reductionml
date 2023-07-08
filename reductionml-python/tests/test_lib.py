import reductionml


def test_package():
    config = {"entryReduction": {"config": {}, "typename": "Coin"}, "globalConfig": {}}
    assert reductionml.Workspace(config)
