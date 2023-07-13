import reductionml as reml


def test_package():
    config = {"entryReduction": {"config": {}, "typename": "Coin"}, "globalConfig": {}}
    assert reml.Workspace.create_from_config(config)


def test_create_features():
    assert reml.SparseFeatures() is not None


def test_package_str():
    config = {"entryReduction": {"config": {}, "typename": "Coin"}, "globalConfig": {}}
    w = reml.Workspace.create_from_config(config)
    text_parser = w.create_parser(reml.FormatType.VwText)
    features, label = text_parser.parse("1 |f 1:1 2:1 3:1 4:1 5:1 6:1 7:1 8:1")
    assert isinstance(features, reml.SparseFeatures)
    assert isinstance(label, reml.SimpleLabel)
    w.learn(features, label)
    pred = w.predict(features)
    assert isinstance(pred, reml.ScalarPred)


def test_package_json():
    config = {"entryReduction": {"config": {}, "typename": "Coin"}, "globalConfig": {}}
    w = reml.Workspace.create_from_config(config)
    text_parser = w.create_parser(reml.FormatType.Json)
    features, label = text_parser.parse(
        {
            "label": {
                "value": 1.0,
                "weight": 1.0,
            },
            "features": {"f": [1, 1, 1, 1, 1, 1, 1, 1]},
        }
    )
    assert isinstance(features, reml.SparseFeatures)
    assert isinstance(label, reml.SimpleLabel)
    w.learn(features, label)
    pred = w.predict(features)
    assert isinstance(pred, reml.ScalarPred)
