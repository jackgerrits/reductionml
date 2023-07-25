from __future__ import annotations
from . import _reductionml
import typing
import sys
from typing import Any, Dict, Literal, Optional, List, Union

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

SimpleLabel = _reductionml.SimpleLabel
ActionProbsPred = _reductionml.ActionProbsPred
ActionScoresPred = _reductionml.ActionScoresPred
CbAdfFeatures = _reductionml.CbAdfFeatures
CbLabel = _reductionml.CbLabel
FormatType = _reductionml.FormatType
JsonParser = _reductionml.JsonParser
ReductionType = _reductionml.ReductionType
ScalarPred = _reductionml.ScalarPred
SparseFeatures = _reductionml.SparseFeatures
TextParser = _reductionml.TextParser
ReductionTypesDescription = _reductionml.ReductionTypesDescription

__all__ = [
    "ActionProbsPred",
    "ActionScoresPred",
    "CbAdfFeatures",
    "CbLabel",
    "Config",
    "EntryReductionConfig",
    "Features",
    "FormatType",
    "GlobalConfig",
    "Interaction",
    "JsonParser",
    "Label",
    "NameInteraction",
    "Prediction",
    "ReductionType",
    "ReductionTypesDescription",
    "ScalarPred",
    "SimpleLabel",
    "SparseFeatures",
    "TextParser",
    "Workspace",
]


class NameInteraction(TypedDict):
    name: str


Interaction = Union[Literal["Default"], NameInteraction]


class GlobalConfig(TypedDict):
    numBits: Optional[int]
    hashSeed: Optional[int]
    constantFeatureEnabled: Optional[bool]
    interactions: Optional[List[Interaction]]


class EntryReductionConfig(TypedDict):
    config: Dict[str, Any]
    typename: str


class Config(TypedDict):
    globalConfig: GlobalConfig
    entryReduction: EntryReductionConfig


Features = typing.Union[SparseFeatures, CbAdfFeatures]
Label = typing.Union[SimpleLabel, CbLabel]
Prediction = typing.Union[ScalarPred, ActionScoresPred, ActionProbsPred]


class Workspace:
    __create_key = object()

    def __init__(self, create_key, workspace: _reductionml.Workspace):
        assert (
            create_key == Workspace.__create_key
        ), "Workspace objects must be created using Workspace.create_from_config, Workspace.create_from_model, or Workspace.create_from_json_model"
        self._workspace = workspace

    @classmethod
    def create_from_config(cls, config: Config) -> Workspace:
        cast_config = typing.cast(Dict[str, Any], config)
        return Workspace(
            cls.__create_key, _reductionml.Workspace.create_from_config(cast_config)
        )

    @classmethod
    def create_from_model(cls, data: bytearray) -> Workspace:
        return Workspace(
            cls.__create_key, _reductionml.Workspace.create_from_model(data)
        )

    @classmethod
    def create_from_json_model(
        cls, model_json: typing.Dict[str, typing.Any]
    ) -> Workspace:
        return Workspace(
            cls.__create_key, _reductionml.Workspace.create_from_json_model(model_json)
        )

    def serialize(self) -> bytearray:
        return self._workspace.serialize()

    def serialize_to_json(self) -> typing.Dict[str, typing.Any]:
        return self._workspace.serialize_to_json()

    @typing.overload
    def create_parser(
        self,
        format_type: typing.Literal[FormatType.VwText],
    ) -> TextParser:
        ...

    @typing.overload
    def create_parser(
        self,
        format_type: typing.Union[
            typing.Literal[FormatType.DsJson], typing.Literal[FormatType.Json]
        ],
    ) -> JsonParser:
        ...

    def create_parser(
        self,
        format_type: FormatType,
    ) -> typing.Union[TextParser, JsonParser]:
        return self._workspace.create_parser(format_type)

    def predict(self, features: Features) -> Prediction:
        return self._workspace.predict(features)

    def learn(
        self,
        features: Features,
        label: Label,
    ) -> None:
        return self._workspace.learn(features, label)

    def predict_then_learn(
        self,
        features: Features,
        label: Label,
    ) -> Prediction:
        return self._workspace.predict_then_learn(features, label)
