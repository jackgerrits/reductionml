from enum import Enum
from typing import Optional, Union, List, Tuple, Dict, Any

class SimpleLabel:
    def __init__(self, value: float, weight: float) -> None: ...
    @property
    def value(self) -> float: ...
    @property
    def weight(self) -> float: ...

class CbLabel:
    def __init__(self, action: int, cost: float, probability: float) -> None: ...
    @property
    def action(self) -> int: ...
    @property
    def cost(self) -> float: ...
    @property
    def probability(self) -> float: ...

class ScalarPred:
    @property
    def prediction(self) -> float: ...
    @property
    def raw_prediction(self) -> float: ...

class ActionScoresPred:
    @property
    def value(self) -> List[Tuple[int, float]]: ...

class ActionProbsPred:
    @property
    def value(self) -> List[Tuple[int, float]]: ...

class SparseFeatures:
    def __init__(self) -> None: ...

class CbAdfFeatures:
    def __init__(self) -> None: ...


# TODO: are integers correct here?
class FormatType(Enum):
    VwText = 1,
    Json = 2
    DsJson = 3

# TODO: are integers correct here?
class ReductionType(Enum):
    CB = 1,
    Simple = 2

Label = Union[SimpleLabel, CbLabel]
Features = Union[SparseFeatures, CbAdfFeatures]
Prediction = Union[ScalarPred, ActionScoresPred, ActionProbsPred]

class Parser:
    @staticmethod
    def create_parser_with_workspace(format_type: FormatType, workspace: Workspace) -> Parser: ...
    @staticmethod
    def create_parser(format_type: FormatType, reduction_type: ReductionType, hash_seed: int, num_bits: int) -> Parser: ...
    def parse(self, line: str) -> Tuple[Features, Option[Label]]: ...

class Workspace:
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> Workspace: ...
    @staticmethod
    def create_from_model(data: bytearray) -> Workspace: ...
    @staticmethod
    def create_from_json_model(data: Dict[str, Any]) -> Workspace: ...

    def serialize(self) -> bytearray: ...
    def serialize_to_json(self) -> Dict[str, Any]: ...
    def predict(self, features: Features) -> Prediction: ...
    def learn(self, features: Features, label: Label) -> None: ...
    def predict_then_learn(self, features: Features, label: Label) -> Prediction: ...

