from enum import Enum
from typing import Optional, Union, List, Tuple, Dict, Any, final, overload, Literal

@final
class SimpleLabel:
    def __init__(self, value: float, weight: float) -> None: ...
    @property
    def value(self) -> float: ...
    @property
    def weight(self) -> float: ...

@final
class CbLabel:
    def __init__(self, action: int, cost: float, probability: float) -> None: ...
    @property
    def action(self) -> int: ...
    @property
    def cost(self) -> float: ...
    @property
    def probability(self) -> float: ...

@final
class ScalarPred:
    @property
    def prediction(self) -> float: ...
    @property
    def raw_prediction(self) -> float: ...

@final
class ActionScoresPred:
    @property
    def value(self) -> List[Tuple[int, float]]: ...

@final
class ActionProbsPred:
    @property
    def value(self) -> List[Tuple[int, float]]: ...

@final
class SparseFeatures:
    def __init__(self, /, *args, **kwargs) -> None: ...

@final
class CbAdfFeatures:
    def __init__(self, /, *args, **kwargs) -> None: ...

# TODO: are integers correct here?
@final
class FormatType(Enum):
    VwText = (1,)
    Json = 2
    DsJson = 3

# TODO: are integers correct here?
@final
class ReductionType(Enum):
    CB = (1,)
    Simple = 2

@overload
def create_parser(
    format_type: Literal[FormatType.VwText],
    reduction_type: ReductionType,
    hash_seed: int,
    num_bits: int,
) -> TextParser: ...
@overload
def create_parser(
    format_type: Union[Literal[FormatType.DsJson], Literal[FormatType.Json]],
    reduction_type: ReductionType,
    hash_seed: int,
    num_bits: int,
) -> JsonParser: ...
@final
class TextParser:
    def parse(
        self, input: str
    ) -> Tuple[
        Union[SparseFeatures, CbAdfFeatures], Optional[Union[SimpleLabel, CbLabel]]
    ]: ...

@final
class JsonParser:
    def parse(
        self, input: Union[str, Dict[str, Any]]
    ) -> Tuple[
        Union[SparseFeatures, CbAdfFeatures], Optional[Union[SimpleLabel, CbLabel]]
    ]: ...

@final
class Workspace:
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> Workspace: ...
    @staticmethod
    def create_from_model(data: bytearray) -> Workspace: ...
    @staticmethod
    def create_from_json_model(model_json: Dict[str, Any]) -> Workspace: ...
    def serialize(self) -> bytearray: ...
    def serialize_to_json(self) -> Dict[str, Any]: ...
    @overload
    def create_parser(
        self,
        format_type: Literal[FormatType.VwText],
    ) -> TextParser: ...
    @overload
    def create_parser(
        self,
        format_type: Union[Literal[FormatType.DsJson], Literal[FormatType.Json]],
    ) -> JsonParser: ...
    def predict(
        self, features: Union[SparseFeatures, CbAdfFeatures]
    ) -> Union[ScalarPred, ActionScoresPred, ActionProbsPred]: ...
    def learn(
        self,
        features: Union[SparseFeatures, CbAdfFeatures],
        label: Union[SimpleLabel, CbLabel],
    ) -> None: ...
    def predict_then_learn(
        self,
        features: Union[SparseFeatures, CbAdfFeatures],
        label: Union[SimpleLabel, CbLabel],
    ) -> Union[ScalarPred, ActionScoresPred, ActionProbsPred]: ...
