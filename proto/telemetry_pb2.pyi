from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Sample(_message.Message):
    __slots__ = ("timestamp_unix_ns", "node_id", "source_type", "entity_type", "entity_id", "metric_name", "unit", "is_categorical", "value", "delta_short", "delta_long", "rolling_var", "labels")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TIMESTAMP_UNIX_NS_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    METRIC_NAME_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    IS_CATEGORICAL_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    DELTA_SHORT_FIELD_NUMBER: _ClassVar[int]
    DELTA_LONG_FIELD_NUMBER: _ClassVar[int]
    ROLLING_VAR_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    timestamp_unix_ns: int
    node_id: str
    source_type: str
    entity_type: str
    entity_id: str
    metric_name: str
    unit: str
    is_categorical: bool
    value: float
    delta_short: float
    delta_long: float
    rolling_var: float
    labels: _containers.ScalarMap[str, str]
    def __init__(self, timestamp_unix_ns: _Optional[int] = ..., node_id: _Optional[str] = ..., source_type: _Optional[str] = ..., entity_type: _Optional[str] = ..., entity_id: _Optional[str] = ..., metric_name: _Optional[str] = ..., unit: _Optional[str] = ..., is_categorical: bool = ..., value: _Optional[float] = ..., delta_short: _Optional[float] = ..., delta_long: _Optional[float] = ..., rolling_var: _Optional[float] = ..., labels: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Batch(_message.Message):
    __slots__ = ("node_id", "batch_timestamp_unix_ns", "samples")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_TIMESTAMP_UNIX_NS_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    batch_timestamp_unix_ns: int
    samples: _containers.RepeatedCompositeFieldContainer[Sample]
    def __init__(self, node_id: _Optional[str] = ..., batch_timestamp_unix_ns: _Optional[int] = ..., samples: _Optional[_Iterable[_Union[Sample, _Mapping]]] = ...) -> None: ...
