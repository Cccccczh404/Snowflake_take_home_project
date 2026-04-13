"""Manages Snowflake dynamic table."""

from ..dynamic_table._generated.models import (
    DownstreamLag,
    DynamicTable,
    DynamicTableClone,
    DynamicTableColumn,
    ErrorResponse,
    PointOfTime,
    PointOfTimeOffset,
    PointOfTimeStatement,
    PointOfTimeTimestamp,
    SuccessAcceptedResponse,
    SuccessResponse,
    TargetLag,
    UserDefinedLag,
)
from ._dynamic_table import DynamicTableCollection, DynamicTableResource


# TODO: validate if all classes should be exported
__all__ = [
    "DynamicTableResource",
    "DynamicTableCollection",
    "DownstreamLag",
    "DynamicTable",
    "DynamicTableClone",
    "DynamicTableColumn",
    "ErrorResponse",
    "PointOfTime",
    "PointOfTimeOffset",
    "PointOfTimeStatement",
    "PointOfTimeTimestamp",
    "SuccessAcceptedResponse",
    "SuccessResponse",
    "TargetLag",
    "UserDefinedLag",
]
