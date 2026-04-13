# mypy: ignore-errors

from concurrent.futures import Future
from datetime import time
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload

from pydantic import StrictStr

from snowflake.core import PollingOperation

from .._internal.telemetry import api_telemetry
from .._internal.utils import deprecated
from ..exceptions import InvalidArgumentsError, InvalidResultError
from ._generated.api.function_api_base import FunctionCollectionBase, FunctionResourceBase
from ._generated.models.function_argument import FunctionArgument


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


def _cast_result(result: Any, returns: StrictStr) -> Any:
    if returns in ["NUMBER", "INT", "FIXED"]:
        return int(result)
    if returns == "REAL":
        return float(result)
    if returns == "TEXT":
        return str(result)
    if returns == "TIME":
        return time(result)
    if returns == "BOOLEAN":
        return bool(int(result))
    return result


class FunctionCollection(FunctionCollectionBase):
    """Represents the collection operations on the Snowflake Function resource.

    With this collection, you can create, iterate through, and search for function that you have access to in the
    current context.

    Examples
    ________
    Creating a function instance:

    >>> functions = root.databases["my_db"].schemas["my_schema"].functions
    >>> new_function = Function(
    ...     name="foo",
    ...     returns="NUMBER",
    ...     arguments=[FunctionArgument(datatype="NUMBER")],
    ...     service="python",
    ...     endpoint="https://example.com",
    ...     path="example.py",
    ... )
    >>> functions.create(new_function)
    """

    _identifier_requires_args = True

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, FunctionResource)


class FunctionResource(FunctionResourceBase):
    """Represents a reference to a Snowflake function.

    With this function reference, you can create and fetch information about functions, as well
    as perform certain actions on them.
    """

    _identifier_requires_args = True
    _plural_name = "functions"

    def __init__(self, name_with_args: StrictStr, collection_class: FunctionCollection) -> None:
        super().__init__(name_with_args, collection_class)

    @api_telemetry
    @deprecated("drop")
    def delete(self, if_exists: bool = False) -> None:
        self.drop(if_exists=if_exists)

    @api_telemetry
    def execute(self, input_args: Optional[list[Any]] = None) -> Any:
        """Execute this function.

        Parameters
        __________
        input_args: list[Any], optional
            A list of arguments to pass to the function. The number of arguments must match the number of arguments
            the function expects.

        Examples
        ________
        Executing a function using its reference:

        >>> function_reference.execute(input_args=[1, 2, "word"])
        """
        return self._execute(input_args=input_args, async_req=False)

    @api_telemetry
    def execute_async(self, input_args: Optional[list[Any]] = None) -> PollingOperation[Any]:
        """An asynchronous version of :func:`execute`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        return self._execute(input_args=input_args, async_req=True)

    @overload
    def _execute(self, input_args: Optional[list[Any]], async_req: Literal[True]) -> PollingOperation[Any]: ...

    @overload
    def _execute(self, input_args: Optional[list[Any]], async_req: Literal[False]) -> Any: ...

    def _execute(self, input_args: Optional[list[Any]], async_req: bool) -> Union[Any, PollingOperation[Any]]:
        function = self.fetch()
        args_count = len(function.arguments) if function.arguments is not None else 0

        if input_args is None:
            input_args = []

        if len(input_args) != args_count:
            raise InvalidArgumentsError(f"Function expects {args_count} arguments but received {len(input_args)}")

        function_args = []
        for i in range(args_count):
            argument = FunctionArgument()
            argument.value = input_args[i]
            argument.datatype = function.arguments[i].datatype
            function_args.append(argument)

        result_or_future = self.collection._api.execute_function(
            self.database.name, self.schema.name, function.name, function_args, async_req=async_req
        )

        def map_result(result: object) -> Any:
            if not isinstance(result, dict) or len(result.values()) != 1:
                raise InvalidResultError(f"Function result {str(result)} is invalid or empty")

            result = list(result.values())[0]
            return _cast_result(result, str(function.returns))

        if isinstance(result_or_future, Future):
            return PollingOperation(result_or_future, map_result)
        return map_result(result_or_future)
