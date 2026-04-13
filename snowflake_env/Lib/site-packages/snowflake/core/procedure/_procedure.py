# mypy: ignore-errors
import warnings

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload

from pydantic import StrictStr

from snowflake.core import PollingOperation

from .._internal.telemetry import api_telemetry
from .._utils import map_result
from ._generated.api.procedure_api_base import ProcedureCollectionBase, ProcedureResourceBase
from ._generated.models.call_argument_list import CallArgumentList


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class ProcedureCollection(ProcedureCollectionBase):
    """Represents the collection operations on the Snowflake Procedure resource.

    With this collection, you can create, iterate through, and fetch procedures that you have access to
    in the current context.

    Examples
    ________
    Creating a procedure instance:

    >>> procedure = Procedure(
    ...     name="sql_proc_table_func",
    ...     arguments=[Argument(name="id", datatype="VARCHAR")],
    ...     return_type=ReturnTable(
    ...         column_list=[
    ...             ColumnType(name="id", datatype="NUMBER"),
    ...             ColumnType(name="price", datatype="NUMBER"),
    ...         ]
    ...     ),
    ...     language_config=SQLFunction(),
    ...     body="
    ...         DECLARE
    ...             res RESULTSET DEFAULT (SELECT * FROM invoices WHERE id = :id);
    ...         BEGIN
    ...             RETURN TABLE(res);
    ...         END;
    ...     ",
    ... )
    >>> procedures = root.databases["my_db"].schemas["my_schema"].procedures
    >>> procedures.create(procedure)
    """

    _identifier_requires_args = True

    def __init__(self, schema: "SchemaResource") -> None:
        super().__init__(schema, ProcedureResource)


class ProcedureResource(ProcedureResourceBase):
    """Represents a reference to a Snowflake procedure.

    With this procedure reference, you can create and fetch information about procedures, as well as
    perform certain actions on them.
    """

    _identifier_requires_args = True
    _plural_name = "procedures"

    def __init__(self, name_with_args: StrictStr, collection_class: ProcedureCollection) -> None:
        super().__init__(name_with_args, collection_class)

    @api_telemetry
    def call(self, call_argument_list: Optional[CallArgumentList] = None, extract: Optional[bool] = False) -> Any:
        """Call this procedure.

        Examples
        ________
        Calling a procedure with no arguments using its reference:

        >>> procedure_reference.call(call_argument_list=CallArgumentList(call_arguments=[]))

        Calling a procedure with 2 arguments using its reference:

        >>> procedure_reference.call(
        ...     call_argument_list=CallArgumentList(
        ...         call_arguments=[
        ...             CallArgument(name="id", datatype="NUMBER", value=1),
        ...             CallArgument(name="tableName", datatype="VARCHAR", value="my_table_name"),
        ...         ]
        ...     )
        ... )
        """
        return self._call(call_argument_list=call_argument_list, async_req=False, extract=extract)

    @api_telemetry
    def call_async(
        self, call_argument_list: Optional[CallArgumentList] = None, extract: Optional[bool] = False
    ) -> PollingOperation[Any]:
        """An asynchronous version of :func:`call`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        return self._call(call_argument_list=call_argument_list, async_req=True, extract=extract)

    @overload
    def _call(
        self, call_argument_list: Optional[CallArgumentList], extract: Optional[bool], async_req: Literal[True]
    ) -> PollingOperation[Any]: ...

    @overload
    def _call(
        self, call_argument_list: Optional[CallArgumentList], extract: Optional[bool], async_req: Literal[False]
    ) -> Any: ...

    def _call(
        self, call_argument_list: Optional[CallArgumentList], async_req: bool, extract: bool = False
    ) -> Union[Any, PollingOperation[Any]]:
        if extract is False:
            warnings.warn(
                "Please use `extract=True` when calling procedure. This will extract "
                "result from [{sproc_name: result}] object. This will become default behavior.",
                DeprecationWarning,
                stacklevel=4,
            )

        # None is not supported by self.collection._api.call_procedure
        if call_argument_list is None:
            call_argument_list = CallArgumentList(call_arguments=[])

        procedure = self.fetch()
        for argument in procedure.arguments:
            if argument.default_value is None:
                assert any(
                    argument.name.upper() == call_argument.name.upper()
                    for call_argument in call_argument_list.call_arguments
                )

        result_or_future = self.collection._api.call_procedure(
            self.database.name,
            self.schema.name,
            procedure.name,
            call_argument_list=call_argument_list,
            async_req=async_req,
        )

        def mapper(r):
            return map_result(procedure, r, extract=extract)

        if isinstance(result_or_future, Future):
            return PollingOperation(result_or_future, mapper)
        return mapper(result_or_future)
