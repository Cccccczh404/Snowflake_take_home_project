from concurrent.futures import Future
from typing import TYPE_CHECKING, Literal, Optional, Union, overload

from pydantic import StrictStr

from snowflake.core import FQN, PollingOperation
from snowflake.core._common import (
    Clone,
    CreateMode,
    PointOfTime,
)

from .._internal.telemetry import api_telemetry
from .._internal.utils import deprecated
from ._generated import SuccessResponse
from ._generated.api.dynamic_table_api_base import DynamicTableCollectionBase, DynamicTableResourceBase
from ._generated.models.dynamic_table import DynamicTable
from ._generated.models.dynamic_table_clone import DynamicTableClone
from ._generated.models.point_of_time import PointOfTime as TablePointOfTime


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class DynamicTableCollection(DynamicTableCollectionBase):
    """Represents the collection operations on the Snowflake Dynamic Table resource.

    With this collection, you can create, iterate through, and search for dynamic tables that you have access to in the
    current context.

    Examples
    ________
    Creating a dynamic table instance:

    >>> dynamic_tables = root.databases["my_db"].schemas["my_schema"].dynamic_tables
    >>> dynamic_tables.create(
    ...     DynamicTable(
    ...         name="my_dynamic_table",
    ...         columns=[
    ...             DynamicTableColumn(name="c1"),
    ...             DynamicTableColumn(name='"cc2"', datatype="varchar"),
    ...         ],
    ...         warehouse=db_parameters["my_warehouse"],
    ...         target_lag=UserDefinedLag(seconds=60),
    ...         query="SELECT * FROM my_table",
    ...     ),
    ...     mode=CreateMode.error_if_exists,
    ... )
    """

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, DynamicTableResource)

    @api_telemetry
    def create(  # type: ignore[override]
        self,
        table: Union[DynamicTable, DynamicTableClone, str],
        *,
        clone_table: Optional[Union[str, Clone]] = None,
        copy_grants: Optional[bool] = False,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> "DynamicTableResource":
        """Create a dynamic table.

        Parameters
        __________
        table: DynamicTable | DynamicTableClone | str
            1. The ``DynamicTable`` object, together with the dynamic table's properties:
                name, target_lag, warehouse, query;
                columns, refresh_mode, initialize, cluster_by, comment are optional.
            2. The ``DynamicTableClone`` object, when it's used with `clone_table`.
            3. The table name.
        clone_table: Clone, optional
            The source table to clone from.
        copy_grants: bool, optional
            Whether to enable copy grants when creating the object. Default is ``False``.
        mode: CreateMode, optional
            One of the following enum values.

            ``CreateMode.error_if_exists``: Throw an :class:`snowflake.core.exceptions.ConflictError`
            if the dynamic table already exists in Snowflake.  Equivalent to SQL ``create dynamic table <name> ...``.

            ``CreateMode.or_replace``: Replace if the dynamic table already exists in Snowflake. Equivalent to SQL
            ``create or replace dynamic table <name> ...``.

            ``CreateMode.if_not_exists``: Do nothing if the dynamic table already exists in Snowflake.
            Equivalent to SQL ``create dynamic table <name> if not exists...``

            Default is ``CreateMode.error_if_exists``.

        Examples
        ________
        Creating a dynamic table, replacing any existing dynamic table with the same name:

        >>> dynamic_tables = root.databases["my_db"].schemas["my_schema"].dynamic_tables
        >>> dynamic_tables.create(
        ...     DynamicTable(
        ...         name="my_dynamic_table",
        ...         columns=[
        ...             DynamicTableColumn(name="c1"),
        ...             DynamicTableColumn(name='"cc2"', datatype="varchar"),
        ...         ],
        ...         warehouse=db_parameters["my_warehouse"],
        ...         target_lag=UserDefinedLag(seconds=60),
        ...         query="SELECT * FROM my_table",
        ...     ),
        ...     mode=CreateMode.error_if_exists,
        ... )

        Creating a dynamic table by cloning an existing table:

        >>> dynamic_tables = root.databases["my_db"].schemas["my_schema"].dynamic_tables
        >>> dynamic_tables.create(
        ...     DynamicTableClone(
        ...         name="my_dynamic_table",
        ...         target_lag=UserDefinedLag(seconds=120),
        ...     ),
        ...     clone_table=Clone(
        ...         source="my_source_dynamic_table",
        ...         point_of_time=PointOfTimeOffset(reference="before", when="-1"),
        ...     ),
        ...     copy_grants=True,
        ... )

        Creating a dynamic table by cloning an existing table in a different database and schema:

        >>> dynamic_tables = root.databases["my_db"].schemas["my_schema"].dynamic_tables
        >>> dynamic_tables.create(
        ...     DynamicTableClone(
        ...         name="my_dynamic_table",
        ...         target_lag=UserDefinedLag(seconds=120),
        ...     ),
        ...     clone_table=Clone(
        ...         source="database_of_source_table.schema_of_source_table.my_source_dynamic_table",
        ...         point_of_time=PointOfTimeOffset(reference="before", when="-1"),
        ...     ),
        ...     copy_grants=True,
        ... )
        """
        self._create(
            table=table,
            clone_table=clone_table,
            copy_grants=copy_grants,
            mode=mode,
            async_req=False,
        )
        return DynamicTableResource(table if isinstance(table, str) else table.name, self)

    @api_telemetry
    def create_async(  # type: ignore[override]
        self,
        table: Union[DynamicTable, DynamicTableClone, str],
        *,
        clone_table: Optional[Union[str, Clone]] = None,
        copy_grants: Optional[bool] = False,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> PollingOperation["DynamicTableResource"]:
        """An asynchronous version of :func:`create`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        future = self._create(
            table=table,
            clone_table=clone_table,
            copy_grants=copy_grants,
            mode=mode,
            async_req=True,
        )
        return PollingOperation(
            future, lambda _: DynamicTableResource(table if isinstance(table, str) else table.name, self)
        )

    @overload
    def _create(
        self,
        table: Union[DynamicTable, DynamicTableClone, str],
        clone_table: Optional[Union[str, Clone]],
        copy_grants: Optional[bool],
        mode: CreateMode,
        async_req: Literal[True],
    ) -> Future[SuccessResponse]: ...

    @overload
    def _create(
        self,
        table: Union[DynamicTable, DynamicTableClone, str],
        clone_table: Optional[Union[str, Clone]],
        copy_grants: Optional[bool],
        mode: CreateMode,
        async_req: Literal[False],
    ) -> SuccessResponse: ...

    def _create(
        self,
        table: Union[DynamicTable, DynamicTableClone, str],
        clone_table: Optional[Union[str, Clone]],
        copy_grants: Optional[bool],
        mode: CreateMode,
        async_req: bool,
    ) -> Union[SuccessResponse, Future[SuccessResponse]]:
        real_mode = CreateMode[mode].value

        if clone_table:
            # create table by clone

            if isinstance(table, str):
                table = DynamicTableClone(name=table)

            pot: Optional[TablePointOfTime] = None
            if isinstance(clone_table, Clone) and isinstance(clone_table.point_of_time, PointOfTime):
                pot = TablePointOfTime.from_dict(clone_table.point_of_time.to_dict())
            real_clone = Clone(source=clone_table) if isinstance(clone_table, str) else clone_table
            req = DynamicTableClone(
                name=table.name,
                target_lag=table.target_lag,
                warehouse=table.warehouse,
                point_of_time=pot,
            )

            source_table_fqn = FQN.from_string(real_clone.source)
            return self._api.clone_dynamic_table(
                source_table_fqn.database or self.database.name,
                source_table_fqn.schema or self.schema.name,
                source_table_fqn.name,
                req,
                create_mode=StrictStr(real_mode),
                copy_grants=copy_grants,
                target_database=self.database.name,
                target_schema=self.schema.name,
                async_req=async_req,
            )

        # create empty table

        if not isinstance(table, DynamicTable):
            raise ValueError("`table` must be a `DynamicTable` unless `clone_table` is used")

        return self._api.create_dynamic_table(
            self.database.name,
            self.schema.name,
            table,
            create_mode=StrictStr(real_mode),
            async_req=async_req,
        )


class DynamicTableResource(DynamicTableResourceBase):
    """Represents a reference to a Snowflake dynamic table.

    With this dynamic table reference, you can create, drop, undrop, suspend, resume, swap_with other table,
    suspend recluster, resume recluster and fetch information about dynamic tables, as well
    as perform certain actions on them.
    """

    _plural_name = "dynamic_tables"

    @api_telemetry
    @deprecated("drop")
    def delete(self) -> None:
        self.drop()

    @api_telemetry
    @deprecated("undrop")
    def undelete(self) -> None:
        self.undrop()

    @api_telemetry
    def swap_with(  # type: ignore[override]
        self,
        to_swap_table_name: str,
        if_exists: Optional[bool] = None,
    ) -> None:
        """Swap the name with another dynamic table.

        Parameters
        __________
        to_swap_table_name: str
            The name of the table to swap with.
        if_exists: bool, optional
            Check the existence of this dynamic table before swapping its name.
            Default is ``None``, which is equivalent to ``False``.

        Examples
        ________
        Swaping name with another dynamic table using its reference:
        >>> dynamic_table_reference.swap_with("my_other_dynamic_table")
        """
        super().swap_with(
            target_name=to_swap_table_name,
            # target_schema=self.schema.name,
            # target_database=self.database.name,
            if_exists=if_exists,
        )

    @api_telemetry
    def swap_with_async(  # type: ignore[override]
        self,
        to_swap_table_name: str,
        if_exists: Optional[bool] = None,
    ) -> PollingOperation[None]:
        """An asynchronous version of :func:`swap_with`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        return super().swap_with_async(
            target_name=to_swap_table_name,
            # target_schema=self.schema.name,
            # target_database=self.database.name,
            if_exists=if_exists,
        )
