from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

from snowflake.core import PollingOperation
from snowflake.core._common import CreateMode
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core._internal.utils import deprecated
from snowflake.core.compute_pool._generated.models.compute_pool import ComputePoolModel as ComputePool

from ._generated.api.compute_pool_api_base import ComputePoolCollectionBase, ComputePoolResourceBase


if TYPE_CHECKING:
    from snowflake.core import Root


class ComputePoolCollection(ComputePoolCollectionBase):
    """Represents the collection operations on the Snowflake Compute Pool resource.

    With this collection, you can create, iterate through, and search for compute pools that you have access to
    in the current context.

    Examples
    ________
    Creating a compute pool instance:

    >>> compute_pool = ComputePool(
    ...     name="my_compute_pool", instance_family="CPU_X64_XS", min_nodes=1, max_nodes=2
    ... )
    >>> compute_pool_reference = root.compute_pools.create(compute_pool)
    """

    def __init__(self, root: "Root") -> None:
        super().__init__(root, ComputePoolResource)

    @api_telemetry
    def create(  # type: ignore[override]
        self,
        compute_pool: ComputePool,
        *,
        mode: CreateMode = CreateMode.error_if_exists,
        initially_suspended: bool = False,
    ) -> "ComputePoolResource":
        """Create a compute pool in Snowflake.

        Parameters
        __________
        compute_pool: ComputePool
            The ``ComputePool`` object, together with the ``ComputePool``'s properties:
            name, min_nodes, max_nodes, instance_family;
            auto_resume, initially_suspended, auto_suspend_secs, comment are optional
        mode: CreateMode, optional
            One of the below enum values.

            ``CreateMode.error_if_exists``: Throw an :class:`snowflake.core.exceptions.ConflictError` if the compute
            pool already exists in Snowflake. Equivalent to SQL ``create compute pool <name> ...``.

            ``CreateMode.if_not_exists``: Do nothing if the compute pool already exists in Snowflake. Equivalent to SQL
            ``create compute pool <name> if not exists...``.

            Default value is ``CreateMode.error_if_exists``.
        initially_suspended: bool, optional
            Determines if the compute pool should be suspended when initially created. Default value is False.

        Examples
        ________
        Creating a compute pool, replacing an existing compute pool with the same name:

        >>> compute_pool = ComputePool(
        ...     name="my_compute_pool", instance_family="CPU_X64_XS", min_nodes=1, max_nodes=2
        ... )
        >>> compute_pool_reference = compute_pools.create(compute_pool, mode=CreateMode.or_replace)

        Creating a compute pool that is initially suspended:

        >>> compute_pool = ComputePool(
        ...     name="my_compute_pool", instance_family="CPU_X64_XS", min_nodes=1, max_nodes=5
        ... )
        >>> compute_pool_reference = compute_pools.create(compute_pool, initially_suspended=True)
        """
        super().create(
            compute_pool=compute_pool._to_model(),
            mode=mode,
            initially_suspended=initially_suspended,
        )
        return self[compute_pool.name]

    @api_telemetry
    def create_async(  # type: ignore[override]
        self,
        compute_pool: ComputePool,
        *,
        mode: CreateMode = CreateMode.error_if_exists,
        initially_suspended: bool = False,
    ) -> PollingOperation["ComputePoolResource"]:
        """An asynchronous version of :func:`create`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        future = super().create_async(
            compute_pool=compute_pool._to_model(),
            mode=mode,
            initially_suspended=initially_suspended,
        )
        return PollingOperation(future, lambda _: self[compute_pool.name])

    @api_telemetry
    def iter(  # type: ignore[override]
        self,
        *,
        like: Optional[str] = None,
        starts_with: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[ComputePool]:
        """Iterate through ``Compute Pool`` objects in Snowflake, filtering on any optional 'like' pattern.

        Parameters
        __________
        like: str, optional
            A case-insensitive string functioning as a filter, with support for SQL
            wildcard characters (% and _).
        starts_with: str, optional
            String used to filter the command output based on the string of characters that appear
            at the beginning of the object name. Uses case-sensitive pattern matching.
        limit: int, optional
            Limit of the maximum number of rows returned by iter(). The default is ``None``, which behaves equivalently
            to show_limit=10000. This value must be between ``1`` and ``10000``.

        Examples
        ________
        Showing all compute pools that you have access to see:

        >>> compute_pools = root.compute_pools.iter()

        Showing information of the exact compute pool you want to see:

        >>> compute_pools = root.compute_pools.iter(like="your-compute-pool-name")

        Showing compute pools starting with 'your-compute-pool-name-':

        >>> compute_pools = root.compute_pools.iter(like="your-compute-pool-name-%")

        Using a for loop to retrieve information from iterator:

        >>> for compute_pool in compute_pools:
        >>>     print(compute_pool.name)
        """
        compute_pools = super().iter(
            like=like,
            starts_with=starts_with,
            limit=limit,
        )
        return map(ComputePool._from_model, iter(compute_pools))

    @api_telemetry
    def iter_async(  # type: ignore[override]
        self,
        *,
        like: Optional[str] = None,
        starts_with: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> PollingOperation[Iterator[ComputePool]]:
        """An asynchronous version of :func:`iter`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        future = super().iter_async(
            like=like,
            starts_with=starts_with,
            limit=limit,
        )
        return PollingOperation(future, lambda rest_models: map(ComputePool._from_model, iter(rest_models)))


class ComputePoolResource(ComputePoolResourceBase):
    """Represents a reference to a Snowflake compute pool.

    With this compute pool reference, you can create and fetch information about compute pools, as well as
    perform certain actions on them.
    """

    _plural_name = "compute_pools"

    @api_telemetry
    def create_or_alter(self, compute_pool: ComputePool) -> None:  # type: ignore[override]
        """Create a compute pool in Snowflake or alter one if it already exists.

        Parameters
        __________
        compute_pool: ComputePool
            An instance of :class:`ComputePool`.

        Examples
        ________
        Creating or updating a compute pool in Snowflake:

        >>> cp_parameters = ComputePool(
        ...     name="your-cp-name",
        ...     instance_family="CPU_X64_XS",
        ...     min_nodes=1,
        ...     max_nodes=1,
        ...)

        # Using a ``ComputePoolCollection`` to create or update a compute pool in Snowflake:
        >>> root.compute_pools["your-cp-name"].create_or_alter(cp_parameters)
        """
        super().create_or_alter(compute_pool._to_model())

    @api_telemetry
    def create_or_alter_async(self, compute_pool: ComputePool) -> PollingOperation[None]:  # type: ignore[override]
        """An asynchronous version of :func:`create_or_alter`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        return super().create_or_alter_async(compute_pool._to_model())

    @api_telemetry
    @deprecated("drop")
    def delete(self) -> None:
        self.drop()

    @api_telemetry
    def fetch(  # type: ignore[override]
        self,
    ) -> ComputePool:
        """Fetch a compute pool.

        Parameters
        __________
        """
        return ComputePool(**super().fetch().to_dict())

    @api_telemetry
    def fetch_async(  # type: ignore[override]
        self,
    ) -> PollingOperation[ComputePool]:
        """An asynchronous version of :func:`fetch`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        future = self.collection._api.fetch_compute_pool(
            self.name,
            async_req=True,
        )

        return PollingOperation(future, lambda rest_model: ComputePool(**rest_model.to_dict()))
