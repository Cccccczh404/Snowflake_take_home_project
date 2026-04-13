from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

from snowflake.core import PollingOperation
from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core.account._generated.models.account import AccountModel as Account

from ._generated.api.account_api_base import AccountCollectionBase, AccountResourceBase


if TYPE_CHECKING:
    from snowflake.core import Root


class AccountCollection(AccountCollectionBase):
    """Represents the collection operations of the Snowflake Account resource.

    With this collection, you can create, iterate through, and search for an account that you have access to
    in the current context.
    """

    def __init__(self, root: "Root") -> None:
        super().__init__(root, AccountResource)

    @api_telemetry
    def create(  # type: ignore[override]
        self,
        account: Account,
    ) -> "AccountResource":
        """Create an account in Snowflake.

        Parameters
        __________
        account: Account
            The ``Account`` object, together with the ``Account``'s properties:
            name, admin_name, email, edition; admin_password, first_name, last_name, must_change_password, region_group,
            region, comment, polaris are optional.

        """
        super().create(account=account._to_model())
        return AccountResource(account.name, self)

    @api_telemetry
    def create_async(  # type: ignore[override]
        self,
        account: Account,
    ) -> PollingOperation["AccountResource"]:
        """An asynchronous version of :func:`create`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        future = super().create_async(account=account._to_model())
        return PollingOperation(future, lambda _: AccountResource(account.name, self))

    @api_telemetry
    def iter(  # type: ignore[override]
        self,
        *,
        like: Optional[str] = None,
        limit: Optional[int] = None,
        history: Optional[bool] = None,
    ) -> Iterator[Account]:
        """Iterate through ``Account`` objects in Snowflake, filtering on any optional ``like`` pattern.

        Parameters
        __________
        like: str, optional
            A case-insensitive string functioning as a filter, with support for SQL
            wildcard characters (% and _).
        limit: int, optional
            Limit of the maximum number of rows returned by iter(). The default is ``None``, which behaves equivalently
            to show_limit=10000. This value must be between ``1`` and ``10000``.
        history: bool, optional
            If ``True``, includes dropped accounts that have not yet been deleted. The default is ``None``, which
            behaves equivalently to ``False``.
        """
        result = super().iter(
            like=like,
            limit=limit,
            history=history,
        )
        return (Account(**r.to_dict()) for r in result)

    @api_telemetry
    def iter_async(  # type: ignore[override]
        self,
        *,
        like: Optional[str] = None,
        limit: Optional[int] = None,
        history: Optional[bool] = None,
    ) -> PollingOperation[Iterator[Account]]:
        """An asynchronous version of :func:`iter`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        future = super().iter_async(
            like=like,
            limit=limit,
            history=history,
        )
        return PollingOperation(future, lambda acc: (Account(**a.to_dict()) for a in iter(acc)))


class AccountResource(AccountResourceBase):
    """Represents a reference to a Snowflake account.

    With this account reference, you can fetch information about accounts, as well
    as perform certain actions on them.
    """

    def __init__(self, name: str, collection: AccountCollection) -> None:
        super().__init__(name, collection)
