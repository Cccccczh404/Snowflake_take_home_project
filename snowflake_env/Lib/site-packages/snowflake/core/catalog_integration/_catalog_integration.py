from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core._operation import PollingOperation

from . import CatalogIntegration
from ._generated.api.catalog_integration_api_base import (
    CatalogIntegrationCollectionBase,
    CatalogIntegrationResourceBase,
)


if TYPE_CHECKING:
    from snowflake.core._root import Root


class CatalogIntegrationCollection(CatalogIntegrationCollectionBase):
    """Represents the collection operations on the Snowflake Catalog Integration resource.

    With this collection, you can create, iterate through, and fetch catalog integrations
    that you have access to in the current context.

    Examples
    ________
    Creating a catalog integration instance by glue:

    >>> root.catalog_integrations.create(
    ...     CatalogIntegration(
    ...         name="my_catalog_integration",
    ...         catalog=Glue(
    ...             catalog_namespace="abcd-ns",
    ...             glue_aws_role_arn="arn:aws:iam::123456789012:role/sqsAccess",
    ...             glue_catalog_id="1234567",
    ...         ),
    ...         table_format="ICEBERG",
    ...         enabled=True,
    ...     )
    ... )

    Creating a catalog integration instance by object store:

    >>> root.catalog_integrations.create(
    ...     CatalogIntegration(
    ...         name="my_catalog_integration",
    ...         catalog=ObjectStore(),
    ...         table_format="ICEBERG",
    ...         enabled=True,
    ...     )
    ... )

    Creating a catalog integration instance by polaris:

    >>> root.catalog_integrations.create(
    ...     CatalogIntegration(
    ...         name="my_catalog_integration",
    ...         catalog=Polaris(
    ...             catalog_namespace="abcd-ns",
    ...             rest_config=RestConfig(
    ...                 catalog_uri="https://my_account.snowflakecomputing.com/polaris/api/catalog",
    ...                 warehouse_name="my_warehouse",
    ...             ),
    ...             rest_authenticator=OAuth(
    ...                 type="OAUTH",
    ...                 oauth_client_id="my_oauth_client_id",
    ...                 oauth_client_secret="my_oauth_client_secret",
    ...                 oauth_allowed_scopes=["PRINCIPAL_ROLE:ALL"],
    ...             ),
    ...         ),
    ...         table_format="ICEBERG",
    ...         enabled=True,
    ...     )
    ... )
    """

    def __init__(self, root: "Root"):
        super().__init__(root, CatalogIntegrationResource)

    @api_telemetry
    def iter(self, like: Optional[str] = None) -> Iterator[CatalogIntegration]:
        """Iterate through ``CatalogIntegration`` objects from Snowflake.

        Parameters
        __________
        like: str, optional
            A case-insensitive string functioning as a filter, with support for SQL
            wildcard characters (% and _).
        """
        return super().iter(like=like)

    @api_telemetry
    def iter_async(self, like: Optional[str] = None) -> PollingOperation[Iterator[CatalogIntegration]]:
        """An asynchronous version of :func:`iter`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        return super().iter_async(like=like)


class CatalogIntegrationResource(CatalogIntegrationResourceBase):
    """Represents a reference to a Snowflake Catalog Integration resource.

    With this catalog integration reference, you can create, update, and fetch information about catalog integrations,
    as well as perform certain actions on them.
    """

    _plural_name = "catalog_integrations"
