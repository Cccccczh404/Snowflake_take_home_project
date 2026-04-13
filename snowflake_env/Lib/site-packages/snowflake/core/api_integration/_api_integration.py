from typing import TYPE_CHECKING

from pydantic import StrictStr

from snowflake.core._operation import PollingOperation, PollingOperations

from .._internal.telemetry import api_telemetry
from . import ApiIntegration
from ._generated.api.api_integration_api_base import ApiIntegrationCollectionBase, ApiIntegrationResourceBase


if TYPE_CHECKING:
    from snowflake.core import PollingOperation, Root


class ApiIntegrationCollection(ApiIntegrationCollectionBase):
    """Represents the collection operations on the Snowflake api integration resource.

    With this collection, you can create, iterate through, and search for api integration that you
    have access to in the current context.

    Examples
    ________
    Creating an ApiIntegration instance using AWS API Gateway:

    >>> api_integrations = root.api_integrations
    >>> new_api_integration = ApiIntegration(
    ...     name="name",
    ...     api_hook=AwsHook(
    ...         api_provider="AWS_API_GATEWAY",
    ...         api_aws_role_arn="your_arn",
    ...         api_key=os.environ.get("YOUR_API_KEY"),
    ...     ),
    ...     api_allowed_prefixes=["https://snowflake.com"],
    ...     enabled=True,
    ... )
    >>> api_integrations.create(new_api_integration)
    """

    def __init__(self, root: "Root"):
        super().__init__(root, ApiIntegrationResource)


class ApiIntegrationResource(ApiIntegrationResourceBase):
    """Represents a reference to a Snowflake api integration.

    With this api integration reference, you can create, update, delete and fetch information about
    api integrations, as well as perform certain actions on them.
    """

    _plural_name = "api_integrations"

    def __init__(self, name: StrictStr, collection: ApiIntegrationCollection) -> None:
        super().__init__(name, collection)

    @api_telemetry
    def create_or_alter(self, api_integration: ApiIntegration) -> None:
        """Create or alter an API integration.

        The operation is limited by the fact that api_key will not be updated
        and api_blocked_prefixes cannot be unset.

        Parameters
        __________
        api_integration: ApiIntegration
            The ``ApiIntegration`` object.

        Examples
        ________
        Creating a new API integration:

        >>> root.api_integrations["my_api"].create_or_alter(my_api_def)

        See ``ApiIntegrationCollection.create`` for more examples.
        """
        self.collection._api.create_or_alter_api_integration(
            api_integration.name, api_integration=api_integration, async_req=False
        )

    @api_telemetry
    def create_or_alter_async(self, api_integration: ApiIntegration) -> PollingOperation[None]:
        """An asynchronous version of :func:`create_or_alter`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        future = self.collection._api.create_or_alter_api_integration(
            api_integration.name, api_integration=api_integration, async_req=True
        )
        return PollingOperations.empty(future)
