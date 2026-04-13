from typing import TYPE_CHECKING

from ._generated.api.notification_integration_api_base import (
    NotificationIntegrationCollectionBase,
    NotificationIntegrationResourceBase,
)


if TYPE_CHECKING:
    from snowflake.core._root import Root


class NotificationIntegrationCollection(NotificationIntegrationCollectionBase):
    """Represents the collection operations on the Snowflake Notification Integration resource.

    With this collection, you can create, update, and iterate through notification integrations that you have access
    to in the current context.

    Examples
    ________
    Creating a notification integrations instance:

    >>> # This example assumes that mySecret already exists
    >>> notification_integrations = root.notification_integrations
    >>> new_ni = NotificationIntegration(
    ...     name="my_notification_integration",
    ...     enabled=True,
    ...     notification_hook=NotificationWebhook(
    ...         webhook_url="https://events.pagerduty.com/v2/enqueue",
    ...         webhook_secret=WebhookSecret(
    ...             name="mySecret".upper(), database_name=database, schema_name=schema
    ...         ),
    ...         webhook_body_template='{"key": "SNOWFLAKE_WEBHOOK_SECRET", "msg": "SNOWFLAKE_WEBHOOK_MESSAGE"}',
    ...         webhook_headers={"content-type": "application/json", "user-content": "chrome"},
    ...     ),
    ... )
    >>> notification_integrations.create(new_ni)
    """

    def __init__(self, root: "Root") -> None:
        super().__init__(root, NotificationIntegrationResource)


class NotificationIntegrationResource(NotificationIntegrationResourceBase):
    """Represents a reference to a Snowflake Notification Integration resource.

    With this notification integration reference you can delete, and fetch information about them.
    """

    _plural_name = "notification_integrations"
