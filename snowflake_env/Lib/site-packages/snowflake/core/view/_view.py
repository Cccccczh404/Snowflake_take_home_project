from typing import TYPE_CHECKING

from pydantic import StrictStr

from ._generated.api.view_api_base import ViewCollectionBase, ViewResourceBase


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class ViewCollection(ViewCollectionBase):
    """Represents the collection operations on the Snowflake View resource.

    With this collection, you can create, iterate through, and search for views that you have access to in the
    current context.

    Examples
    ________
    Creating a view instance:

    >>> views = root.databases["my_db"].schemas["my_schema"].views
    >>> new_view = View(
    ...     name="my_view",
    ...     columns=[ViewColumn(name="col1"), ViewColumn(name="col2"), ViewColumn(name="col3")],
    ...     query="SELECT * FROM my_table",
    ... )
    >>> views.create(new_view)
    """

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, ViewResource)


class ViewResource(ViewResourceBase):
    """Represents a reference to a Snowflake view.

    With this view reference, you can drop and fetch information about views.
    """

    _plural_name = "views"

    def __init__(self, name: StrictStr, collection: ViewCollection) -> None:
        super().__init__(name, collection)
