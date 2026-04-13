from typing import TYPE_CHECKING

from snowflake.core.pipe._generated.api.pipe_api_base import PipeCollectionBase, PipeResourceBase


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class PipeCollection(PipeCollectionBase):
    """Represents the collection operations of the Snowflake Pipe resource.

    With this collection, you can create, iterate through, and search for pipes that you have access to
    in the current context.

    Examples
    ________
    Creaing a pipe instance:

    >>> pipes = root.databases["my_db"].schemas["my_schema"].pipes
    >>>     new_pipe = Pipe(
    ...         name="my_pipe",
    ...         comment="This is a pipe")
    >>> pipes.create(new_pipe)
    """

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, PipeResource)


class PipeResource(PipeResourceBase):
    """Represents a reference to a Snowflake pipe.

    With this pipe reference, you can fetch information about pipes, as well
    as perform certain actions on them.
    """

    _plural_name = "pipes"
