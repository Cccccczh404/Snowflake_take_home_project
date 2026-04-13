from typing import TYPE_CHECKING, Optional

from snowflake.core import PollingOperation
from snowflake.core._common import CreateMode
from snowflake.core._internal.telemetry import api_telemetry

from ._generated.api.notebook_api_base import NotebookCollectionBase, NotebookResourceBase
from ._generated.models.notebook import Notebook
from ._generated.models.version_details import VersionDetails  # noqa


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class NotebookCollection(NotebookCollectionBase):
    """Represents the collection operations of the Snowflake Notebook resource.

    With this collection, you can create, iterate through, and search for notebooks that you have access to
    in the current context.

    Examples
    ________
    Creating a notebook instance:

    >>> notebooks = root.databases["my_db"].schemas["my_schema"].notebooks
    >>> new_notebook = Notebook(name="my_notebook", comment="This is a notebook")
    >>> notebooks.create(new_notebook)
    """

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, NotebookResource)

    @api_telemetry
    def create(  # type: ignore[override]
        self,
        notebook: Notebook,
        *,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> "NotebookResource":
        """Create a notebook in Snowflake.

        Parameters
        __________
        notebook: Notebook
            The ``Notebook`` object that you want to create in Snowflake.
        mode: CreateMode, optional
            One of the following strings.

            ``CreateMode.error_if_exists``: Throw an :class:`snowflake.core.exceptions.ConflictError`
            if the notebook already exists in Snowflake. Equivalent to SQL ``create notebook <name> ...``.

            ``CreateMode.or_replace``: Replace if the notebook already exists in Snowflake. Equivalent to SQL
            ``create or replace notebook <name> ...``.

            ``CreateMode.if_not_exists``: Do nothing if the notebook already exists in Snowflake. Equivalent to SQL
            ``create notebook <name> if not exists...``

            Default value is ``CreateMode.error_if_exists``.

        Examples
        ________
        Creating a notebook in Snowflake and getting the reference to it:

        >>> notebook = Notebook(name="my_notebook", version="notebook_ver1", comment="This is a notebook")
        >>> # Use the notebook collection created before to create a reference to the notebook resource
        >>> # in Snowflake.
        >>> notebook_reference = notebook_collection.create(notebook)
        """
        super().create(
            notebook=notebook,
            mode=mode,
        )
        return NotebookResource(notebook.name, self)

    @api_telemetry
    def create_async(  # type: ignore[override]
        self,
        notebook: Notebook,
        *,
        mode: CreateMode = CreateMode.error_if_exists,
    ) -> PollingOperation["NotebookResource"]:
        """An asynchronous version of :func:`create`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        future = super().create_async(
            notebook=notebook,
            mode=mode,
        )
        return PollingOperation(future, lambda _: NotebookResource(notebook.name, self))


class NotebookResource(NotebookResourceBase):
    """Represents a reference to a Snowflake notebook.

    With this notebook reference, you can fetch information about notebooks, as well
    as perform certain actions on them: renaming, executing, committing, and managing
    versions.
    """

    _plural_name = "notebooks"

    @api_telemetry
    def rename(  # type: ignore[override]
        self,
        target_name: str,
        target_database: Optional[str] = None,
        target_schema: Optional[str] = None,
        if_exists: Optional[bool] = None,
    ) -> None:
        """Rename this notebook.

        Parameters
        __________
        target_name: str
            The new name of the notebook
        target_database: str, optional
            The new database name of the notebook. If not provided,
            the current database name is used. The default is ``None``.
        target_schema: str, optional
            The new schema name of the notebook. If not provided,
            the current schema name is used. The default is ``None``.
        if_exists: bool, optional
            Whether to check for the existence of notebook before
            renaming. The default is ``None``, which is equivalent to ``False``.

        Examples
        ________
        Renaming this notebook using its reference:

        >>> notebook_reference.rename("my_other_notebook")

        Renaming this notebook if it exists:

        >>> notebook_reference.rename("my_other_notebook", if_exists=True)
        """
        if target_database is None:
            target_database = self.database.name

        if target_schema is None:
            target_schema = self.schema.name
        super().rename(
            target_database=target_database,
            target_schema=target_schema,
            target_name=target_name,
            if_exists=if_exists,
        )

    @api_telemetry
    def rename_async(  # type: ignore[override]
        self,
        target_name: str,
        target_database: Optional[str] = None,
        target_schema: Optional[str] = None,
        if_exists: Optional[bool] = None,
    ) -> PollingOperation[None]:
        """An asynchronous version of :func:`rename`.

        Refer to :class:`~snowflake.core.PollingOperation` for more information on asynchronous execution and
        the return type.
        """  # noqa: D401
        if target_database is None:
            target_database = self.database.name

        if target_schema is None:
            target_schema = self.schema.name
        return super().rename_async(
            target_database=target_database,
            target_schema=target_schema,
            target_name=target_name,
            if_exists=if_exists,
        )
