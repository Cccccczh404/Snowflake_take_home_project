# mypy: ignore-errors

from typing import TYPE_CHECKING, Optional

from snowflake.core import PollingOperation

from .._internal.telemetry import api_telemetry
from ._generated.api.user_defined_function_api_base import (
    UserDefinedFunctionCollectionBase,
    UserDefinedFunctionResourceBase,
)


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class UserDefinedFunctionCollection(UserDefinedFunctionCollectionBase):
    """Represents the collection operations on the Snowflake User Defined Function resource.

    With this collection, you can create, iterate through, and fetch user defined functions
    that you have access to in the current context.

    Examples
    ________
    Creating a user defined function instance of python language:

    >>> user_defined_functions.create(
    ...     UserDefinedFunction(
    ...         name="my_python_function",
    ...         arguments=[],
    ...         return_type=ReturnDataType(datatype="VARIANT"),
    ...         language_config=PythonFunction(runtime_version="3.9", packages=[], handler="udf"),
    ...         body='''
    ... def udf():
    ...     return {"key": "value"}
    ...             ''',
    ...     )
    ... )
    """

    _identifier_requires_args = True

    def __init__(self, schema: "SchemaResource") -> None:
        super().__init__(schema, UserDefinedFunctionResource)


class UserDefinedFunctionResource(UserDefinedFunctionResourceBase):
    """Represents a reference to a Snowflake user defined function.

    With this user defined function reference, you can create, drop, rename
    and fetch information about user defined functions.
    """

    _identifier_requires_args = True
    _plural_name = "user_defined_functions"

    def __init__(self, name_with_args: str, collection_class: UserDefinedFunctionCollection) -> None:
        super().__init__(name_with_args, collection_class)

    @api_telemetry
    def rename(
        self,
        target_name: str,
        target_database: Optional[str] = None,
        target_schema: Optional[str] = None,
        if_exists: Optional[bool] = None,
    ) -> None:
        """Rename this user defined function.

        Parameters
        __________
        target_name: str
            The new name of the user defined function
        target_database: str, optional
            The database where the user defined function will be located
        target_schema: str, optional
            The schema where the user defined function will be located
        if_exists: bool, optional
            Check the existence of user defined function before rename

        Examples
        ________
        Renaming this user defined function using its reference:

        >>> user_defined_function_reference.rename("my_other_user_defined_function")

        Renaming this user defined function if it exists:

        >>> user_defined_function_reference.rename("my_other_user_defined_function", if_exists=True)

        Renaming this user defined function and relocating it to another schema within same database:

        >>> user_defined_function_reference.rename(
        ...     "my_other_user_defined_function", target_schema="my_other_schema", if_exists=True
        ... )

        Renaming this user defined function and relocating it to another database and schema:

        >>> user_defined_function_reference.rename(
        ...     "my_other_user_defined_function",
        ...     target_database="my_other_database",
        ...     target_schema="my_other_schema",
        ...     if_exists=True,
        ... )
        """
        if target_database is None:
            target_database = self.database.name
        if target_schema is None:
            target_schema = self.schema.name

        super().rename(
            target_name=target_name,
            target_database=target_database,
            target_schema=target_schema,
            if_exists=if_exists,
        )

    @api_telemetry
    def rename_async(
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
            target_name=target_name,
            target_database=target_database,
            target_schema=target_schema,
            if_exists=if_exists,
        )
