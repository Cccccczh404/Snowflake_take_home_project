from os import PathLike, fspath
from typing import TYPE_CHECKING, Optional, Union

from snowflake.core._internal.telemetry import api_telemetry
from snowflake.core._internal.utils import deprecated, get_file, put_file
from snowflake.core.stage._generated.api.stage_api_base import StageCollectionBase, StageResourceBase


if TYPE_CHECKING:
    from snowflake.core.schema import SchemaResource


class StageCollection(StageCollectionBase):
    """Represents the collection operations on the Snowflake Stage resource.

    With this collection, you can create, iterate through, and fetch stages
    that you have access to in the current context.

    Examples
    ________
    Creating a stage instance:

    >>> stages = root.databases["my_db"].schemas["my_schema"].stages
    >>> new_stage = Stage(name="my_stage", comment="This is a stage")
    >>> stages.create(new_stage)
    """

    def __init__(self, schema: "SchemaResource"):
        super().__init__(schema, StageResource)


class StageResource(StageResourceBase):
    """Represents a reference to a Snowflake stage.

    With this stage reference, you can drop, list files, put files, get files,
    and fetch information about stages.
    """

    _plural_name = "stages"

    def __init__(self, name: str, collection: StageCollection) -> None:
        super().__init__(name, collection)

    @api_telemetry
    @deprecated("drop")
    def delete(self) -> None:
        self.drop()

    @api_telemetry
    @deprecated("put")
    def upload_file(
        self, file_path: str, stage_folder_path: str, *, auto_compress: bool = True, overwrite: bool = False
    ) -> None:
        self.put(file_path, stage_folder_path, auto_compress=auto_compress, overwrite=overwrite)

    @api_telemetry
    @deprecated("get")
    def download_file(self, stage_path: str, file_folder_path: str) -> None:
        self.get(stage_path, file_folder_path)

    @api_telemetry
    def put(
        self,
        local_file_name: Union[str, PathLike],  # type: ignore[type-arg]
        stage_location: str,
        *,
        parallel: int = 4,
        auto_compress: bool = True,
        source_compression: str = "AUTO_DETECT",
        overwrite: bool = False,
    ) -> None:
        """Upload local files to a path in the stage.

        References: `Snowflake PUT command <https://docs.snowflake.com/en/sql-reference/sql/put.html>`_.

        Parameters
        __________
        local_file_name: str, ``PathLike``
            The path to the local files to upload. To match multiple files in the path,
            you can specify the wildcard characters ``*`` and ``?``.
        stage_location: str
            The prefix where you want to upload the files. e.g. ``/folder`` or ``/``
        parallel: int, optional
            Specifies the number of threads to use for uploading files. The upload process separates batches
            of data files by size:

            * Small files (< 64 MB) are staged in parallel as individual files.

            * Larger files are automatically split into chunks, staged concurrently, and reassembled
              in the target stage. A single thread can upload multiple chunks.

            Increasing the number of threads can improve performance when uploading large files.
            Supported values: Any integer value from 1 (no parallelism) to 99 (use 99 threads for uploading files).
        auto_compress: boolean, optional
            Specifies whether Snowflake uses gzip to compress files during upload.
            Default is ``True``.
        source_compression: str, optional
            Specifies the method of compression used on already-compressed files that are being staged.

            Values can be ``AUTO_DETECT``, ``GZIP``, ``BZ2``, ``BROTLI``, ``ZSTD``, ``DEFLATE``, ``RAW_DEFLATE``,
            ``NONE``, default is ``AUTO_DETECT``.
        overwrite: boolean, optional
            Specifies whether Snowflake will overwrite an existing file with the same name during upload.
            Default is ``False``.

        Examples
        ________
        Putting file on stage and compressing it using the stage's reference:

        >>> stage_reference.put("local_file.csv", "/folder", auto_compress=True)
        """
        stage_name = f"@{self.database.name}.{self.schema.name}.{self.name}"
        norm_stage_path = stage_location if stage_location.startswith("/") else f"/{stage_location}"
        put_file(
            self.collection.root,
            fspath(local_file_name),
            f"{stage_name}{norm_stage_path}",
            parallel=parallel,
            auto_compress=auto_compress,
            source_compression=source_compression,
            overwrite=overwrite,
        )

    @api_telemetry
    def get(
        self,
        stage_location: str,
        target_directory: Union[str, PathLike],  # type: ignore[type-arg]
        *,
        parallel: int = 4,
        pattern: Optional[str] = None,
    ) -> None:
        """Download the specified files from a path in the stage to a local directory.

        References: `Snowflake GET command <https://docs.snowflake.com/en/sql-reference/sql/get.html>`_.

        Parameters
        __________
        stage_location: str
            A directory or filename on a stage, from which you want to download the files.
            e.g. ``/folder/file_name.txt`` or ``/folder``
        target_directory: str, ``PathLike``
            The path to the local directory where the files should be downloaded.
            If ``target_directory`` does not already exist, the method creates the directory.
        parallel: int, optional
            Specifies the number of threads to use for downloading the files.
            The granularity unit for downloading is one file.
            Increasing the number of threads might improve performance when downloading large files.
            Valid values: Any integer value from 1 (no parallelism) to 99 (use 99 threads for downloading files).
        pattern: str, optional
            Specifies a regular expression pattern for filtering files to download.
            The command lists all files in the specified path and applies the regular expression pattern on each of
            the files found.
            Default: ``None`` (all files in the specified stage are downloaded).

        Examples
        ________
        Getting file from stage:

        >>> stage_reference.get("/folder/file_name.txt", "/local_folder")

        Getting files with a specific pattern:

        >>> stage_reference.get("/folder", "/local_folder", pattern=".*.txt")
        """
        stage_name = f"@{self.database.name}.{self.schema.name}.{self.name}"
        norm_stage_path = stage_location if stage_location.startswith("/") else f"/{stage_location}"
        get_file(
            self.collection.root,
            f"{stage_name}{norm_stage_path}",
            fspath(target_directory),
            parallel=parallel,
            pattern=pattern,
        )
