from typing import TYPE_CHECKING

from snowflake.core.external_volume._generated.api.external_volume_api_base import (
    ExternalVolumeCollectionBase,
    ExternalVolumeResourceBase,
)


if TYPE_CHECKING:
    from snowflake.core import Root


class ExternalVolumeCollection(ExternalVolumeCollectionBase):
    """Represents the collection operations of the Snowflake External Volume resource.

    With this collection, you can create, iterate through, and search for external volume that you have access to
    in the current context.

    Examples
    ________
    Creating an external volume instance:

    >>> external_volume_collection = root.external_volumes
    >>> external_volume = ExternalVolume(
    ...     name="MY_EXTERNAL_VOLUME",
    ...     storage_location=StorageLocationS3(
    ...         name="abcd-my-s3-us-west-2",
    ...         storage_base_url="s3://MY_EXAMPLE_BUCKET/",
    ...         storage_aws_role_arn="arn:aws:iam::123456789022:role/myrole",
    ...         encryption=Encryption(
    ...             type="AWS_SSE_KMS", kms_key_id="1234abcd-12ab-34cd-56ef-1234567890ab"
    ...         ),
    ...     ),
    ...     comment="This is my external volume",
    ... )
    >>> external_volume_collection.create(external_volume)
    """

    def __init__(self, root: "Root") -> None:
        super().__init__(root, ExternalVolumeResource)


class ExternalVolumeResource(ExternalVolumeResourceBase):
    """Represents a reference to a Snowflake external volume.

    With this external volume reference, you can fetch information about external volumes, as well
    as perform certain actions on them.
    """

    _plural_name = "external_volumes"
