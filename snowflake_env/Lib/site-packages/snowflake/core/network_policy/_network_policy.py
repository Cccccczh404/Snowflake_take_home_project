from typing import TYPE_CHECKING

from ._generated.api.network_policy_api_base import NetworkPolicyCollectionBase, NetworkPolicyResourceBase


if TYPE_CHECKING:
    from snowflake.core._root import Root


class NetworkPolicyCollection(NetworkPolicyCollectionBase):
    """Represents the collection operations on the Snowflake Network Policy resource.

    With this collection, you can create, iterate through, and fetch network policies
    that you have access to in the current context.

    Examples
    ________
    Creating a network policy instance with only a single ip allowed:

    >>> network_policies = root.network_policies
    >>> new_network_policy = NetworkPolicy(
    ...     name="single_ip_policy", allowed_ip_list=["192.168.1.32/32"], blocked_ip_list=["0.0.0.0"]
    ... )
    >>> network_policies.create(new_network_policy)
    """

    def __init__(self, root: "Root"):
        super().__init__(root, NetworkPolicyResource)


class NetworkPolicyResource(NetworkPolicyResourceBase):
    """Represents a reference to a Snowflake Network Policy resource.

    With this network policy reference, you can create, update, and fetch information about network policies, as well
    as perform certain actions on them.
    """

    _plural_name = "network_policies"
