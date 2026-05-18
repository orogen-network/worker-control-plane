"""Worker control plane.

Implements RFC-0003 (off-chain heartbeat) and RFC-0009 (operator registration)
helpers. Sits alongside any of the `infer-worker-*` daemons and centralises:

- heartbeat sender (push every 12s to gateway)
- capability merge across multiple workers on the same host
- chain registration extrinsic builder + mock submission
"""

from worker_control_plane.app import build_app
from worker_control_plane.capability import (
    CapabilitySource,
    merge_capabilities,
    snapshot_load,
)
from worker_control_plane.chain import (
    MockRegistryClient,
    OperatorRegistration,
    RegistryClient,
    RegistryError,
)
from worker_control_plane.config import ControlPlaneConfig
from worker_control_plane.heartbeat import HeartbeatPusher, build_heartbeat

__version__ = "0.1.0"

__all__ = [
    "CapabilitySource",
    "ControlPlaneConfig",
    "HeartbeatPusher",
    "MockRegistryClient",
    "OperatorRegistration",
    "RegistryClient",
    "RegistryError",
    "build_app",
    "build_heartbeat",
    "merge_capabilities",
    "snapshot_load",
]
