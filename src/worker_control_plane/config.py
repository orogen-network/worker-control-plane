"""Control-plane configuration.

The `operator_private_key_hex` is wrapped in `pydantic.SecretStr` so it never
appears in log lines, `repr()`, or accidental JSON serializations of the config.
Use `.get_secret_value()` to obtain the underlying hex string when signing.
(HIGH-SVC-009.)
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import SecretStr


def _as_secret(value: str | SecretStr) -> SecretStr:
    return value if isinstance(value, SecretStr) else SecretStr(value)


@dataclass
class ControlPlaneConfig:
    operator_id: str
    operator_private_key_hex: SecretStr
    operator_public_key_hex: str
    gateway_ws_url: str
    attestation_report_hash: str
    geo_region: str = "US"
    heartbeat_interval_s: float = 12.0
    price_per_million_tokens: int = 2_000_000
    endpoint_url: str = ""
    # On-chain operator stake amount declared at registration (denominated in CUC).
    stake_amount: int = 5000_000_000
    # Optional: bond commission in bps (1000 = 10%).
    commission_bps: int = 1000

    def __post_init__(self) -> None:
        # Allow plain-str input for ergonomics; coerce to SecretStr.
        object.__setattr__(
            self,
            "operator_private_key_hex",
            _as_secret(self.operator_private_key_hex),
        )

    def operator_private_key(self) -> str:
        """Return the raw hex private key — only ever call this from a signer."""
        return self.operator_private_key_hex.get_secret_value()
