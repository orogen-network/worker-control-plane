"""Operator registration helper for RFC-0009.

Builds the `pallet-operator-registry::register_operator` extrinsic body and submits
through a `RegistryClient` Protocol. The default `MockRegistryClient` keeps registrations
in-memory and is fine for dev / tests; production wires substrate-interface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from mining_types.crypto import blake2_256, canonical_json, sign_ed25519
from pydantic import BaseModel, Field


class OperatorRegistration(BaseModel):
    """RFC-0009 §3 — extrinsic body for `register_operator`."""

    operator_id: str
    operator_public_key_hex: str
    attestation_report_hash: str
    geo_region: str
    endpoint_url: str
    stake_amount: int
    commission_bps: int
    advertised_models: list[str] = Field(default_factory=list)
    nonce: int = 0
    signature: str = ""

    def signing_payload(self) -> bytes:
        d = self.model_dump(mode="json")
        d.pop("signature", None)
        return canonical_json(d)

    def sign(self, operator_private_key_hex: str) -> OperatorRegistration:
        sig = sign_ed25519(operator_private_key_hex, self.signing_payload())
        return self.model_copy(update={"signature": sig})

    def extrinsic_hash(self) -> str:
        return blake2_256(self.signing_payload())


class RegistryError(RuntimeError):
    """Raised when registration is rejected by the chain client."""


class RegistryClient(Protocol):
    def submit(self, reg: OperatorRegistration) -> str: ...
    def is_registered(self, operator_id: str) -> bool: ...


@dataclass
class MockRegistryClient:
    """In-memory chain stub. Deterministic tx hash; rejects duplicate registration."""

    _registered: dict[str, OperatorRegistration] = field(default_factory=dict)
    submissions: list[tuple[str, OperatorRegistration]] = field(default_factory=list)

    def submit(self, reg: OperatorRegistration) -> str:
        if not reg.signature:
            raise RegistryError("registration is unsigned")
        if reg.operator_id in self._registered:
            raise RegistryError(f"operator {reg.operator_id!r} already registered")
        if reg.stake_amount <= 0:
            raise RegistryError("stake_amount must be > 0")
        tx_hash = hashlib.sha256(reg.signing_payload()).hexdigest()
        self._registered[reg.operator_id] = reg
        self.submissions.append((tx_hash, reg))
        return tx_hash

    def is_registered(self, operator_id: str) -> bool:
        return operator_id in self._registered
