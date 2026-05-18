"""FastAPI app for the worker-control-plane.

Endpoints:
- `GET  /status`            — capability + load snapshot (advertised view).
- `POST /register`          — submit operator-registration extrinsic.
- `POST /heartbeat-now`     — force-send one heartbeat (returns the sent body).
- `POST /internal/sources`  — workers POST their capability snapshots here.
- `GET  /healthz`

Security:
- Every non-healthz route is gated by `INTERNAL_AUTH_TOKEN` bearer.
- The operator's private key never leaves this process; it's wrapped in
  `pydantic.SecretStr` (HIGH-SVC-009) and is only retrieved via
  `config.operator_private_key()` from the signer path.
- Health endpoint MUST NOT leak any field containing `private` in its name.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from worker_control_plane.auth import require_internal_auth, require_internal_token
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
from worker_control_plane.heartbeat import HeartbeatPusher


class RegisterRequest(BaseModel):
    advertised_models: list[str] = []
    nonce: int = 0


def _assert_no_secret_leakage(payload: dict[str, Any]) -> None:
    """Defence-in-depth: refuse to ever serialize a 'private' field in responses."""
    for k in payload:
        if "private" in k.lower():
            raise RuntimeError(f"refusing to expose field {k!r} in response")


def build_app(
    config: ControlPlaneConfig,
    registry: RegistryClient | None = None,
    *,
    autostart_heartbeat: bool = False,
) -> FastAPI:
    require_internal_token()

    reg_client: RegistryClient = registry or MockRegistryClient()

    # Sources are pushed in by the workers via /internal/sources. The control
    # plane owns the merge.
    sources: dict[str, CapabilitySource] = {}

    def _provider() -> list[CapabilitySource]:
        return list(sources.values())

    pusher = HeartbeatPusher(config, _provider)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        if autostart_heartbeat:
            pusher.start()
        try:
            yield
        finally:
            await pusher.stop()

    app = FastAPI(title="worker-control-plane", version="0.1.0", lifespan=lifespan)
    allowed_hosts = [
        h.strip()
        for h in os.environ.get("ALLOWED_HOSTS", "*").split(",")
        if h.strip()
    ] or ["*"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    app.state.sources = sources
    app.state.pusher = pusher
    app.state.registry = reg_client
    app.state.config = config

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        body = {
            "ok": True,
            "operator_id": config.operator_id,
            "sources": len(sources),
            "heartbeats_sent": pusher.sent_count,
        }
        _assert_no_secret_leakage(body)
        return body

    @app.get("/status", dependencies=[Depends(require_internal_auth)])
    async def status() -> dict[str, Any]:
        snapshots = list(sources.values())
        caps = [c.model_dump() for c in merge_capabilities(snapshots)]
        load = snapshot_load(snapshots).model_dump()
        body = {
            "operator_id": config.operator_id,
            "capabilities": caps,
            "load": load,
            "registered": reg_client.is_registered(config.operator_id),
        }
        _assert_no_secret_leakage(body)
        return body

    @app.post("/internal/sources", dependencies=[Depends(require_internal_auth)])
    async def update_source(src: CapabilitySource) -> dict[str, Any]:
        sources[src.source_id] = src
        return {"ok": True, "source_id": src.source_id, "total_sources": len(sources)}

    @app.post("/register", dependencies=[Depends(require_internal_auth)])
    async def register(req: RegisterRequest) -> dict[str, Any]:
        if reg_client.is_registered(config.operator_id):
            raise HTTPException(
                status_code=409, detail=f"operator {config.operator_id} already registered",
            )
        models = req.advertised_models or [
            c.base_model_id for c in merge_capabilities(list(sources.values()))
        ]
        reg = OperatorRegistration(
            operator_id=config.operator_id,
            operator_public_key_hex=config.operator_public_key_hex,
            attestation_report_hash=config.attestation_report_hash,
            geo_region=config.geo_region,
            endpoint_url=config.endpoint_url,
            stake_amount=config.stake_amount,
            commission_bps=config.commission_bps,
            advertised_models=models,
            nonce=req.nonce,
        ).sign(config.operator_private_key())
        try:
            tx_hash = reg_client.submit(reg)
        except RegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        body = {
            "ok": True,
            "tx_hash": tx_hash,
            "extrinsic_hash": reg.extrinsic_hash(),
            "registration": reg.model_dump(),
        }
        _assert_no_secret_leakage(body)
        return body

    @app.post("/heartbeat-now", dependencies=[Depends(require_internal_auth)])
    async def heartbeat_now() -> dict[str, Any]:
        hb = await pusher.push_once()
        body = {
            "ok": True,
            "sent_count": pusher.sent_count,
            "error_count": pusher.error_count,
            "heartbeat": hb.model_dump(mode="json"),
        }
        _assert_no_secret_leakage(body)
        return body

    return app
