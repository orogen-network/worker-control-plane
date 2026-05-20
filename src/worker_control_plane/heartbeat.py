"""Heartbeat sender — runs on the control plane and pushes for the whole host."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable

import httpx
from mining_types import (
    AttestationFreshness,
    Capability,
    LoadSnapshot,
    OffChainHeartbeat,
    WatchdogState,
)

from worker_control_plane.capability import (
    CapabilitySource,
    merge_capabilities,
    snapshot_load,
)
from worker_control_plane.config import ControlPlaneConfig

CapabilityProvider = Callable[[], list[CapabilitySource]]


def _gateway_auth_headers(config: ControlPlaneConfig) -> dict[str, str]:
    token = (
        config.gateway_auth_token
        or os.environ.get("GATEWAY_INTERNAL_AUTH_TOKEN", "")
        or os.environ.get("INTERNAL_AUTH_TOKEN", "")
    ).strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def build_heartbeat(
    config: ControlPlaneConfig,
    sources: list[CapabilitySource],
    *,
    block_number: int = 0,
    last_completed_job_id: str | None = None,
) -> OffChainHeartbeat:
    caps: list[Capability] = merge_capabilities(sources)
    load: LoadSnapshot = snapshot_load(sources)
    now_ms = int(time.time() * 1000)
    hb = OffChainHeartbeat(
        operator_id=config.operator_id,
        block_number=block_number,
        capabilities=caps,
        current_load=load,
        kv_cache_pressure=load.gpu_utilization_pct / 100.0,
        last_completed_job_id=last_completed_job_id,
        attestation_freshness=AttestationFreshness(
            last_attested_at_ms=now_ms,
            expires_at_ms=now_ms + 7 * 86400 * 1000,
            current_report_hash=config.attestation_report_hash,
        ),
        watchdog_state=WatchdogState(vllm_pid_alive=True, vllm_last_log_ms=now_ms),
        endpoint_url=config.endpoint_url,
        price_per_million_tokens=config.price_per_million_tokens,
        geo_region=config.geo_region,
    )
    return hb.sign(config.operator_private_key())


class HeartbeatPusher:
    """Sends merged heartbeats every `interval_s` until stopped.

    The gateway URL is treated as HTTP for tests; in production this becomes a
    persistent WebSocket — the upgrade path is the same payload over the same path.
    """

    def __init__(
        self,
        config: ControlPlaneConfig,
        provider: CapabilityProvider,
        *,
        interval_s: float | None = None,
        gateway_http_path: str = "/internal/heartbeat",
    ) -> None:
        self.config = config
        self.provider = provider
        self.interval_s = interval_s or config.heartbeat_interval_s
        self.gateway_http_path = gateway_http_path
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.last_hb: OffChainHeartbeat | None = None
        self.sent_count = 0
        self.error_count = 0

    async def push_once(self) -> OffChainHeartbeat:
        """Build and push a single heartbeat (used by `POST /heartbeat-now`)."""
        sources = self.provider()
        hb = build_heartbeat(self.config, sources)
        self.last_hb = hb
        url = f"{self.config.gateway_ws_url}{self.gateway_http_path}"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post(
                    url,
                    json=hb.model_dump(mode="json"),
                    headers=_gateway_auth_headers(self.config),
                )
            self.sent_count += 1
        except httpx.HTTPError:
            self.error_count += 1
        return hb

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await self.push_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except TimeoutError:
                self._task.cancel()
