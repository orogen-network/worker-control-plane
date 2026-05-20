"""worker-control-plane tests."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from mining_types import LoadSnapshot, Quantization, generate_keypair
from pydantic import SecretStr

from worker_control_plane import (
    CapabilitySource,
    ControlPlaneConfig,
    HeartbeatPusher,
    MockRegistryClient,
    OperatorRegistration,
    build_app,
    build_heartbeat,
    merge_capabilities,
)

INTERNAL_TOKEN = "test-wcp-internal"


@pytest.fixture(autouse=True)
def auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERNAL_AUTH_TOKEN", INTERNAL_TOKEN)
    monkeypatch.delenv("OROGEN_ENV", raising=False)


def _hdrs() -> dict[str, str]:
    return {"Authorization": f"Bearer {INTERNAL_TOKEN}"}


@pytest.fixture
def config() -> ControlPlaneConfig:
    priv, pub = generate_keypair()
    return ControlPlaneConfig(
        operator_id="op-cp",
        operator_private_key_hex=priv,
        operator_public_key_hex=pub,
        gateway_ws_url="http://gateway",
        attestation_report_hash="aa" * 32,
        endpoint_url="http://worker-host:8000",
    )


def test_capability_merge_unions_adapters() -> None:
    a = CapabilitySource(
        source_id="vllm-1",
        base_model_id="m-7b",
        adapter_ids=["adp-a", "adp-b"],
        max_concurrent_requests=8,
    )
    b = CapabilitySource(
        source_id="vllm-2",
        base_model_id="m-7b",
        adapter_ids=["adp-b", "adp-c"],
        max_concurrent_requests=4,
    )
    merged = merge_capabilities([a, b])
    assert len(merged) == 1
    assert merged[0].adapter_ids == ["adp-a", "adp-b", "adp-c"]
    assert merged[0].max_concurrent_requests == 12


def test_capability_merge_quant_priority_keeps_higher_precision() -> None:
    int4 = CapabilitySource(
        source_id="edge",
        base_model_id="m-7b",
        quantization=Quantization.INT4,
        deterministic_mode=True,
    )
    fp16 = CapabilitySource(
        source_id="vllm",
        base_model_id="m-7b",
        quantization=Quantization.FP16,
        deterministic_mode=False,
    )
    merged = merge_capabilities([int4, fp16])
    assert merged[0].quantization == Quantization.FP16
    assert merged[0].deterministic_mode is False  # all-must-agree


def test_heartbeat_signs_and_has_merged_caps(config: ControlPlaneConfig) -> None:
    src = CapabilitySource(
        source_id="vllm-1",
        base_model_id="m-7b",
        adapter_ids=["adp-a"],
        load=LoadSnapshot(active_requests=2, gpu_utilization_pct=50.0),
    )
    hb = build_heartbeat(config, [src])
    assert hb.operator_id == "op-cp"
    assert hb.signature
    assert hb.capabilities[0].base_model_id == "m-7b"
    assert hb.current_load.active_requests == 2
    assert hb.geo_region == "US"
    assert hb.endpoint_url == "http://worker-host:8000"


def test_private_key_is_wrapped_in_secretstr(config: ControlPlaneConfig) -> None:
    """HIGH-SVC-009: never expose the private key in repr or str."""
    assert isinstance(config.operator_private_key_hex, SecretStr)
    assert "**" in repr(config.operator_private_key_hex)
    assert "**" in str(config.operator_private_key_hex)
    # The raw value must be accessible only via the explicit accessor.
    assert len(config.operator_private_key()) == 64


def test_registration_payload_shape_and_signing(config: ControlPlaneConfig) -> None:
    reg = OperatorRegistration(
        operator_id=config.operator_id,
        operator_public_key_hex=config.operator_public_key_hex,
        attestation_report_hash=config.attestation_report_hash,
        geo_region=config.geo_region,
        endpoint_url=config.endpoint_url,
        stake_amount=config.stake_amount,
        commission_bps=config.commission_bps,
        advertised_models=["m-7b"],
    ).sign(config.operator_private_key())
    assert reg.signature
    assert len(reg.extrinsic_hash()) == 64  # hex-256
    # Tampering breaks the hash.
    tampered = reg.model_copy(update={"stake_amount": reg.stake_amount + 1})
    assert tampered.extrinsic_hash() != reg.extrinsic_hash()


def test_routes_require_internal_auth(config: ControlPlaneConfig) -> None:
    chain = MockRegistryClient()
    app = build_app(config, registry=chain)
    with TestClient(app) as client:
        # /register without auth → 401.
        r = client.post("/register", json={})
        assert r.status_code == 401
        # /internal/sources without auth → 401.
        src = CapabilitySource(source_id="vllm-1", base_model_id="m-7b")
        r2 = client.post("/internal/sources", json=src.model_dump(mode="json"))
        assert r2.status_code == 401
        # /healthz is open.
        r3 = client.get("/healthz")
        assert r3.status_code == 200


def test_healthz_does_not_leak_private_key(config: ControlPlaneConfig) -> None:
    app = build_app(config, registry=MockRegistryClient())
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        # Serialize a much wider tree (status) and ensure no secret bleeds.
        body = json.dumps(r.json()).lower()
        assert "private" not in body
        assert config.operator_private_key() not in body


def test_register_endpoint_uses_mock_chain(config: ControlPlaneConfig) -> None:
    chain = MockRegistryClient()
    app = build_app(config, registry=chain)
    with TestClient(app) as client:
        # Source heartbeat first so register has advertised models.
        src = CapabilitySource(
            source_id="vllm-1",
            base_model_id="m-7b",
        )
        client.post(
            "/internal/sources", json=src.model_dump(mode="json"), headers=_hdrs(),
        )

        r = client.post("/register", json={}, headers=_hdrs())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["registration"]["operator_id"] == "op-cp"
        assert body["registration"]["advertised_models"] == ["m-7b"]
        assert chain.is_registered("op-cp")
        # No private key field anywhere in the response.
        flat = json.dumps(body).lower()
        assert "private" not in flat
        assert config.operator_private_key() not in flat

        # Duplicate → 409.
        r2 = client.post("/register", json={}, headers=_hdrs())
        assert r2.status_code == 409


def test_register_rejects_zero_stake(config: ControlPlaneConfig) -> None:
    bad_config = ControlPlaneConfig(
        operator_id=config.operator_id,
        operator_private_key_hex=config.operator_private_key(),
        operator_public_key_hex=config.operator_public_key_hex,
        gateway_ws_url=config.gateway_ws_url,
        attestation_report_hash=config.attestation_report_hash,
        endpoint_url=config.endpoint_url,
        stake_amount=0,
    )
    app = build_app(bad_config, registry=MockRegistryClient())
    with TestClient(app) as client:
        r = client.post(
            "/register", json={"advertised_models": ["m-7b"]}, headers=_hdrs(),
        )
        assert r.status_code == 400


def test_status_endpoint(config: ControlPlaneConfig) -> None:
    app = build_app(config, registry=MockRegistryClient())
    with TestClient(app) as client:
        src1 = CapabilitySource(
            source_id="vllm-1",
            base_model_id="m-7b",
            load=LoadSnapshot(active_requests=3, gpu_utilization_pct=40.0),
        )
        src2 = CapabilitySource(
            source_id="sgl-1",
            base_model_id="m-13b",
            load=LoadSnapshot(active_requests=1, gpu_utilization_pct=80.0),
        )
        client.post(
            "/internal/sources", json=src1.model_dump(mode="json"), headers=_hdrs(),
        )
        client.post(
            "/internal/sources", json=src2.model_dump(mode="json"), headers=_hdrs(),
        )
        r = client.get("/status", headers=_hdrs())
        assert r.status_code == 200
        body = r.json()
        models = sorted(c["base_model_id"] for c in body["capabilities"])
        assert models == ["m-13b", "m-7b"]
        assert body["load"]["active_requests"] == 4
        assert body["load"]["gpu_utilization_pct"] == pytest.approx(60.0)
        assert body["registered"] is False


async def test_heartbeat_now_pushes_to_mocked_gateway(
    config: ControlPlaneConfig,
) -> None:
    app = build_app(config, registry=MockRegistryClient())
    with respx.mock(assert_all_called=True) as mocker:
        mocker.post("http://gateway/internal/heartbeat").mock(
            return_value=httpx.Response(200, json={"ok": True}),
        )
        with TestClient(app) as client:
            src = CapabilitySource(
                source_id="vllm-1",
                base_model_id="m-7b",
            )
            client.post(
                "/internal/sources", json=src.model_dump(mode="json"), headers=_hdrs(),
            )
            r = client.post("/heartbeat-now", headers=_hdrs())
            assert r.status_code == 200
            body = r.json()
            assert body["sent_count"] == 1
            assert body["heartbeat"]["operator_id"] == "op-cp"
            assert body["heartbeat"]["signature"]
        assert (
            mocker.calls[0].request.headers["authorization"]
            == f"Bearer {INTERNAL_TOKEN}"
        )


def test_pusher_push_once_increments_error_count(config: ControlPlaneConfig) -> None:
    import asyncio

    bad_config = ControlPlaneConfig(
        operator_id=config.operator_id,
        operator_private_key_hex=config.operator_private_key(),
        operator_public_key_hex=config.operator_public_key_hex,
        gateway_ws_url="http://127.0.0.1:1",  # unbound
        attestation_report_hash=config.attestation_report_hash,
        endpoint_url=config.endpoint_url,
    )
    pusher = HeartbeatPusher(
        bad_config,
        lambda: [CapabilitySource(source_id="x", base_model_id="m-7b")],
    )
    asyncio.run(pusher.push_once())
    assert pusher.error_count == 1
    assert pusher.sent_count == 0
