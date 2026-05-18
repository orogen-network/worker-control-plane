# worker-control-plane

Per-host control plane that sits alongside one or more `infer-worker-*` daemons. Owns:

1. **Heartbeat sender** (RFC-0003) — every 12s, push merged capability + load to the
   gateway. HTTP today, WebSocket in production (same payload).
2. **Capability merger** — union over multiple workers on the same host (vLLM + SGLang +
   llama.cpp coexistence). Adapter IDs union, max-context/concurrency take max/sum,
   deterministic_mode is the AND.
3. **Operator registration** (RFC-0009) — build + sign the
   `pallet-operator-registry::register_operator` extrinsic and submit via a
   `RegistryClient`. `MockRegistryClient` keeps state in-memory for dev/tests; production
   wires substrate-interface to the live chain-node.

## Why Python (not Rust)

Plan §1.3 originally specified Rust. Agent #2's reference (this directory) chose Python
to share the FastAPI + pydantic stack with the other operator-side services. The choice
is captured in DECISIONS H16. The Rust port is a Phase-3 follow-up; until then the Python
control plane covers all heartbeat + registration semantics with full RFC compliance.

## HTTP API

| Method | Path                  | Description |
|--------|-----------------------|-------------|
| GET    | `/status`             | Merged capability + load snapshot. |
| POST   | `/register`           | One-time operator registration on chain (mock client OK). |
| POST   | `/heartbeat-now`      | Force-send a heartbeat (harness probes). |
| POST   | `/internal/sources`   | Workers POST their `CapabilitySource` snapshots here. |
| GET    | `/healthz`            | Liveness. |

## Multi-worker example

```python
config = ControlPlaneConfig(
    operator_id="op-bigcorp",
    operator_private_key_hex=...,
    operator_public_key_hex=...,
    gateway_ws_url="wss://gw.orogen.network",
    attestation_report_hash=...,
)
app = build_app(config, autostart_heartbeat=True)

# vllm-7b worker registers its capability:
POST /internal/sources
{
  "source_id": "vllm-7b",
  "base_model_id": "llama-3-7b",
  "adapter_ids": ["cust-acme-v3"],
  "max_concurrent_requests": 8,
  "load": {"active_requests": 3, "gpu_utilization_pct": 65.0}
}
```
