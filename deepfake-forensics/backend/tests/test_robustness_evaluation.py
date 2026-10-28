"""Robustness endpoint validation, evaluation jobs and result browsing."""
from __future__ import annotations

import time

from tests.conftest import image_bytes


def _rob(client, **fields):
    data = {"track": "ai_generated", **{k: str(v) for k, v in fields.items()}}
    return client.post("/api/robustness/analyze", files={"file": ("a.png", image_bytes(size=(128, 128)), "image/png")}, data=data)


def test_robustness_requires_a_perturbation(client):
    r = _rob(client)
    assert r.status_code == 422


def test_robustness_parameter_ranges(client):
    assert _rob(client, jpeg_quality=0).status_code == 422
    assert _rob(client, jpeg_quality=101).status_code == 422
    assert _rob(client, downscale_max_side=4).status_code == 422
    assert _rob(client, blur_sigma=-1).status_code == 422
    assert _rob(client, noise_std=1000).status_code == 422


def test_robustness_runs_model_on_variants(client):
    r = _rob(client, jpeg_quality=50, blur_sigma=1.5, noise_std=10, downscale_max_side=48)
    assert r.status_code == 200, r.text
    b = r.json()
    assert len(b["variants"]) == 5  # 4 individual + combined
    for v in [b["original"], *b["variants"]]:
        assert v["prediction"] in ("REAL", "FAKE")
        assert 0 <= v["fake_score"] <= 1
        assert v["image_data_url"].startswith("data:image/png;base64,")
    assert "not a benchmark" in b["note"]
    # Robustness runs are demonstrations and are not stored as analyses.
    assert client.get("/api/analyses").json()["total"] == 0


def test_robustness_without_model(client_no_model):
    r = client_no_model.post("/api/robustness/analyze", files={"file": ("a.png", image_bytes(), "image/png")}, data={"track": "ai_generated", "jpeg_quality": "50"})
    assert r.status_code == 503


def test_evaluation_empty_state(client_no_model):
    s = client_no_model.get("/api/evaluation/summary").json()
    assert s["available"] is False
    assert "No evaluation results available" in s["message"]
    assert client_no_model.get("/api/evaluation/experiments").json() == []
    assert client_no_model.get("/api/evaluation/experiments/eval--nothing").status_code == 404


def test_job_not_found(client):
    assert client.get("/api/jobs/0123456789abcdef").status_code == 404


def test_evaluation_run_validation(client):
    assert client.post("/api/evaluation/run", json={"model_id": "TEST_fixture_model"}).status_code == 422
    r = client.post("/api/evaluation/run", json={"model_id": "nope", "dataset_root": "x", "manifest": "y.csv"})
    assert r.status_code == 404
    # Paths outside DFF_DATA_ROOTS are refused
    r = client.post("/api/evaluation/run", json={"model_id": "TEST_fixture_model", "dataset_root": "C:/Windows", "manifest": "C:/Windows/win.ini"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "path_not_allowed"
    r = client.post("/api/evaluation/run", json={"model_id": "TEST_fixture_model", "dataset_name": "not_configured"})
    assert r.status_code == 422


def test_evaluation_job_end_to_end(client, tmp_root):
    from ml.data.fixture import make_fixture

    data_dir = tmp_root / "data" / "fixture"
    manifests = make_fixture(data_dir, n_per_class=6, size=64)
    r = client.post("/api/evaluation/run", json={
        "model_id": "TEST_fixture_model",
        "dataset_root": str(data_dir),
        "manifest": str(manifests["ai_generated"]),
        "splits": ["train", "val", "test"],
        "batch_size": 8,
        "robustness": True,
    })
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + 240
    job = None
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    assert job is not None and job["status"] == "succeeded", job
    assert job["progress"] == 1.0 and job["logs"]
    # Filesystem paths are not echoed back in job params
    assert "dataset_root" not in job["params"] and "manifest" not in job["params"]
    exp_id = job["result"]["experiment_id"]
    detail = client.get(f"/api/evaluation/experiments/{exp_id}").json()
    res = detail["result"]
    assert res["kind"] == "robustness"
    assert res["dataset"]["is_synthetic_fixture"] is True
    assert {"accuracy", "f1_macro", "roc_auc", "precision_fake", "recall_real"} <= set(res["metrics"])
    assert len(res["confusion_matrix"]) == 2
    assert res["robustness"]["conditions"][0]["name"] == "clean"
    assert res["latency"]["mean_ms"] > 0
    names = {a["name"] for a in detail["artifacts"]}
    assert {"roc.png", "confusion_matrix.png", "predictions.csv", "result.json"} <= names
    png = client.get(f"/api/evaluation/experiments/{exp_id}/artifacts/roc.png")
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG")
    s = client.get("/api/evaluation/summary").json()
    assert s["available"] and s["counts"]["robustness"] == 1
    csv = client.get("/api/evaluation/metrics.csv")
    assert csv.status_code == 200 and exp_id in csv.text
