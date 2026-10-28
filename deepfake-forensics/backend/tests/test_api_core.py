"""Health, models, settings, uploads, inference, history, forensics, reports."""
from __future__ import annotations

from tests.conftest import image_bytes, make_settings


def _analyze(client, data=None, track="ai_generated", name="img.png", mime="image/png"):
    return client.post("/api/analyze", files={"file": (name, data if data is not None else image_bytes(), mime)}, data={"track": track})


# ----------------------------------------------------------------------------- health & models
def test_health(client_no_model):
    r = client_no_model.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"]["inference_device"] == "cpu"
    assert "cuda_available" in body["device"]
    assert body["services"]["database"] == "ok"
    assert body["services"]["model:ai_generated"] == "no checkpoint installed"


def test_models_without_checkpoints(client_no_model):
    r = client_no_model.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["models"] == []
    assert {t["track"]: t["status"] for t in body["tracks"]} == {"ai_generated": "unavailable", "face_deepfake": "unavailable"}
    assert all(t["message"] for t in body["tracks"])
    assert {a["arch"] for a in body["architectures"]} == {"resnet50", "efficientnet_b0"}
    assert all(a["status"] in ("pretrained", "unavailable") for a in body["architectures"])


def test_models_with_checkpoint(client):
    body = client.get("/api/models").json()
    assert len(body["models"]) == 1
    m = body["models"][0]
    assert m["status"] == "trained" and m["is_active"] and m["is_synthetic_fixture"]
    tracks = {t["track"]: t for t in body["tracks"]}
    assert tracks["ai_generated"]["status"] == "ready"
    assert tracks["face_deepfake"]["status"] == "unavailable"


def test_invalid_checkpoint_reported(client):
    bad = client.app_settings.checkpoints_dir / "ai_generated" / "broken.pt"
    bad.write_bytes(b"not a checkpoint")
    models = client.get("/api/models").json()["models"]
    broken = [m for m in models if m["model_id"] == "broken"][0]
    assert broken["status"] == "invalid" and broken["error"]
    # A broken checkpoint must never become active.
    assert client.get("/api/settings").json()["active_models"]["ai_generated"] == "TEST_fixture_model"


def test_set_active_model_validation(client):
    assert client.put("/api/models/ai_generated/active", json={"model_id": "does_not_exist"}).status_code == 404
    assert client.put("/api/models/face_deepfake/active", json={"model_id": "TEST_fixture_model"}).status_code == 404
    assert client.put("/api/models/ai_generated/active", json={"model_id": "TEST_fixture_model"}).status_code == 200


# ----------------------------------------------------------------------------- uploads
def test_missing_model_returns_503_and_no_prediction(client_no_model):
    r = _analyze(client_no_model)
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "model_unavailable"
    assert "prediction" not in r.json()
    assert client_no_model.get("/api/analyses").json()["total"] == 0


def test_face_track_without_model(client):
    r = _analyze(client, track="face_deepfake")
    assert r.status_code == 503


def test_unsupported_format_by_content(client):
    r = _analyze(client, data=b"GIF89a" + b"\x00" * 100, name="fake.png")
    assert r.status_code == 415
    r = _analyze(client, data=b"%PDF-1.4 not an image", name="x.jpg", mime="image/jpeg")
    assert r.status_code == 415


def test_extension_is_not_trusted(client):
    # Valid PNG named .txt is accepted (content decides) ...
    assert _analyze(client, data=image_bytes("PNG"), name="picture.txt", mime="text/plain").status_code == 200


def test_empty_and_truncated_upload(client):
    assert _analyze(client, data=b"").status_code == 422
    png = image_bytes("PNG", size=(200, 200))
    r = _analyze(client, data=png[: len(png) // 2])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_image"


def test_too_small_image(client):
    r = _analyze(client, data=image_bytes("PNG", size=(10, 10)))
    assert r.status_code == 422


def test_upload_size_limit(tmp_root):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import write_test_checkpoint

    s = make_settings(tmp_root, max_upload_mb=0.01)
    write_test_checkpoint(s.artifacts_dir)
    with TestClient(create_app(s)) as c:
        r = _analyze(c, data=image_bytes("PNG", size=(300, 300)))
        assert r.status_code == 413


def test_invalid_track(client):
    r = client.post("/api/analyze", files={"file": ("a.png", image_bytes(), "image/png")}, data={"track": "videos"})
    assert r.status_code == 422


# ----------------------------------------------------------------------------- inference & storage
def test_analyze_response_schema_and_history(client):
    r = _analyze(client, data=image_bytes("JPEG", exif=True), name="../../etc/passwd.jpg", mime="image/jpeg")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["prediction"] in ("REAL", "FAKE")
    assert 0.0 <= b["fake_score"] <= 1.0
    assert abs(b["fake_score"] + b["real_score"] - 1.0) < 1e-6
    assert b["prediction"] == ("FAKE" if b["fake_score"] >= b["threshold"] else "REAL")
    assert b["threshold"] == 0.5 and "checkpoint" in b["threshold_source"]
    assert b["inference_ms"] > 0
    assert b["model"]["model_id"] == "TEST_fixture_model" and b["model"]["is_synthetic_fixture"]
    assert any("SYNTHETIC" in w for w in b["warnings"])
    assert "not proof" in b["disclaimer"]
    assert b["explanation"]["available"] and b["explanation"]["explained_class"] == b["prediction"]
    assert set(b["artifacts"]) == {"original", "input", "gradcam", "heatmap", "spectrum"}
    # Client file name is sanitised and never used as a path
    assert "/" not in b["image"]["file_name"] and ".." not in b["image"]["file_name"].split(".")[0]
    for url in b["artifacts"].values():
        a = client.get(url)
        assert a.status_code == 200 and len(a.content) > 0
    # Sensitive EXIF redacted, non-sensitive shown
    exif = b["forensics"]["exif"]
    assert exif["tags"].get("Software") == "UnitTestSoftware 1.0"
    assert "Artist" in exif["redacted"] and "Artist" not in exif["tags"]
    assert "Secret Person" not in r.text
    # History
    lst = client.get("/api/analyses").json()
    assert lst["total"] == 1 and lst["items"][0]["id"] == b["id"]
    assert client.get(f"/api/analyses/{b['id']}").json()["fake_score"] == b["fake_score"]


def test_inference_is_deterministic(client):
    data = image_bytes("PNG", seed=3)
    s1 = _analyze(client, data=data).json()["fake_score"]
    s2 = _analyze(client, data=data).json()["fake_score"]
    assert abs(s1 - s2) < 1e-6


def test_threshold_override(client):
    assert client.put("/api/settings", json={"thresholds": {"ai_generated": 1.5}}).status_code == 422
    assert client.put("/api/settings", json={"thresholds": {"ai_generated": 0.0}}).status_code == 422
    r = client.put("/api/settings", json={"thresholds": {"ai_generated": 0.99}})
    assert r.status_code == 200 and r.json()["thresholds"]["ai_generated"] == 0.99
    b = _analyze(client).json()
    assert b["threshold"] == 0.99 and b["threshold_source"] == "user setting"
    assert b["prediction"] == ("FAKE" if b["fake_score"] >= 0.99 else "REAL")
    client.put("/api/settings", json={"thresholds": {"ai_generated": None}})
    assert _analyze(client).json()["threshold"] == 0.5


def test_history_filters_and_pagination(client):
    for i in range(3):
        _analyze(client, data=image_bytes(seed=i))
    page = client.get("/api/analyses", params={"page": 1, "page_size": 2}).json()
    assert page["total"] == 3 and len(page["items"]) == 2
    assert client.get("/api/analyses", params={"track": "face_deepfake"}).json()["total"] == 0
    preds = [i["prediction"] for i in client.get("/api/analyses").json()["items"]]
    for p in ("REAL", "FAKE"):
        assert client.get("/api/analyses", params={"prediction": p}).json()["total"] == preds.count(p)
    assert client.get("/api/analyses", params={"date_from": "2999-01-01"}).json()["total"] == 0
    assert client.get("/api/analyses", params={"prediction": "MAYBE"}).status_code == 422
    stats = client.get("/api/stats").json()
    assert stats["total"] == 3 and stats["real"] + stats["fake"] == 3 and stats["fixture_model_analyses"] == 3


def test_forensics_endpoint(client):
    b = _analyze(client, data=image_bytes("JPEG"), mime="image/jpeg").json()
    f = client.get(f"/api/forensics/{b['id']}").json()
    assert f["properties"]["format"] == "JPEG"
    assert f["properties"]["width"] == 96 and f["properties"]["height"] == 80
    q = f["jpeg"]["quality_estimate"]["estimated_quality"]
    assert 70 <= q <= 80  # Pillow default quality is 75 with standard IJG tables
    assert f["frequency"]["radial_profile"]["log_magnitude"]
    assert "not model inferences" in f["note"]


def test_png_without_exif_observation(client):
    b = _analyze(client, data=image_bytes("PNG")).json()
    obs = client.get(f"/api/forensics/{b['id']}").json()["observations"]
    assert any("No EXIF" in o["text"] and "does not indicate" in o["caveat"] for o in obs)


def test_report_generation(client):
    b = _analyze(client).json()
    r = client.get(f"/api/analyses/{b['id']}/report")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    assert "attachment" in r.headers["content-disposition"]


def test_delete_removes_record_and_files(client):
    b = _analyze(client).json()
    adir = client.app_settings.analyses_dir / b["id"]
    assert adir.is_dir()
    assert client.delete(f"/api/analyses/{b['id']}").status_code == 204
    assert not adir.exists()
    assert client.get(f"/api/analyses/{b['id']}").status_code == 404
    assert client.delete(f"/api/analyses/{b['id']}").status_code == 404


def test_path_traversal_rejected(client):
    b = _analyze(client).json()
    for url in (
        f"/api/analyses/{b['id']}/artifacts/..%2F..%2Fapp.db",
        f"/api/analyses/{b['id']}/artifacts/payload",
        "/api/analyses/..%2F..%2Fstorage/report",
        "/api/evaluation/experiments/eval--..%2F..%2Fstorage/artifacts/app.db",
    ):
        assert client.get(url).status_code == 404, url


def test_api_key_enforced(tmp_root):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(make_settings(tmp_root, api_key="s3cret"))) as c:
        assert c.get("/api/health").status_code == 401
        assert c.get("/api/health", headers={"X-API-Key": "wrong"}).status_code == 401
        r = c.get("/api/health", headers={"X-API-Key": "s3cret"})
        assert r.status_code == 200 and r.json()["auth_required"] is True


def test_error_format(client):
    r = client.get("/api/analyses/doesnotexist")
    assert r.status_code == 404
    assert set(r.json()["error"]) == {"code", "message", "details"}
