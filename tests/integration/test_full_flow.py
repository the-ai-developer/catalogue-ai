"""Cross-service integration tests for Catalogue AI.

Runs against a LIVE stack (docker compose up): Go API + model-server + Postgres.
Pure stdlib so it needs no install:  python3 -m pytest tests/integration -q
or directly:                            python3 tests/integration/test_full_flow.py

Env:
    API_BASE   (default http://localhost:8080)
    API_KEY    (admin key; default matches the local bootstrap key in .env.example)
    EDITOR_KEY (editor key; defaults to API_KEY)
    VIEWER_KEY (viewer key; defaults to API_KEY)
    WAIT_SECS  (per-job poll budget, default 120)
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import sys
import time
import unittest
import urllib.error
import urllib.request
import zlib
import uuid

API_BASE = os.environ.get("API_BASE", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY", "local-dev-admin-key")
EDITOR_KEY = os.environ.get("EDITOR_KEY", API_KEY)
VIEWER_KEY = os.environ.get("VIEWER_KEY", API_KEY)
WAIT_SECS = int(os.environ.get("WAIT_SECS", "120"))


# --------------------------------------------------------------------------- #
# tiny stdlib HTTP client + PNG factory
# --------------------------------------------------------------------------- #

def _call(method: str, path: str, *, body=None, key=API_KEY, raw=None,
          headers=None, timeout=60):
    url = API_BASE + path
    data = raw
    hdrs = {"X-API-Key": key}
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            code = resp.status
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        code = exc.code
    try:
        parsed = json.loads(payload) if payload else {}
    except json.JSONDecodeError:
        parsed = {"_raw": payload.decode(errors="replace")}
    return code, parsed


def tiny_png(width: int = 8, height: int = 8, rgb=(200, 120, 40)) -> bytes:
    """Minimal valid PNG built with stdlib only."""
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + ctype + data
                + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    idat = chunk(b"IDAT", zlib.compress(raw))
    return sig + ihdr + idat + chunk(b"IEND", b"")


def multipart_file(field: str, filename: str, content: bytes,
                   mime: str = "image/png"):
    boundary = "----cat" + uuid.uuid4().hex
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="{field}"; '
               f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
    body.write(content)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def wait_job(path: str, terminal=("succeeded", "failed", "draft_ready",
                                 "approved", "rejected", "published")):
    deadline = time.time() + WAIT_SECS
    last = {}
    while time.time() < deadline:
        code, last = _call("GET", path)
        assert code == 200, f"{path} -> {code} {last}"
        if last.get("status") in terminal:
            return last
        time.sleep(2)
    raise AssertionError(f"job {path} not terminal within {WAIT_SECS}s: {last}")


def make_item() -> dict:
    code, item = _call("POST", "/api/v1/items", body={
        "sku": f"IT-{uuid.uuid4().hex[:10].upper()}",
        "title": "Kestrel 12 oz Insulated Bottle",
        "category": "drinkware",
        "material": "18/8 stainless steel",
        "dimensions": {"height_cm": 26.0, "diameter_cm": 7.2, "capacity_oz": 12},
        "features": ["double-wall vacuum insulation", "leak-proof lid", "BPA-free"],
        "extra": {"colours": ["slate", "sand"]},
    })
    assert code == 201, f"create item -> {code} {item}"
    return item


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

class TestPlatform(unittest.TestCase):
    def test_health_and_ready(self):
        code, body = _call("GET", "/healthz", key="")
        self.assertEqual(code, 200)
        code, body = _call("GET", "/readyz", key="")
        self.assertEqual(code, 200, f"readyz -> {body}")

    def test_auth_envelope(self):
        code, body = _call("GET", "/api/v1/items", key="wrong-key")
        self.assertEqual(code, 401)
        self.assertEqual(body.get("error", {}).get("code"), "unauthorized")

    def test_error_envelope_shape(self):
        code, body = _call("GET", "/api/v1/items/00000000-0000-0000-0000-000000000000")
        self.assertEqual(code, 404)
        self.assertEqual(body.get("error", {}).get("code"), "not_found")

    def test_viewer_cannot_create(self):
        code, body = _call("POST", "/api/v1/items", key=VIEWER_KEY,
                           body={"sku": "X", "title": "t", "category": "c"})
        self.assertIn(code, (401, 403))


class TestCatalogue(unittest.TestCase):
    def test_crud_and_validation(self):
        item = make_item()
        self.assertEqual(item["title"], "Kestrel 12 oz Insulated Bottle")
        self.assertEqual(item["status"], "draft")

        code, got = _call("GET", f"/api/v1/items/{item['id']}")
        self.assertEqual(code, 200)
        self.assertEqual(got["sku"], item["sku"])

        code, upd = _call("PATCH", f"/api/v1/items/{item['id']}",
                          body={"status": "active", "material": "titanium"})
        self.assertEqual(code, 200)
        self.assertEqual(upd["status"], "active")

        code, bad = _call("POST", "/api/v1/items", body={"title": "no sku"})
        self.assertEqual(code, 400)
        self.assertEqual(bad["error"]["code"], "bad_request")

        code, _ = _call("DELETE", f"/api/v1/items/{item['id']}")
        self.assertEqual(code, 204)
        code, archived = _call("GET", f"/api/v1/items/{item['id']}")
        self.assertIn(code, (200, 404))
        if code == 200:
            self.assertEqual(archived["status"], "archived")

    def test_asset_upload(self):
        item = make_item()
        png = tiny_png()
        body, headers = multipart_file("file", "bottle.png", png)
        code, asset = _call("POST", f"/api/v1/items/{item['id']}/assets",
                            raw=body, headers=headers)
        self.assertEqual(code, 201, asset)
        self.assertEqual(asset["mime"], "image/png")
        self.assertEqual(asset["width"], 8)
        self.assertEqual(asset["sha256"], __import__("hashlib").sha256(png).hexdigest())


class TestProject1QA(unittest.TestCase):
    def test_ingest_and_grounded_answer(self):
        item = make_item()
        code, _ = _call("PATCH", f"/api/v1/items/{item['id']}",
                        body={"status": "active"})
        self.assertEqual(code, 200)
        body, headers = multipart_file("file", "bottle.png", tiny_png())
        code, _ = _call("POST", f"/api/v1/items/{item['id']}/assets",
                        raw=body, headers=headers)
        self.assertEqual(code, 201)

        code, job = _call("POST", f"/api/v1/items/{item['id']}/ingest", body={})
        self.assertEqual(code, 202, job)
        done = wait_job(f"/api/v1/jobs/ingest/{job['job_id']}")
        self.assertEqual(done["status"], "succeeded", done)
        self.assertGreater(done["chunks_indexed"], 0)

        code, res = _call("POST", "/api/v1/qa/ask", body={
            "question": "Is the Kestrel bottle dishwasher safe "
                        "and what is its capacity?",
            "top_k": 6, "use_images": True, "composition": "extractive",
            "user_ref": "itest:buyer",
        })
        self.assertEqual(code, 200, res)
        self.assertIn("answer", res)
        self.assertIn("citation_check", res)
        self.assertIsInstance(res["citation_check"]["passed"], bool)
        self.assertIn("citations", res)
        self.assertIn("retrieval", res)
        self.assertIn("latency_ms", res)
        # contract rule: every citation carries sentence_index + modality + score
        for cit in res["citations"]:
            self.assertIn("sentence_index", cit)
            self.assertIn(cit["modality"], ("text", "image"))
            self.assertIsInstance(cit["score"], (int, float))

        # history + fetch round-trip
        code, hist = _call("GET", "/api/v1/qa/history?user_ref=itest:buyer")
        self.assertEqual(code, 200)
        self.assertTrue(any(a.get("answer_id") == res["answer_id"]
                            or a.get("id") == res["answer_id"]
                            for a in hist.get("items", hist.get("answers", []))))
        code, stored = _call("GET", f"/api/v1/qa/answers/{res['answer_id']}")
        self.assertEqual(code, 200)
        self.assertEqual(stored["answer"], res["answer"])


class TestProject2Descriptions(unittest.TestCase):
    def test_editor_gate_flow(self):
        code, job = _call("POST", "/api/v1/descriptions/jobs", key=EDITOR_KEY, body={
            "spec": {
                "title": "Kestrel 12 oz Insulated Bottle",
                "category": "drinkware",
                "material": "18/8 stainless steel",
                "dimensions": {"height_cm": 26.0, "capacity_oz": 12},
                "features": ["double-wall vacuum insulation", "leak-proof lid"],
            },
            "beam_width": 4, "max_len": 192,
        })
        self.assertEqual(code, 202, job)
        done = wait_job(f"/api/v1/descriptions/jobs/{job['job_id']}")
        self.assertEqual(done["status"], "draft_ready", done)
        self.assertTrue(done.get("drafts"), "no drafts returned")
        self.assertEqual(done["drafts"][0]["rank"], 1)

        # publish before approval must FAIL (editor gate)
        code, early = _call("POST",
                            f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                            key=EDITOR_KEY, body={})
        self.assertIn(code, (409, 422), early)

        # edit-inline path: editor rewrites, then publish
        top = done["drafts"][0]
        code, reviewed = _call(
            "POST", f"/api/v1/descriptions/jobs/{job['job_id']}/review",
            key=EDITOR_KEY,
            body={"decision": "edit", "draft_id": top["id"],
                  "edited_text": top["text"] + " Built for the daily commute.",
                  "notes": "tightened the close"})
        self.assertEqual(code, 200, reviewed)
        self.assertEqual(reviewed["status"], "approved")

        code, pub = _call("POST",
                          f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                          key=EDITOR_KEY, body={})
        self.assertEqual(code, 201, pub)
        self.assertTrue(pub["text"].endswith("Built for the daily commute."))

        code, hist = _call(
            "GET", f"/api/v1/descriptions/published/{pub['item_id']}")
        self.assertEqual(code, 200)

    def test_reject_path(self):
        code, job = _call("POST", "/api/v1/descriptions/jobs", key=EDITOR_KEY, body={
            "spec": {"title": "Ridge 40L Travel Pack", "category": "bags",
                     "material": "recycled nylon",
                     "features": ["laptop sleeve", "rain cover"]},
            "beam_width": 2, "max_len": 128,
        })
        self.assertEqual(code, 202, job)
        done = wait_job(f"/api/v1/descriptions/jobs/{job['job_id']}")
        self.assertEqual(done["status"], "draft_ready", done)
        code, rejected = _call(
            "POST", f"/api/v1/descriptions/jobs/{job['job_id']}/review",
            key=EDITOR_KEY,
            body={"decision": "reject", "notes": "tone off-brand"})
        self.assertEqual(code, 200, rejected)
        self.assertEqual(rejected["status"], "rejected")
        code, _ = _call("POST",
                        f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                        key=EDITOR_KEY, body={})
        self.assertIn(code, (409, 422))


if __name__ == "__main__":
    unittest.main(verbosity=2)
    sys.exit(0)
