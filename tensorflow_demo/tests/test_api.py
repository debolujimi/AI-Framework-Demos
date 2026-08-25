# ============================================================
# DEPLOYED API INTEGRATION TESTS
# Cats vs Dogs AI
#
# These tests communicate with the running Docker service
# through HTTP.
#
# No TensorFlow or requests imports are required locally.
# ============================================================

import json
import urllib.request
import urllib.error
from pathlib import Path
import uuid


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "http://127.0.0.1:5000"

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_IMAGE = (
    PROJECT_DIR
    / "cats_and_dogs"
    / "test"
    / "1.jpg"
)


# ============================================================
# HELPER: GET REQUEST
# ============================================================

def http_get(path):

    with urllib.request.urlopen(
        f"{BASE_URL}{path}",
        timeout=10
    ) as response:

        body = response.read().decode("utf-8")

        return (
            response.status,
            body
        )


# ============================================================
# HELPER: POST WITHOUT IMAGE
# ============================================================

def http_post_empty(path):

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=b"",
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:

            body = response.read().decode("utf-8")

            return (
                response.status,
                body
            )

    except urllib.error.HTTPError as error:

        body = error.read().decode("utf-8")

        return (
            error.code,
            body
        )


# ============================================================
# HELPER: MULTIPART IMAGE POST
# ============================================================

def http_post_image(path, image_path):

    boundary = (
        "----PythonBoundary"
        + uuid.uuid4().hex
    )

    image_bytes = image_path.read_bytes()

    body = b""

    body += (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; '
        'name="image"; filename="1.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")

    body += image_bytes

    body += (
        f"\r\n--{boundary}--\r\n"
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        method="POST"
    )

    request.add_header(
        "Content-Type",
        f"multipart/form-data; boundary={boundary}"
    )

    request.add_header(
        "Content-Length",
        str(len(body))
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        response_body = (
            response
            .read()
            .decode("utf-8")
        )

        return (
            response.status,
            response_body
        )


# ============================================================
# TEST 1 — HOME PAGE
# ============================================================

def test_home_page():

    status, body = http_get("/")

    assert status == 200

    assert "Cats vs Dogs AI" in body


# ============================================================
# TEST 2 — HEALTH ENDPOINT
# ============================================================

def test_health_endpoint():

    status, body = http_get(
        "/health"
    )

    assert status == 200

    data = json.loads(body)

    assert data["status"] == "healthy"

    assert data["service"] == "cats-dogs-ai"

    assert data["model"] == "model1_basic_cnn"


# ============================================================
# TEST 3 — PREDICT WITHOUT IMAGE
# ============================================================

def test_predict_without_image():

    status, body = http_post_empty(
        "/predict"
    )

    assert status == 400

    data = json.loads(body)

    assert "error" in data


# ============================================================
# TEST 4 — VALID IMAGE PREDICTION
# ============================================================

def test_predict_with_valid_image():

    assert TEST_IMAGE.exists(), (
        f"Test image not found: {TEST_IMAGE}"
    )

    status, body = http_post_image(
        "/predict",
        TEST_IMAGE
    )

    assert status == 200

    data = json.loads(body)

    assert "prediction" in data

    assert "confidence" in data

    assert data["prediction"] in [
        "CATS",
        "DOGS"
    ]

    assert (
        0.0
        <= data["confidence"]
        <= 1.0
    )


# ============================================================
# TEST 5 — KNOWN DEMO IMAGE
# ============================================================

def test_known_demo_image():

    status, body = http_post_image(
        "/predict",
        TEST_IMAGE
    )

    assert status == 200

    data = json.loads(body)

    assert data["prediction"] == "DOGS"