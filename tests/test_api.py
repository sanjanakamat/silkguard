from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_home_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/frontend/index.html"
