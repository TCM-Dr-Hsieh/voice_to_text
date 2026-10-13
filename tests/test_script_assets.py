"""Versioned third-party conversion data should remain browser-cacheable."""
from fastapi.testclient import TestClient
from nicegui import app

import voice_app.ui  # noqa: F401 - registers the static routes


def test_opencc_bundle_uses_long_cache():
    response = TestClient(app).get('/opencc.full.js?v=1.4.2')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'public, max-age=31536000'


def test_own_conversion_script_revalidates_on_reload():
    response = TestClient(app).get('/script_converter.js?v=3')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'public, max-age=0'


def test_meeting_view_script_is_served_and_revalidated():
    response = TestClient(app).get('/meeting.js?v=1')
    assert response.status_code == 200 and 'class MeetingView' in response.text
    assert response.headers['cache-control'] == 'public, max-age=0'
