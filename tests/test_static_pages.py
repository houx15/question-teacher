from fastapi.testclient import TestClient

from app.main import create_app


def page_client():
    return TestClient(create_app())


def test_generation_page_has_focused_authoring_form():
    response = page_client().get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="lesson-form"' in html
    assert 'name="problem_text"' in html
    assert 'name="reference_answer"' in html
    assert 'name="required_method"' in html
    assert 'name="lesson_length"' in html
    assert 'id="model-status"' in html
    assert 'id="voice-status"' in html
    assert 'id="generation-progress"' in html
    assert 'src="/static/generate.js"' in html
    assert "OPENAI_API_KEY" not in html
    assert "validation_report" not in html


def test_lesson_page_has_fullscreen_classroom_regions():
    response = page_client().get("/lesson/example")

    assert response.status_code == 200
    html = response.text
    for region_id in (
        "classroom-shell",
        "lesson-topbar",
        "problem-display",
        "route-strip",
        "board-stage",
        "base-board",
        "layer-stage",
        "narration-line",
        "interaction-stage",
        "lesson-controls",
        "start-overlay",
        "loading-state",
        "error-state",
        "empty-state",
        "rotate-state",
    ):
        assert f'id="{region_id}"' in html
    assert 'type="module"' in html
    assert 'src="/static/lesson.js"' in html
    assert 'class="sidebar"' not in html


def test_static_pages_include_accessibility_and_responsive_contracts():
    client = page_client()
    index_html = client.get("/").text
    lesson_html = client.get("/lesson/example").text
    styles = client.get("/static/styles.css")

    assert 'aria-live="polite"' in index_html
    assert 'aria-live="polite"' in lesson_html
    assert 'aria-label="暂停或继续"' in lesson_html
    assert 'aria-label="进入全屏"' in lesson_html
    assert styles.status_code == 200
    assert "@media (orientation: portrait)" in styles.text
    assert "--board:" in styles.text
    assert "--focus:" in styles.text


def test_interaction_submission_uses_server_authoritative_contract():
    source = page_client().get("/static/lesson.js").text

    assert "lesson_id: lesson.lesson_id" in source
    assert "interaction_id: interaction.interaction_id" in source
    assert "expected: interaction.expected_answer" not in source


def test_point_select_prompt_does_not_block_board_pointer_or_keyboard_access():
    client = page_client()
    source = client.get("/static/lesson.js").text
    styles = client.get("/static/styles.css").text

    assert 'classList.toggle("is-point-select"' in source
    assert ".interaction-stage.is-point-select" in styles
    assert "pointer-events: none" in styles
    assert ".interaction-stage.is-point-select .interaction-card" in styles
    assert "pointer-events: auto" in styles
    assert 'node.setAttribute("role", "button")' in source
    assert 'event.key === "Enter" || event.key === " "' in source


def test_primary_control_uses_runtime_intent_for_ended_pause_beats():
    source = page_client().get("/static/lesson.js").text

    assert "runtime.primaryControlIntent(paused)" in source
    assert 'if (intent === "advance")' in source
    assert 'if (action.type === "pause") setPaused(true)' not in source
