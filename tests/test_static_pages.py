from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from scripts.smoke_live import assert_generated_lesson_contract


def page_client():
    return TestClient(create_app())


def test_live_smoke_asserts_method_first_choice_contract_without_answers():
    choice = SimpleNamespace(
        kind="choice",
        options=[
            SimpleNamespace(
                label=rf"\\(x={value}\\)",
                feedback="诊断反馈。",
                feedback_audio_url=f"/audio/option-{value}.mp3",
            )
            for value in ("1", "2", "3")
        ],
    )
    lesson = SimpleNamespace(
        beats=[
            SimpleNamespace(
                purpose="进入问题",
                layer="base",
                narration="先看题目。",
                board_actions=[],
                interaction=None,
                audio_url="/audio/opening.mp3",
            ),
            SimpleNamespace(
                purpose="先认识方法",
                layer="micro_explanation",
                narration="今天用配方法。",
                board_actions=[SimpleNamespace(content="配方法")],
                interaction=None,
                audio_url="/audio/method.mp3",
            ),
            SimpleNamespace(
                purpose="诊断",
                layer="interaction",
                narration="请选择。",
                board_actions=[],
                interaction=choice,
                audio_url="/audio/diagnostic.mp3",
            ),
            SimpleNamespace(
                purpose="完成近迁移",
                layer="interaction",
                narration="现在迁移。",
                board_actions=[],
                interaction=choice,
                audio_url="/audio/transfer.mp3",
            ),
        ]
    )

    summary = assert_generated_lesson_contract(lesson)

    assert summary == {
        "method_first": True,
        "interaction_kinds": ["choice", "choice"],
        "diagnostic_choice_count": 2,
        "option_feedback_audio_ready": True,
        "formula_labels_ready": True,
        "audio_ready": True,
    }
    assert "expected_answer" not in summary


def test_readme_documents_method_first_choice_generation_and_local_katex():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    for phrase in (
        "先认识方法",
        "配方法",
        "选择或点选",
        "兼容读取",
        "诊断",
        "选项",
        "KaTeX",
        "npm install",
        "npm test",
        "package-lock.json",
        "CDN",
        "python -m compileall -q app scripts tests",
        "浏览器",
        "教学",
    ):
        assert phrase in readme


def test_generation_page_has_focused_authoring_form():
    response = page_client().get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="lesson-form"' in html
    assert 'name="problem_text"' in html
    assert 'name="reference_answer"' in html
    assert 'name="reference_solution_text"' in html
    assert 'id="reference_solution_text"' in html
    assert 'maxlength="12000"' in html
    assert "参考解析" in html
    assert "输入题目、参考答案与可选参考解析" in html
    assert 'name="required_method"' in html
    assert 'name="lesson_length"' in html
    assert 'id="model-status"' in html
    assert 'id="voice-status"' in html
    assert 'id="generation-progress"' in html
    assert 'src="/static/generate.js"' in html
    assert "OPENAI_API_KEY" not in html
    assert "validation_report" not in html


def test_generation_page_submits_optional_reference_solution():
    source = page_client().get("/static/generate.js").text
    html = page_client().get("/").text

    assert 'data.get("reference_solution_text")' in source
    assert "reference_solution_text: referenceSolution || null" in source
    assert '"正在审阅参考解析"' in source
    assert 'data-stage="正在审阅参考解析"' in html


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
    assert '<link rel="stylesheet" href="/static/vendor/katex/katex.min.css">' in html
    assert 'class="sidebar"' not in html


def test_lesson_runtime_renders_math_and_tracks_unrendered_board_sources():
    source = page_client().get("/static/lesson.js").text

    assert 'import { renderMathText } from "./math-text.mjs";' in source
    assert "renderMathText(dom.title, lesson.title)" in source
    assert "renderMathText(dom.problem, lesson.problem.problem_text)" in source
    assert "renderMathText(dom.narration, beat.narration)" in source
    assert "renderMathText(heading, interaction.prompt)" in source
    assert "renderMathText(button, option.label)" in source
    assert "renderMathText(ui.feedback, presentation.message)" in source
    assert "content.dataset.source" in source
    assert "content.textContent !== value.content" not in source
    assert "renderMathText(content, source)" in source


def test_choice_submission_passes_selected_option_without_exposing_answer_key():
    source = page_client().get("/static/lesson.js").text

    assert "submitInteraction(interaction, option.option_id, option," in source
    assert "async function submitInteraction(interaction, answer, selectedOption, ui)" in source
    assert "resolveInteractionPresentation" in source
    assert "expected: interaction.expected_answer" not in source
    assert "interaction.expected_answer" not in source


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
    assert '<meta name="theme-color" content="#f4efe5">' in lesson_html
    assert "--classroom-canvas: #f4efe5;" in styles.text
    assert "--board-surface: #fbfaf6;" in styles.text
    assert "--board-ink: #203047;" in styles.text
    assert "--classroom-panel: #fffdf8;" in styles.text


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
    assert "const activeRegion = runtime.layerStack.length > 0" in source
    assert 'activeRegion.querySelectorAll(".board-object")' in source
    assert "node.getClientRects().length > 0" in source
    assert 'selectablePrefix.className = "sr-only board-selectable-prefix"' in source
    assert 'selectablePrefix.textContent = "选择板书："' in source
    assert 'selectablePrefix.setAttribute("aria-hidden", "true")' in source
    assert 'prefix?.removeAttribute("aria-hidden")' in source
    assert 'prefix?.setAttribute("aria-hidden", "true")' in source
    assert 'node.setAttribute("aria-label"' not in source
    assert "humanizeTarget(node.dataset.boardTarget)" not in source
    assert "boardSource" not in source
    assert 'event.key === "Enter" || event.key === " "' in source


def test_primary_control_uses_runtime_intent_for_ended_pause_beats():
    source = page_client().get("/static/lesson.js").text

    assert "runtime.primaryControlIntent(paused)" in source
    assert 'if (intent === "advance")' in source
    assert 'if (action.type === "pause") setPaused(true)' not in source


def test_local_katex_assets_are_served_with_the_math_text_module():
    client = page_client()

    for path, media_type in (
        ("/static/math-text.mjs", "text/javascript"),
        ("/static/vendor/katex/katex.mjs", "text/javascript"),
        ("/static/vendor/katex/katex.min.css", "text/css"),
        ("/static/vendor/katex/fonts/KaTeX_Main-Regular.woff2", "font/woff2"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(media_type)


def test_interaction_evaluation_and_feedback_audio_have_bounded_lifecycles():
    source = page_client().get("/static/lesson.js").text

    assert "const EVALUATION_TIMEOUT_MS = 14000;" in source
    assert "const FEEDBACK_AUDIO_TIMEOUT_MS = 12000;" in source
    assert "let activeEvaluationController = null;" in source
    assert "let interactionSubmitting = false;" in source
    assert "const controller = new AbortController();" in source
    assert "activeEvaluationController = controller;" in source
    assert "activeEvaluationController === controller" in source
    assert "signal: controller.signal" in source
    assert "controller.abort()" in source
    assert "clearTimeout(timeout)" in source
    assert "createBoundedSettlement" in source
    assert "feedbackAudioFinalizer?.()" in source
    assert 'audio.addEventListener("ended", onEnded' in source
    assert 'audio.addEventListener("error", onError' in source
    assert "playAttempt?.catch(settle)" in source
    assert "window.setTimeout(resolve, 650)" not in source
    assert 'audio.removeAttribute("src")' in source
    assert "audio.load()" in source


def test_submission_state_guards_every_async_boundary_and_navigation():
    source = page_client().get("/static/lesson.js").text
    submit_source = source[
        source.index("async function submitInteraction"):
        source.index("async function toggleFullscreen")
    ]

    assert "let submissionSequence = 0;" in source
    assert "submissionSequence += 1;" in source
    assert "interactionSubmitting = true;" in submit_source
    assert "const originatingBeatToken = beatToken;" in submit_source
    assert "const originatingInteractionId = interaction.interaction_id;" in submit_source
    assert "const isCurrentSubmission = () =>" in submit_source
    assert submit_source.count("if (!isCurrentSubmission()) return;") >= 3
    assert submit_source.index("if (!isCurrentSubmission()) return;") < (
        submit_source.index("runtime.recordAnswer")
    )
    assert "interactionSubmitting = false;" in submit_source
    assert "activeEvaluationController?.abort()" in source
    assert "|| interactionSubmitting" in source


def test_global_shortcuts_preserve_native_interactive_keyboard_behavior():
    source = page_client().get("/static/lesson.js").text

    assert "isNativeInteractiveTarget(event.target)" in source
    assert "isNativeInteractiveTarget(document.activeElement)" in source
    assert 'if (!dom.previous.disabled) previousBeat();' in source
    assert 'if (!dom.replay.disabled) replayCurrentBeat();' in source


def test_needs_review_waits_for_explicit_continue_and_errors_clear_stale_hints():
    source = page_client().get("/static/lesson.js").text
    submit_source = source[
        source.index("async function submitInteraction"):
        source.index("async function toggleFullscreen")
    ]
    catch_source = submit_source.rsplit("} catch {", maxsplit=1)[1]

    assert 'presentation.advanceMode === "manual"' in submit_source
    assert "ui.continueButton.focus()" in submit_source
    assert "return;" in submit_source
    assert 'renderMathText(ui.hint, "");' in catch_source
    assert 'interaction.kind === "point_select"' in catch_source
    assert "enablePointSelection((retryAnswer)" in catch_source


def test_interaction_card_scrolls_safely_in_short_landscape():
    styles = page_client().get("/static/styles.css").text

    assert ".interaction-card {" in styles
    assert "max-height: 100%;" in styles
    assert "overflow-y: auto;" in styles
    assert ".interaction-card h2 {" in styles
    assert "overflow-x: auto;" in styles
    assert "@media (max-height: 600px) and (orientation: landscape)" in styles
    assert ".interaction-stage {" in styles
