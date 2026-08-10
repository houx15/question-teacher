# Persistent Lesson ID Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist every completed lesson in local SQLite and let a teacher reopen the complete lesson after a service restart by entering its course ID.

**Architecture:** Keep generation jobs as in-process state, but add an optional SQLite backing file to the existing store for complete `RuntimeLesson` objects. Save the validated private runtime JSON before a job is marked complete; on a memory miss, load and revalidate that JSON from SQLite. The generation page becomes the teacher-facing ID handoff: it shows/copies the new ID and validates an existing ID before entering the full-screen classroom.

**Tech Stack:** Python 3.9, standard-library `sqlite3`, FastAPI, Pydantic v2, vanilla JavaScript/CSS, pytest, Node test runner

---

## File map

- `app/store.py`: keep ephemeral jobs and hot lesson objects; add SQLite schema creation, transactional lesson insert, lazy ID load, and full Pydantic revalidation.
- `app/main.py`: configure the production database at `var/lessons.sqlite3` and revalidate the unversioned generation HTML shell.
- `app/api.py`: preserve the existing generation contract and guarantee persistence completes before the job receives a lesson ID.
- `app/static/index.html`: add the existing-ID entry form and generation-complete ID card.
- `app/static/generate.js`: stop auto-redirecting on completion; copy/open IDs and validate stored IDs before navigation.
- `app/static/styles.css`: style the teacher-only ID controls without changing the full-screen lesson page.
- `tests/test_store.py`: storage persistence, restart, corruption, duplicate, validation, and parent-directory behavior.
- `tests/test_api.py`: persistence-failure ordering, app restart recovery, public redaction, interaction recovery, and no automatic input reuse.
- `tests/test_static_pages.py`: HTML/JS/CSS contracts and generation-shell cache policy.
- `.gitignore`: ignore SQLite database, WAL, and shared-memory sidecars.
- `README.md`: document ID persistence, restart recovery, and the demo workflow.

### Task 1: Persist complete lessons by ID

**Files:**
- Create: `tests/test_store.py`
- Modify: `app/store.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing round-trip and restart tests**

Create `tests/test_store.py` with a minimal complete runtime fixture and tests that construct two independent stores over the same temporary database:

```python
import sqlite3

import pytest

from app.schemas import Interaction, ProblemInput, RuntimeBeat, RuntimeLesson
from app.store import MemoryStore


def stored_lesson(lesson_id="lesson-persisted"):
    problem = ProblemInput(
        problem_text="2x+3=7",
        reference_answer="x=2",
        reference_solution_text="两边减3，再除以2。",
        lesson_length="standard",
    )
    interaction = Interaction(
        interaction_id="check-1",
        kind="choice",
        prompt="x等于多少？",
        expected_answer="o1",
        options=[
            {"option_id": "o1", "label": "x=2"},
            {"option_id": "o2", "label": "x=5"},
        ],
    )
    return RuntimeLesson(
        lesson_id=lesson_id,
        problem=problem,
        title="一次方程",
        learning_goal="理解等式变形",
        beats=[
            RuntimeBeat(
                beat_id="beat-1",
                purpose="求解",
                narration="先减三。",
                board_actions=[],
                layer="base",
                interaction=interaction,
            )
        ],
        summary="完成",
        transfer_item={
            "problem_text": "2x=4",
            "expected_answer": "x=2",
            "method_signal": "保持等式平衡",
        },
        validation_report={"math_status": "verified"},
    )


def test_sqlite_store_restores_complete_lesson_after_restart(tmp_path):
    database = tmp_path / "nested" / "lessons.sqlite3"
    lesson = stored_lesson()
    MemoryStore(database).save_lesson(lesson)

    restored = MemoryStore(database).get_lesson(lesson.lesson_id)

    assert restored == lesson
    assert database.is_file()


def test_sqlite_store_restores_private_interaction_after_restart(tmp_path):
    database = tmp_path / "lessons.sqlite3"
    lesson = stored_lesson()
    MemoryStore(database).save_lesson(lesson)

    interaction = MemoryStore(database).get_interaction(
        lesson.lesson_id,
        "check-1",
    )

    assert interaction is not None
    assert interaction.expected_answer == "o1"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_store.py
```

Expected: FAIL because `MemoryStore` does not accept a database path and creates no SQLite record.

- [ ] **Step 3: Add corruption, duplicate, invalid-ID, and missing-ID tests**

Append focused tests that define the failure contract:

```python
def test_sqlite_store_missing_and_unsafe_ids_are_not_found(tmp_path):
    store = MemoryStore(tmp_path / "lessons.sqlite3")
    assert store.get_lesson("missing") is None
    assert store.get_lesson("../lesson") is None


def test_sqlite_store_rejects_duplicate_lesson_id(tmp_path):
    store = MemoryStore(tmp_path / "lessons.sqlite3")
    store.save_lesson(stored_lesson())
    with pytest.raises(ValueError, match="already exists"):
        store.save_lesson(stored_lesson())
    assert store.get_lesson("lesson-persisted") == stored_lesson()


def test_sqlite_store_fails_closed_for_corrupt_runtime_json(tmp_path):
    database = tmp_path / "lessons.sqlite3"
    store = MemoryStore(database)
    store.save_lesson(stored_lesson())
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE lessons SET runtime_json = ? WHERE lesson_id = ?",
            ("{not-json", "lesson-persisted"),
        )
    assert MemoryStore(database).get_lesson("lesson-persisted") is None
```

- [ ] **Step 4: Implement the minimal SQLite-backed store**

Modify `app/store.py` so jobs remain memory-only and lessons can be durable:

```python
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Optional, Union

from pydantic import ValidationError

SAFE_LESSON_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
LESSON_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id TEXT PRIMARY KEY,
    problem_text TEXT NOT NULL,
    reference_answer TEXT NOT NULL,
    reference_solution_text TEXT,
    required_method TEXT,
    lesson_length TEXT NOT NULL,
    runtime_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class MemoryStore:
    def __init__(
        self,
        database_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._jobs: Dict[str, GenerationJob] = {}
        self._lessons: Dict[str, RuntimeLesson] = {}
        self._database_path = (
            Path(database_path) if database_path is not None else None
        )
        self._lock = RLock()

    def _connect(self) -> sqlite3.Connection:
        if self._database_path is None:
            raise RuntimeError("lesson database is not configured")
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database_path, timeout=5.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute(LESSON_SCHEMA)
        return connection
```

Implement `save_lesson` with `INSERT`, JSON serialization, UTC timestamp, and database-first ordering. Convert `required_method` to its stored string value when it is an enum. Translate only duplicate-key `sqlite3.IntegrityError` into `ValueError("lesson id already exists")`; allow other I/O/database errors to propagate so generation fails.

Implement `get_lesson` as: reject unsafe IDs, return hot cache when present, return `None` without creating a database when its file is absent, fetch `runtime_json`, call `RuntimeLesson.model_validate_json`, cache a valid result, and return `None` for JSON/Pydantic validation failure. Change `get_interaction` to call `get_lesson` rather than reading `_lessons` directly.

- [ ] **Step 5: Verify store tests GREEN and existing API store tests stay green**

Run:

```bash
pytest -q tests/test_store.py tests/test_api.py::test_memory_store_revalidates_updates_without_partial_mutation tests/test_api.py::test_memory_store_supports_concurrent_job_creation
```

Expected: all selected tests PASS.

- [ ] **Step 6: Ignore SQLite runtime artifacts**

Append to `.gitignore`:

```gitignore
var/*.sqlite3
var/*.sqlite3-wal
var/*.sqlite3-shm
```

- [ ] **Step 7: Commit the durable store**

```bash
git add app/store.py tests/test_store.py .gitignore
git commit -m "feat: persist lessons by id"
```

### Task 2: Wire persistence into generation and restart recovery

**Files:**
- Modify: `app/main.py`
- Modify: `app/api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing generation-order and restart tests**

Add to `tests/test_api.py`:

```python
class FailingLessonStore(RecordingStore):
    def save_lesson(self, lesson):
        raise OSError("private database path")


def test_generation_does_not_return_id_when_persistence_fails():
    store = FailingLessonStore()
    client, _, _ = build_client(store=store)
    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert job["lesson_id"] is None
    assert job["error"] == "课程生成失败，请稍后重试。"


def test_new_app_instance_reads_persisted_lesson_and_interaction(tmp_path):
    database = tmp_path / "lessons.sqlite3"
    first_store = MemoryStore(database)
    lesson, interaction = save_interaction_lesson(
        first_store,
        kind="choice",
        expected="o1",
        options=[
            InteractionOption(option_id="o1", label="x=2"),
            InteractionOption(option_id="o2", label="x=5"),
        ],
    )
    second_client, _, _ = build_client(store=MemoryStore(database))

    lesson_response = second_client.get(f"/api/lessons/{lesson.lesson_id}")
    evaluation = second_client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "o1",
        },
    )

    assert lesson_response.status_code == 200
    assert "reference_answer" not in lesson_response.json()["problem"]
    assert evaluation.json()["classification"] == "correct"
```

Also change `build_client` so tests remain isolated from the production database:

```python
def build_client(**overrides):
    generator = overrides.pop("generator", FakeGenerator())
    audio_service = overrides.pop("audio_service", FakeAudioService())
    overrides.setdefault("store", MemoryStore())
    app = create_app(
        generator=generator,
        audio_service=audio_service,
        **overrides,
    )
    return TestClient(app), generator, audio_service
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_api.py -k "persistence_fails or persisted_lesson"
```

Expected: persistence-failure behavior may already pass; restart recovery fails until SQLite storage is wired correctly. If the first test passes immediately, keep it as a regression guard and confirm the restart test is RED for the intended missing behavior.

- [ ] **Step 3: Configure the production database**

Modify `app/main.py`:

```python
LESSON_DATABASE = PROJECT_ROOT / "var" / "lessons.sqlite3"


def create_app(
    settings: Optional[Settings] = None,
    generator: Any = None,
    audio_service: Any = None,
    store: Optional[MemoryStore] = None,
    math_engine: Optional[MathEngine] = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_store = (
        store if store is not None else MemoryStore(LESSON_DATABASE)
    )
    services = ApiServices(
        settings=resolved_settings,
        store=resolved_store,
        math_engine=math_engine or MathEngine(),
        generator=generator,
        audio_service=audio_service,
    )
```

Do not persist generation jobs. Keep `run_generation` ordering as `attach_audio` → `save_lesson` → completed job update. Add a short comment at `save_lesson` stating that returning the ID is contingent on durable save.

- [ ] **Step 4: Verify API tests GREEN**

Run:

```bash
pytest -q tests/test_api.py
```

Expected: all API tests PASS without creating `var/lessons.sqlite3` from isolated clients.

- [ ] **Step 5: Commit production wiring**

```bash
git add app/main.py app/api.py tests/test_api.py
git commit -m "feat: restore persisted lessons after restart"
```

### Task 3: Show, copy, and reopen course IDs

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/generate.js`
- Modify: `app/static/styles.css`
- Modify: `app/main.py`
- Modify: `tests/test_static_pages.py`

- [ ] **Step 1: Write failing static-page contract tests**

Extend `tests/test_static_pages.py`:

```python
def test_generation_page_exposes_existing_and_completed_lesson_id_flows():
    response = page_client().get("/")
    html = response.text
    assert response.headers["cache-control"] == "no-cache"
    for region_id in (
        "existing-lesson-form",
        "existing-lesson-id",
        "existing-lesson-error",
        "generation-complete",
        "completed-lesson-id",
        "copy-lesson-id",
        "enter-completed-lesson",
        "create-another-lesson",
    ):
        assert f'id="{region_id}"' in html
    assert 'src="/static/generate.js?v=20260810-1"' in html


def test_generation_runtime_shows_id_and_validates_existing_course():
    source = page_client().get("/static/generate.js").text
    assert "function showCompletion(lessonId)" in source
    assert "completedLessonId.value = lessonId" in source
    assert "navigator.clipboard.writeText" in source
    assert 'fetch(`/api/lessons/${encodeURIComponent(lessonId)}`' in source
    assert "window.location.assign" in source
    completed_branch = source[
        source.index('job.status === "completed"'):
        source.index('job.status === "failed"')
    ]
    assert "showCompletion(job.lesson_id)" in completed_branch
    assert "window.location.assign" not in completed_branch
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest -q tests/test_static_pages.py -k "lesson_id_flows or validates_existing_course"
```

Expected: FAIL because the teacher ID forms and completion card do not exist.

- [ ] **Step 3: Add the existing-course form and completion card**

In `app/static/index.html`, add a teacher-only section near the authoring sheet:

```html
<form id="existing-lesson-form" class="existing-lesson-form" novalidate>
  <label for="existing-lesson-id">已有课程 ID</label>
  <div class="lesson-id-row">
    <input id="existing-lesson-id" name="lesson_id" autocomplete="off" required
      maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
      placeholder="粘贴课程 ID">
    <button type="submit">打开课堂</button>
  </div>
  <p id="existing-lesson-error" class="inline-error" role="alert" hidden></p>
</form>
```

Add a hidden completion section after generation progress:

```html
<section id="generation-complete" class="generation-complete" hidden>
  <p class="progress-eyebrow">LESSON SAVED</p>
  <h2>课程已保存</h2>
  <label for="completed-lesson-id">课程 ID</label>
  <div class="lesson-id-row">
    <input id="completed-lesson-id" readonly>
    <button id="copy-lesson-id" type="button">复制课程 ID</button>
  </div>
  <p id="copy-lesson-status" aria-live="polite"></p>
  <div class="completion-actions">
    <a id="enter-completed-lesson" class="generate-button" href="#">进入课堂</a>
    <button id="create-another-lesson" class="text-button" type="button">继续生成新课程</button>
  </div>
</section>
```

Version the entry script as `/static/generate.js?v=20260810-1`.

- [ ] **Step 4: Implement the ID completion and reopen controller**

In `app/static/generate.js`, bind the new nodes. Add:

```javascript
function lessonPath(lessonId) {
  return `/lesson/${encodeURIComponent(lessonId)}`;
}

function showCompletion(lessonId) {
  clearTimeout(pollTimer);
  activeJob = null;
  form.hidden = true;
  progress.hidden = true;
  generationComplete.hidden = false;
  completedLessonId.value = lessonId;
  enterCompletedLesson.href = lessonPath(lessonId);
  copyLessonStatus.textContent = "";
}
```

Replace the completed-job auto-navigation with `showCompletion(job.lesson_id)`.

The copy button must call `navigator.clipboard.writeText(completedLessonId.value)` and report success or manual-copy fallback without clearing the visible ID. The “continue generating” button hides completion and calls `restoreForm()`.

The existing-ID form must trim its input, call `reportValidity`, fetch `/api/lessons/{encoded-id}` with `cache: "no-store"`, then navigate only after a 200 response. A 404 shows `没有找到这个课程 ID。`; other failures show `暂时无法读取课程，请稍后重试。`.

- [ ] **Step 5: Add focused teacher-page styles**

In `app/static/styles.css`, add bounded styles for `.existing-lesson-form`, `.lesson-id-row`, `.generation-complete`, and `.completion-actions`. Reuse current paper, ink, rule, focus, sans, and button tokens. Ensure the input/button row stacks under the existing mobile breakpoint. Do not modify `.classroom-*`, `.board-*`, or lesson overlay rules.

- [ ] **Step 6: Revalidate the generation HTML shell**

Modify the `/` route in `app/main.py` to match the lesson shell:

```python
return FileResponse(
    page,
    headers={"Cache-Control": "no-cache"},
)
```

The versioned `generate.js` response must remain free of `no-cache` and `no-store`; add a reverse assertion modeled on `test_versioned_lesson_module_remains_cacheable`.

- [ ] **Step 7: Run static and Node tests GREEN**

Run:

```bash
pytest -q tests/test_static_pages.py
npm test
node --check app/static/generate.js
```

Expected: all commands exit 0.

- [ ] **Step 8: Commit the teacher ID experience**

```bash
git add app/static/index.html app/static/generate.js app/static/styles.css app/main.py tests/test_static_pages.py
git commit -m "feat: reopen saved lessons by id"
```

### Task 4: Document, verify, and create the reusable demo course

**Files:**
- Modify: `README.md`
- Test: `tests/test_static_pages.py`

- [ ] **Step 1: Write the failing documentation assertion**

Add a README contract to `tests/test_static_pages.py`:

```python
def test_readme_documents_persistent_lesson_ids():
    readme = (repository_root / "README.md").read_text(encoding="utf-8")
    assert "var/lessons.sqlite3" in readme
    assert "输入课程 ID" in readme
    assert "服务重启" in readme
    assert "相同题目再次生成" in readme
```

- [ ] **Step 2: Run the documentation test and verify RED**

Run:

```bash
pytest -q tests/test_static_pages.py::test_readme_documents_persistent_lesson_ids
```

Expected: FAIL because README does not describe persistent lesson IDs.

- [ ] **Step 3: Document the operator workflow and boundary**

Update `README.md` with:

- database and audio locations;
- generation-complete ID handoff;
- restart and open-by-ID procedure;
- statement that the same question generates a new ID;
- warning that deleting either the SQLite row or its audio directory makes the saved classroom incomplete;
- backup instruction that copies `var/lessons.sqlite3` together with `var/audio/` while the service is stopped.

- [ ] **Step 4: Run the complete offline verification**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests
npm test
node --check app/static/generate.js
node --check app/static/lesson.js
node --check app/static/cue-player.mjs
node --check app/static/math-text.mjs
node --check app/static/runtime-core.mjs
git diff --check
git status -sb
```

Expected: all test and syntax commands exit 0; only the pre-existing untracked `.superpowers/` may remain.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md tests/test_static_pages.py
git commit -m "docs: explain saved lesson demo flow"
```

- [ ] **Step 6: Generate and persist the parameter-root demo lesson**

Start the server from the `general` environment with `.env` exported. Submit the known parameter-root problem once:

```json
{
  "problem_text": "若$2n$ ($n\\ne 0$)是关于 x的方程 $x^2-2mx+2n=0$的根，则m-n的值为",
  "reference_answer": "$\\frac{1}{2}$",
  "reference_solution_text": "因为 $2n(n\\ne 0)$ 是关于x的方程$x^2-2mx+2n=0$的解\n所以 $4n^2-4mn+2n=0$\n所以$4n-4m+2=0$\n所以$m-n=\\frac{1}{2}$"
}
```

Wait until the job reports `completed`; record the returned ID and confirm its row exists in `var/lessons.sqlite3` without printing private runtime JSON.

- [ ] **Step 7: Prove restart recovery in the real app**

Stop the server cleanly, start it again, and verify:

```text
GET /api/lessons/{lesson_id} → 200
GET /lesson/{lesson_id} → 200
```

Use a 1280×800 browser session to enter the ID from the homepage, play through the synchronized parameter-root cue, and confirm KaTeX, audio, highlight, board write, pause/resume, choice interaction, and feedback. Save one generation-complete screenshot and one restored-classroom screenshot.

- [ ] **Step 8: Verify new generations do not auto-reuse an ID without paying for a second live lesson**

Use the fake generator API test to submit the same payload twice and assert `generator.calls == 2` and the two completed jobs receive distinct lesson IDs from a generator fixture that returns distinct IDs. Do not make a second paid live LLM/TTS call solely for this assertion.
