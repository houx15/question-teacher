# Sync Cue Board Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace evenly distributed Beat-level board animation with Cue-level Volcengine speech, problem highlighting, board writing, and emphasis lifecycle synchronized to semantic teaching phrases.

**Architecture:** Keep `RuntimeBeat` as the cognitive and interaction boundary, and add ordered `RuntimeSyncCue` objects as the playback boundary. The server compiles stable problem-focus targets, the model may reference only those IDs or previously written board IDs, Volcengine TTS generates one asset per Cue, and the player executes lead/start/end actions around each Cue audio clip. Formula rendering remains inside the existing constrained KaTeX path; highlighting wraps trusted renderer output and never introduces model-authored HTML, CSS, selectors, or string offsets.

**Tech Stack:** Python 3.9, FastAPI, Pydantic v2, pytest, vanilla ES modules, Node test runner, KaTeX, Volcengine TTS v3.

**Design spec:** `docs/superpowers/specs/2026-08-07-sync-cue-board-timing-design.md`

**Workspace note:** Execute on the current `feature/reference-grounded-validation` branch. Do not create a worktree.

---

## File Map

**Create**

- `app/problem_focus.py` — deterministic problem math segmentation and stable focus-target IDs.
- `app/static/cue-player.mjs` — Cue audio/timer sequencing with pause, resume, stop, fallback, and completion.
- `tests/test_problem_focus.py` — server-side focus-target parsing and safety tests.
- `tests/cue-player.test.mjs` — deterministic Cue playback tests with fake audio and timers.
- `tests/fixtures/problem-focus-cases.json` — shared formula/focus cases consumed by Python and JavaScript tests.

**Modify**

- `app/schemas.py` — Cue, visual-action, focus-target, and runtime payload contracts.
- `app/prompts.py` — Director, Revision, Materials, and Reviewer Cue contracts.
- `app/generation.py` — Cue validation, frozen-route coverage, interaction-leak checks, and draft assembly.
- `app/compiler.py` — compile every Beat into Cue-based runtime data while deriving legacy narration.
- `app/audio_service.py` — bounded Cue-level TTS and all-or-nothing lesson audio attachment.
- `app/api.py` — keep Cue audio public while preserving private answer and validation fields.
- `app/static/math-text.mjs` — stable problem math segments and safe focus wrappers around KaTeX output.
- `app/static/runtime-core.mjs` — visual-action reducer and selective emphasis lifecycle.
- `app/static/lesson.js` — Cue player integration, problem-focus rendering, replay snapshots, and interaction gating.
- `app/static/lesson.html` — bump static module version after runtime changes.
- `app/static/styles.css` — active and trace styles for highlight, underline, and red emphasis.
- `scripts/smoke_live.py` — assert Cue audio and fixed parameter-root synchronization contract.
- `tests/test_schemas.py`
- `tests/test_generation.py`
- `tests/test_generation_agents.py`
- `tests/test_compiler.py`
- `tests/test_tts_client.py`
- `tests/test_api.py`
- `tests/test_static_pages.py`
- `tests/math-text.test.mjs`
- `tests/runtime-core.test.mjs`
- `README.md`

---

### Task 1: Add Strict Cue and Visual-Action Contracts

**Files:**

- Modify: `app/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Add tests that establish the only accepted highlight syntax and the optional-visual rule:

```python
import pytest
from pydantic import ValidationError

from app.schemas import NarrativeSyncCue, SyncVisualAction


def test_sync_cue_can_be_voice_only():
    cue = NarrativeSyncCue(
        cue_id="explain-purpose",
        spoken_text="先看题目要求我们求什么。",
        lead_actions=[],
        start_actions=[],
        end_actions=[],
    )

    assert cue.spoken_text == "先看题目要求我们求什么。"


def test_sync_visual_action_uses_enum_style_and_semantic_target():
    action = SyncVisualAction(
        surface="problem",
        type="emphasize",
        target="problem-math-001",
        emphasis_style="underline",
        persistence="trace",
    )

    assert action.target == "problem-math-001"


def test_sync_visual_action_rejects_inline_css_or_selector():
    with pytest.raises(ValidationError):
        SyncVisualAction.model_validate(
            {
                "surface": "problem",
                "type": "emphasize",
                "target": "#problem span:nth-child(2)",
                "emphasis_style": "color:red",
                "persistence": "trace",
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", "problem-math-001[data-secret]"),
        ("target", "problem-math-001:nth-child(2)"),
        ("emphasis_style", "background:url(javascript:alert(1))"),
        ("emphasis_style", "[[red]]"),
    ],
)
def test_sync_visual_action_rejects_markup_and_selector_syntax(field, value):
    payload = {
        "surface": "problem",
        "type": "emphasize",
        "target": "problem-math-001",
        "emphasis_style": "highlight",
        "persistence": "trace",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        SyncVisualAction.model_validate(payload)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_schemas.py -k "sync_cue or sync_visual_action"'
```

Expected: collection or import failure because `NarrativeSyncCue` and `SyncVisualAction` do not exist.

- [ ] **Step 3: Add the schema types**

Add bounded strings and models in `app/schemas.py`:

```python
CueSpokenText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=90),
]


class SyncVisualAction(SchemaModel):
    surface: Literal["problem", "board"]
    type: Literal[
        "write",
        "transform",
        "focus",
        "emphasize",
        "annotate",
        "fade",
        "reveal",
        "clear_focus",
    ]
    target: GeneratedId
    content: Optional[NarrativeBoardContent] = None
    source: Optional[GeneratedId] = None
    relation_target: Optional[GeneratedId] = None
    annotation: Optional[
        Literal["underline", "arrow", "bracket", "label"]
    ] = None
    emphasis_style: Optional[
        Literal["highlight", "underline", "red"]
    ] = None
    persistence: Optional[Literal["transient", "trace"]] = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "SyncVisualAction":
        if self.surface == "problem" and self.type in {
            "write",
            "transform",
            "annotate",
        }:
            raise ValueError("problem actions cannot mutate source text")
        if self.type in {"write", "transform"} and not self.content:
            raise ValueError(f"{self.type} requires content")
        if self.type == "emphasize" and self.emphasis_style is None:
            raise ValueError("emphasize requires emphasis_style")
        if self.type != "emphasize" and self.persistence is not None:
            raise ValueError("persistence is only valid for emphasize")
        return self


class NarrativeSyncCue(SchemaModel):
    cue_id: GeneratedId
    spoken_text: CueSpokenText
    lead_actions: List[SyncVisualAction] = Field(
        default_factory=list,
        max_length=6,
    )
    start_actions: List[SyncVisualAction] = Field(
        default_factory=list,
        max_length=8,
    )
    end_actions: List[SyncVisualAction] = Field(
        default_factory=list,
        max_length=6,
    )


class RuntimeSyncCue(SchemaModel):
    cue_id: NonEmptyString
    spoken_text: NonEmptyString
    lead_actions: List[SyncVisualAction] = Field(default_factory=list)
    start_actions: List[SyncVisualAction] = Field(default_factory=list)
    end_actions: List[SyncVisualAction] = Field(default_factory=list)
    audio_url: Optional[NonEmptyString] = None
```

Keep `BoardAction` unchanged in this task so legacy lessons remain valid.

- [ ] **Step 4: Run schema tests and the full schema module**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_schemas.py'
```

Expected: all schema tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: add sync cue contracts"
```

---

### Task 2: Compile Stable Formula Focus Targets Without Model Offsets

**Files:**

- Create: `app/problem_focus.py`
- Create: `tests/test_problem_focus.py`
- Create: `tests/fixtures/problem-focus-cases.json`
- Modify: `app/schemas.py`
- Modify: `app/static/math-text.mjs`
- Modify: `tests/math-text.test.mjs`

- [ ] **Step 1: Add shared formula cases and failing Python/JavaScript tests**

Create `tests/fixtures/problem-focus-cases.json`:

```json
[
  {
    "name": "dollar_delimited_parameter_root",
    "source": "若$2n$ ($n\\ne 0$)是方程$x^2-2mx+2n=0$的根",
    "math": ["2n", "n\\ne 0", "x^2-2mx+2n=0"]
  },
  {
    "name": "parenthesis_and_display_delimiters",
    "source": "先看\\(x=2\\)，再看\\[\\frac{1}{2}\\]",
    "math": ["x=2", "\\frac{1}{2}"]
  },
  {
    "name": "malformed_delimiter_is_plain_text",
    "source": "未闭合的$x^2+1",
    "math": []
  },
  {
    "name": "escaped_currency_is_not_math",
    "source": "价格\\$100 和 \\$200",
    "math": []
  }
]
```

Add `tests/test_problem_focus.py`:

```python
import json
from pathlib import Path

from app.problem_focus import compile_problem_focus_targets


CASES = json.loads(
    Path("tests/fixtures/problem-focus-cases.json").read_text("utf-8")
)


def test_problem_focus_targets_match_shared_formula_cases():
    for case in CASES:
        targets = compile_problem_focus_targets(case["source"])
        assert [target.math_text for target in targets] == case["math"]
        assert [target.target_id for target in targets] == [
            f"problem-math-{index:03d}"
            for index in range(1, len(case["math"]) + 1)
        ]
```

Add a Node test in `tests/math-text.test.mjs` that loads the same JSON and asserts `problemFocusTargets(case.source)` returns identical `math_text` and IDs.

- [ ] **Step 2: Run both tests and verify RED**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_problem_focus.py'
node --test --test-name-pattern='problem focus targets' tests/math-text.test.mjs
```

Expected: Python import failure and JavaScript import failure because the compilers do not exist.

- [ ] **Step 3: Add the server target model and compiler**

Add to `app/schemas.py`:

```python
class ProblemFocusTarget(SchemaModel):
    target_id: NonEmptyString
    math_text: NonEmptyString
    display_mode: bool = False
    ordinal: int = Field(ge=1, le=64)
```

Create `app/problem_focus.py` with a bounded parser for `$...$`, `$$...$$`,
`\(...\)`, and `\[...\]`. Escaped dollar signs are ignored, mixed or unmatched
delimiters return no focus targets, and input over the existing 4096-character
math-rendering budget returns no targets.

The public function must have this exact signature:

```python
def compile_problem_focus_targets(
    source: str,
) -> list[ProblemFocusTarget]:
    tokens = _explicit_math_tokens(source)
    return [
        ProblemFocusTarget(
            target_id=f"problem-math-{index:03d}",
            math_text=token.content,
            display_mode=token.display_mode,
            ordinal=index,
        )
        for index, token in enumerate(tokens, start=1)
    ]
```

Do not accept model-provided offsets, selectors, HTML, or styles.

- [ ] **Step 4: Add the matching JavaScript target compiler**

Export from `app/static/math-text.mjs`:

```javascript
export function problemFocusTargets(value) {
  return mathSegments(value)
    .filter((segment) => segment.type === "math")
    .map((segment, index) => ({
      target_id: `problem-math-${String(index + 1).padStart(3, "0")}`,
      math_text: segment.value,
      display_mode: segment.displayMode,
      ordinal: index + 1,
    }));
}
```

The existing `mathSegments` parser remains the only JavaScript formula parser.
Do not add a second regular-expression path for highlights.

The Python parser must mirror the same delimiter state machine and the shared
fixtures are the cross-runtime contract. Add an invariant test that renders
each valid case in JavaScript and asserts the wrapper's
`data-focus-target`/math text pair equals the server-produced
`target_id`/`math_text` pair. If the parsers disagree, render the original
problem without focus wrappers and report the mismatch in development logs;
never highlight a different formula.

- [ ] **Step 5: Run focused and full formula tests**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_problem_focus.py tests/test_schemas.py'
npm test
```

Expected: Python focus tests and all Node tests pass; hostile KaTeX tests still show `trust: false`, finite `maxSize`, and no `innerHTML`.

- [ ] **Step 6: Commit**

```bash
git add app/problem_focus.py app/schemas.py app/static/math-text.mjs tests/test_problem_focus.py tests/fixtures/problem-focus-cases.json tests/math-text.test.mjs
git commit -m "feat: compile safe problem focus targets"
```

---

### Task 3: Make Sync Cues the Narrative Source of Truth

**Files:**

- Modify: `app/schemas.py`
- Modify: `app/prompts.py`
- Modify: `app/generation.py`
- Modify: `tests/generation_fakes.py`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_generation_agents.py`

- [ ] **Step 1: Write failing generation tests for Cue authority**

Add tests that construct a `NarrativeMoment` with two Cues and assert:

```python
def test_narrative_moment_derives_spoken_text_and_actions_from_cues():
    moment = NarrativeMoment.model_validate(
        {
            "moment_id": "substitute-root",
            "purpose": "把根代入原方程",
            "sync_cues": [
                {
                    "cue_id": "read-root",
                    "spoken_text": "因为二n是方程的根。",
                    "lead_actions": [
                        {
                            "surface": "problem",
                            "type": "emphasize",
                            "target": "problem-math-001",
                            "emphasis_style": "highlight",
                            "persistence": "trace",
                        }
                    ],
                    "start_actions": [],
                    "end_actions": [],
                },
                {
                    "cue_id": "write-substitution",
                    "spoken_text": "所以把x等于二n代入原方程。",
                    "lead_actions": [],
                    "start_actions": [
                        {
                            "surface": "board",
                            "type": "write",
                            "target": "substitution",
                            "content": "\\(x=2n\\)",
                        }
                    ],
                    "end_actions": [],
                },
            ],
            "layer": "base",
            "interaction_intent": None,
        }
    )

    assert moment.spoken_narration == (
        "因为二n是方程的根。所以把x等于二n代入原方程。"
    )
    assert len(moment.sync_cues) == 2
```

Add rejection tests for:

- LaTeX commands inside `spoken_text`;
- unknown problem target IDs;
- board `focus` before a matching `write` or `transform`;
- `end_actions` weakening a target that was not emphasized;
- every frozen-route `statement_after` missing from all Cue start actions.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_generation_agents.py -k "sync_cue or cue_route or cue_target"'
```

Expected: failures because `NarrativeMoment` still requires `narration` and `board_actions`.

- [ ] **Step 3: Migrate narrative models**

Change `NarrativeMoment` to:

```python
class NarrativeMoment(SchemaModel):
    moment_id: GeneratedId
    purpose: NarrativePurpose
    sync_cues: List[NarrativeSyncCue] = Field(
        min_length=1,
        max_length=5,
    )
    layer: NarrativeLayer = "base"
    interaction_intent: Optional[InteractionIntentText] = None

    @property
    def spoken_narration(self) -> str:
        return "".join(cue.spoken_text for cue in self.sync_cues)
```

Change `LessonMoment` to store `sync_cues` and derive `narration` and flattened
legacy `board_actions` as properties used only by compatibility checks.

Update every fixture in `tests/generation_fakes.py` so each prior moment becomes
one Cue. Do not manufacture `emphasize` actions merely to satisfy the new field.

- [ ] **Step 4: Pass deterministic problem targets to the Director**

In `app/generation.py`, compile targets once before Director generation:

```python
problem_focus_targets = compile_problem_focus_targets(problem.problem_text)
```

Pass only the model-safe projection to `director_prompt`:

```python
[
    {
        "target_id": target.target_id,
        "math_text": target.math_text,
        "display_mode": target.display_mode,
    }
    for target in problem_focus_targets
]
```

Update `DIRECTOR_SYSTEM` and `REVISION_SYSTEM` so:

- Cue `spoken_text` is natural Chinese without LaTeX commands;
- math in action `content` uses `\( ... \)` or `\[ ... \]`;
- problem actions may reference only supplied IDs;
- board focus/emphasis may reference only IDs created earlier by board
  `write`/`transform`;
- no arbitrary highlight markup such as `[[red:...]]`, HTML, CSS, selectors,
  character offsets, or colors is accepted;
- visual actions are optional and must add teaching information.

- [ ] **Step 5: Replace moment-level validators with Cue-aware validators**

Add helpers in `app/generation.py`:

```python
def _cue_actions(moment: NarrativeMoment):
    for cue in moment.sync_cues:
        yield from cue.lead_actions
        yield from cue.start_actions
        yield from cue.end_actions


def _moment_spoken_text(moment: NarrativeMoment) -> str:
    return "".join(cue.spoken_text for cue in moment.sync_cues)
```

Update route coverage, answer-leak detection, narrative-size validation,
method-introduction ordering, board-target lifecycle, and Reviewer evidence to
use these helpers. Route equations count only in board-surface `write` or
`transform` actions under `start_actions`.

- [ ] **Step 6: Run generation tests**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_generation.py tests/test_generation_agents.py'
```

Expected: all generation tests pass, including voice-only Cues and formula/highlight rejection tests.

- [ ] **Step 7: Commit**

```bash
git add app/schemas.py app/prompts.py app/generation.py tests/generation_fakes.py tests/test_generation.py tests/test_generation_agents.py
git commit -m "feat: generate cue-synchronized narratives"
```

---

### Task 4: Compile Cue-Based Runtime Lessons With Legacy Fields Derived

**Files:**

- Modify: `app/schemas.py`
- Modify: `app/compiler.py`
- Modify: `tests/test_compiler.py`
- Modify: `tests/test_schemas.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing compiler tests**

Add a compiler test asserting:

```python
lesson = LessonCompiler(lesson_id_factory=lambda: "lesson-sync").compile(
    problem(),
    cue_based_draft(),
    {"review_status": "approved"},
)

assert lesson.beats[2].sync_cues[0].cue_id == "read-root"
assert lesson.beats[2].narration == "".join(
    cue.spoken_text for cue in lesson.beats[2].sync_cues
)
assert lesson.beats[2].board_actions == [
    converted
    for cue in lesson.beats[2].sync_cues
    for action in cue.start_actions
    if (converted := _legacy_board_action(action)) is not None
]
assert lesson.problem_focus_targets[0].target_id == "problem-math-001"
```

Add a compatibility test that validates a `RuntimeBeat` with no `sync_cues`
and an existing Beat-level `audio_url`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_compiler.py tests/test_schemas.py -k "sync or legacy_runtime"'
```

Expected: failures because runtime models have no Cue or focus-target fields.

- [ ] **Step 3: Extend runtime schemas**

Change `RuntimeBeat`:

```python
class RuntimeBeat(SchemaModel):
    beat_id: NonEmptyString
    purpose: NonEmptyString
    narration: NonEmptyString
    board_actions: List[BoardAction]
    sync_cues: List[RuntimeSyncCue] = Field(default_factory=list)
    layer: LessonLayer
    interaction: Optional[Interaction] = None
    audio_url: Optional[NonEmptyString] = None
    next_beat_id: Optional[NonEmptyString] = None
```

Add `problem_focus_targets: List[ProblemFocusTarget]` to `RuntimeLesson`.

- [ ] **Step 4: Compile all Beat categories into Cues**

In `app/compiler.py`:

- opening, method introduction, summary, and transfer intro each become one
  deterministic Cue;
- generated moments preserve Director Cues;
- `narration` is the concatenation of Cue text;
- `board_actions` is derived only for old-reader compatibility;
- `problem_focus_targets` comes from
  `compile_problem_focus_targets(problem.problem_text)`;
- Beat/Cue IDs are stable and safe for audio asset names.

Use a helper with this interface:

```python
def _legacy_board_action(
    action: SyncVisualAction,
) -> BoardAction | None:
    if action.surface != "board":
        return None
    if action.type not in {"write", "transform", "focus", "reveal"}:
        return None
    return BoardAction(
        type=action.type,
        target=action.target,
        content=action.content,
        source=action.source,
        relation_target=action.relation_target,
        annotation=action.annotation,
    )


def _runtime_cue(
    cue_id: str,
    spoken_text: str,
    lead_actions: list[SyncVisualAction],
    start_actions: list[SyncVisualAction],
    end_actions: list[SyncVisualAction],
) -> RuntimeSyncCue:
    return RuntimeSyncCue(
        cue_id=cue_id,
        spoken_text=spoken_text,
        lead_actions=lead_actions,
        start_actions=start_actions,
        end_actions=end_actions,
    )
```

The helper remains module-local in `app/compiler.py`; the compiler test may
import it directly or compare against explicit `BoardAction(...)` values. Do
not add a conversion method to the Pydantic action model. Flatten only
non-`None` results and preserve Cue/action order.

- [ ] **Step 5: Verify public API privacy**

Update `tests/test_api.py` to assert Cue audio/actions are public, while
`reference_answer`, `reference_solution_text`, validation reports,
`expected_answer`, and private feedback remain hidden.

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_compiler.py tests/test_schemas.py tests/test_api.py'
```

Expected: all compiler, schema, and API tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/schemas.py app/compiler.py tests/test_compiler.py tests/test_schemas.py tests/test_api.py
git commit -m "feat: compile runtime sync cues"
```

---

### Task 5: Generate Volcengine Audio Per Cue With Bounded Concurrency

**Files:**

- Modify: `app/audio_service.py`
- Modify: `tests/test_tts_client.py`

- [ ] **Step 1: Write failing Cue audio tests**

Add tests that assert:

```python
voiced = run(
    LessonAudioService(client, tmp_path).attach_audio(cue_lesson())
)

assert [cue.audio_url for cue in voiced.beats[0].sync_cues] == [
    "/audio/lesson-001/beat-001-cue-001.mp3",
    "/audio/lesson-001/beat-001-cue-002.mp3",
]
assert client.texts[:2] == [
    voiced.beats[0].sync_cues[0].spoken_text,
    voiced.beats[0].sync_cues[1].spoken_text,
]
assert voiced.beats[0].audio_url is None
```

Add tests proving:

- at most three Cue syntheses are active;
- output order follows Cue order even when requests finish out of order;
- one failed Cue retries twice and then removes the whole lesson directory;
- legacy Beat audio still uses `{beat_id}.mp3`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_tts_client.py -k "cue or bounded_concurrency or legacy_beat_audio"'
```

Expected: Cue URLs remain `None` because `attach_audio` voices only Beat narration.

- [ ] **Step 3: Implement bounded Cue synthesis**

Add a service-level semaphore:

```python
cue_semaphore = asyncio.Semaphore(3)


async def voice_cue(beat_id, cue):
    async with cue_semaphore:
        audio_url = await self._write(
            lesson.lesson_id,
            f"{beat_id}-{cue.cue_id}",
            cue.spoken_text,
        )
    return cue.model_copy(update={"audio_url": audio_url})
```

Create tasks for all Cues, gather them in source order, and reconstruct Beats
without mutating the input lesson. If `beat.sync_cues` is empty, retain the
existing Beat-level path. Keep interaction hint/feedback audio behavior and
the all-or-nothing directory cleanup.

- [ ] **Step 4: Run audio and Volcengine tests**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_tts_client.py tests/test_volcengine_tts_client.py'
```

Expected: all Cue, legacy, retry, path-safety, and Volcengine frame tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/audio_service.py tests/test_tts_client.py
git commit -m "feat: generate cue-level lesson audio"
```

---

### Task 6: Render Formula-Safe Problem and Board Emphasis

**Files:**

- Modify: `app/static/math-text.mjs`
- Modify: `app/static/runtime-core.mjs`
- Modify: `app/static/lesson.js`
- Modify: `app/static/styles.css`
- Modify: `tests/math-text.test.mjs`
- Modify: `tests/runtime-core.test.mjs`
- Modify: `tests/test_static_pages.py`

- [ ] **Step 1: Write failing renderer and reducer tests**

Add Node tests that assert:

```javascript
const visual = applySyncVisualAction(emptyVisualState(), {
  surface: "problem",
  type: "emphasize",
  target: "problem-math-001",
  emphasis_style: "highlight",
  persistence: "trace",
});

assert.deepEqual(visual.problem.get("problem-math-001"), {
  style: "highlight",
  strength: "active",
  persistence: "trace",
});
```

Then apply `clear_focus` and assert the target becomes `strength: "trace"`,
while an ordinary board object remains unchanged.

Add a math renderer test that passes:

```javascript
String.raw`若$2n$且$n\ne0$`
```

and asserts:

- two focus wrappers receive `data-focus-target`;
- each wrapper contains a KaTeX render call;
- no `$`, `\(`, or `\)` delimiter is visible;
- renderer options remain `trust: false`;
- hostile `emphasis_style: "background:url(javascript:...)"` is rejected before rendering;
- implementation never assigns `innerHTML`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
node --test --test-name-pattern='problem emphasis|selective trace|focus wrapper' tests/math-text.test.mjs tests/runtime-core.test.mjs
```

Expected: failures because there is no visual state reducer or focus-wrapper renderer.

- [ ] **Step 3: Add a surface-aware visual reducer**

Export from `app/static/runtime-core.mjs`:

```javascript
export function emptyVisualState() {
  return {
    board: new Map(),
    problem: new Map(),
  };
}


export function applySyncVisualAction(current, action) {
  const next = {
    board: cloneBoard(current.board),
    problem: new Map(current.problem),
  };
  if (action.surface === "problem") {
    return applyProblemAction(next, action);
  }
  next.board = applyBoardAction(next.board, action);
  return applyBoardEmphasis(next, action);
}
```

Map only enum values to classes:

```javascript
const EMPHASIS_CLASSES = {
  highlight: "is-highlighted",
  underline: "is-underlined",
  red: "is-red-emphasis",
};
```

Never concatenate model text into a class name or style attribute.

Implement the referenced helpers with these exact state rules:

- `applyProblemAction` accepts only `emphasize`, `focus`, `fade`, and
  `clear_focus`; it ignores unknown target IDs and throws for an unknown action
  type in development/test mode.
- `applyBoardEmphasis` changes only a board item that already exists. It never
  creates board content and never changes the equation/string stored on the
  item.
- `emphasize` stores `{style, strength: "active", persistence}`.
- `clear_focus` deletes transient emphasis; trace emphasis becomes
  `{...previous, strength: "trace"}`.
- `fade` weakens only the named target.
- `write` and `transform` pass through the existing `applyBoardAction`, then
  initialize emphasis only when the action explicitly carries an allowed
  emphasis field.

Add reducer tests for every row above, including unknown problem targets,
focus-before-write, and an action carrying a valid style but an invalid type.

- [ ] **Step 4: Wrap KaTeX output without changing formula syntax**

Add `renderProblemMathText` to `app/static/math-text.mjs`. It must:

- call the existing `mathSegments`;
- assign deterministic IDs only to math segments;
- create one outer `<span data-focus-target="problem-math-001">`;
- call `katex.render` inside that wrapper using the existing safe options;
- apply only allow-listed CSS classes from reducer state;
- use text nodes for non-math text;
- fall back to the original text if delimiters are malformed.

Do not implement highlight markup inside LaTeX and do not use KaTeX `\htmlClass`,
`\style`, `trust: true`, or `innerHTML`.

- [ ] **Step 5: Add active and trace visual styles**

In `app/static/styles.css`, define separate active/trace classes:

```css
.focus-target.is-highlighted.is-active {
  background: var(--focus-highlight-active);
}

.focus-target.is-highlighted.is-trace {
  background: var(--focus-highlight-trace);
}

.focus-target.is-underlined.is-active {
  text-decoration: underline 0.18em var(--focus-underline-active);
}

.focus-target.is-underlined.is-trace {
  text-decoration: underline 0.1em var(--focus-underline-trace);
}

.focus-target.is-red-emphasis.is-active {
  color: var(--focus-red-active);
}

.focus-target.is-red-emphasis.is-trace {
  color: var(--focus-red-trace);
}
```

Trace styles retain the original emphasis family; they do not turn every prior
object green.

- [ ] **Step 6: Run renderer, static, and accessibility tests**

Run:

```bash
npm test
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_static_pages.py'
node --check app/static/lesson.js
```

Expected: all formula, reducer, DOM safety, static-page, and accessibility tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/static/math-text.mjs app/static/runtime-core.mjs app/static/lesson.js app/static/styles.css tests/math-text.test.mjs tests/runtime-core.test.mjs tests/test_static_pages.py
git commit -m "feat: render safe synchronized emphasis"
```

---

### Task 7: Play Cue Audio and Actions as One Pausable Timeline

**Files:**

- Create: `app/static/cue-player.mjs`
- Create: `tests/cue-player.test.mjs`
- Modify: `app/static/lesson.js`
- Modify: `app/static/runtime-core.mjs`
- Modify: `tests/runtime-core.test.mjs`
- Modify: `tests/test_static_pages.py`

- [ ] **Step 1: Write failing Cue-player tests**

Use fake audio and timers to verify this exact event order:

```javascript
assert.deepEqual(events, [
  "lead:problem-root",
  "wait:200",
  "audio:play:cue-001.mp3",
  "start:write-substitution",
  "audio:ended:cue-001.mp3",
  "end:weaken-root",
  "lead:problem-equation",
  "wait:200",
  "audio:play:cue-002.mp3",
  "start:write-result",
]);
```

Add separate tests proving:

- a Cue with no visual actions still plays;
- pause during the 200ms lead delay freezes the timer;
- pause during audio pauses the audio;
- resume continues the same Cue;
- stop prevents stale callbacks from applying actions;
- replay restores the Beat snapshot and restarts Cue 1;
- a missing Cue audio URL uses text-duration fallback but retains lead/start/end order;
- interaction becomes visible only after the final Cue.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
node --test tests/cue-player.test.mjs
```

Expected: module-not-found failure for `app/static/cue-player.mjs`.

- [ ] **Step 3: Implement `CuePlayer`**

Create a dependency-injected class:

```javascript
export class CuePlayer {
  constructor({
    leadMs = 200,
    createAudio,
    createTimeline,
    applyActions,
    fallbackDuration,
    onCueText,
    onBeatComplete,
    onAudioUnavailable,
  }) {
    this.leadMs = leadMs;
    this.createAudio = createAudio;
    this.createTimeline = createTimeline;
    this.applyActions = applyActions;
    this.fallbackDuration = fallbackDuration;
    this.onCueText = onCueText;
    this.onBeatComplete = onBeatComplete;
    this.onAudioUnavailable = onAudioUnavailable;
    this.token = 0;
    this.currentAudio = null;
    this.timeline = null;
  }
}
```

Implement `playBeat`, `playCue`, `pause`, `resume`, and `stop`. Every async
callback must compare a monotonically increasing token before changing state.

The method contracts are:

```javascript
playBeat(beat, { snapshot }) // resets index, stores snapshot, starts Cue 0
playCue(index, token)        // lead actions -> pausable 200ms -> audio/start
pause()                      // pauses current delay or audio, applies no actions
resume()                     // resumes exactly the suspended delay/audio
stop()                       // increments token, cancels timer/audio, no callback
replay()                     // stop -> restore snapshot -> playBeat from Cue 0
```

`ended` applies `end_actions` exactly once, then advances to the next Cue.
`error` switches that Cue to the same pausable text-duration timeline and must
not replay `lead_actions` or `start_actions`. `onBeatComplete` fires exactly
once after the last Cue's end actions.

- [ ] **Step 4: Integrate the player into the classroom**

In `app/static/lesson.js`:

- if `beat.sync_cues.length > 0`, use `CuePlayer`;
- otherwise keep the existing Beat-level legacy path;
- update narration text on every Cue;
- route all actions through `applySyncVisualAction`;
- render problem focus and boards after every action batch;
- pre-create the next Cue `Audio` object after current audio starts;
- show interaction only from `onBeatComplete`;
- reset visual state and Beat snapshot on replay.

Do not use `scheduleBoardActions` for Cue lessons.

- [ ] **Step 5: Run Cue and runtime tests**

Run:

```bash
node --test tests/cue-player.test.mjs tests/runtime-core.test.mjs
npm test
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_static_pages.py'
```

Expected: all Cue sequence, pause/replay, legacy runtime, and static tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/static/cue-player.mjs app/static/lesson.js app/static/runtime-core.mjs tests/cue-player.test.mjs tests/runtime-core.test.mjs tests/test_static_pages.py
git commit -m "feat: synchronize cue audio and board actions"
```

---

### Task 8: Validate the Exact Parameter-Root Lesson End to End

**Files:**

- Modify: `scripts/smoke_live.py`
- Modify: `app/static/lesson.html`
- Modify: `app/static/lesson.js`
- Modify: `README.md`
- Test: `tests/test_static_pages.py`

- [ ] **Step 1: Add deterministic smoke assertions**

Extend the grounded parameter-root smoke so it verifies:

```python
assert all(beat.sync_cues for beat in lesson.beats)
assert all(
    cue.audio_url
    for beat in lesson.beats
    for cue in beat.sync_cues
)
assert any(
    action.surface == "problem"
    and action.target == "problem-math-001"
    for beat in lesson.beats
    for cue in beat.sync_cues
    for action in cue.lead_actions
)
assert any(
    action.surface == "board"
    and action.type in {"write", "transform"}
    and "4n" in (action.content or "")
    for beat in lesson.beats
    for cue in beat.sync_cues
    for action in cue.start_actions
)
```

Keep smoke output bounded: lesson ID, Beat/Cue counts, interaction kinds,
audio-ready boolean, conclusion-present boolean, and review status only.

- [ ] **Step 2: Bump static module versions**

Because the browser previously retained stale formula-rendering modules, update
both:

```html
<script type="module" src="/static/lesson.js?v=20260807-2"></script>
```

and:

```javascript
import {
  mathTextToPlainText,
  renderMathText,
  renderProblemMathText,
} from "./math-text.mjs?v=20260807-2";
import { CuePlayer } from "./cue-player.mjs?v=20260807-2";
```

Update `tests/test_static_pages.py` to require the new version strings.

- [ ] **Step 3: Run the full offline suite**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests'
npm test
node --check app/static/lesson.js
git diff --check
```

Expected: zero Python failures, zero Node failures, valid JavaScript syntax,
and no whitespace errors.

- [ ] **Step 4: Run the exact live generation**

Run:

```bash
bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && set -a && source .env && set +a && python scripts/smoke_live.py --grounded-parameter-root'
```

Expected: exit 0 with `review_status=approved`, all Cue audio ready, choice-only
interactions, and the reference conclusion present. Do not print provider
responses or secrets.

- [ ] **Step 5: Perform browser acceptance on the generated lesson**

Use a fresh generated lesson and verify at Pad landscape size:

1. before “\(2n\) 是方程的根”, the condition is not highlighted;
2. at that Cue lead phase, the condition highlights about 0.2 seconds early;
3. when the substitution Cue begins, \(x=2n\) and the substitution mark appear;
4. when the result Cue begins, \(4n^2-4mn+2n=0\) appears;
5. when the nonzero Cue begins, \(n\ne0\) focuses and division by \(n\) appears;
6. previous explicit emphasis weakens in its own style;
7. ordinary board objects do not change color;
8. pause freezes audio and actions;
9. replay restores the pre-Beat snapshot;
10. interaction waits for the final Cue;
11. no raw `$`, `\frac`, `\ne`, highlight syntax, or target ID is visible.

- [ ] **Step 6: Update README verification boundaries**

Document:

- Cue-level Volcengine audio and semantic synchronization;
- formula rendering and emphasis use separate allow-listed contracts;
- fixed smoke/browser evidence does not prove all题型 or learning effect;
- runtime audio fallback is readable degradation, not synchronized-audio success.

- [ ] **Step 7: Commit**

```bash
git add scripts/smoke_live.py app/static/lesson.html app/static/lesson.js tests/test_static_pages.py README.md
git commit -m "test: verify synchronized parameter-root lesson"
```

---

## Final Verification

- [ ] Run the full Python suite in `conda activate general`.
- [ ] Run the full Node suite.
- [ ] Run JavaScript syntax checks.
- [ ] Run `git diff --check`.
- [ ] Run the live parameter-root smoke with configured model and Volcengine TTS.
- [ ] Open the generated lesson and complete the 11 browser acceptance checks.
- [ ] Confirm `git status -sb` contains no implementation files outside the planned scope.
