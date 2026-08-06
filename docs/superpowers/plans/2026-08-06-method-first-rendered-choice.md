# Method-First Rendered Choice Lessons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every newly generated lesson explain its selected method before operating, render mathematical notation throughout the classroom, and replace formula typing with diagnostic choices including the near-transfer check.

**Architecture:** Extend the draft contract with a structured method introduction and choice feedback, reject new input-style interactions in the deterministic generation gate, and compile the method introduction plus near-transfer choices into the existing beat runtime. Add a small, isolated KaTeX-backed math-text renderer used by the current vanilla JavaScript classroom, while retaining read compatibility for old `expression` and `transfer` lessons.

**Tech Stack:** Python 3.9, FastAPI, Pydantic v2, SymPy, async OpenAI-compatible JSON generation, Volcengine TTS, vanilla ES modules, KaTeX 0.18.1 local static assets, pytest, Node.js built-in test runner.

---

### Task 1: Add the method-introduction and diagnostic-choice contracts

**Files:**
- Modify: `app/schemas.py`
- Modify: `tests/test_schemas.py`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_tts_client.py`

- [ ] **Step 1: Write failing schema tests**

Add imports for `MethodIntroduction` and `TransferOption`, then add:

```python
def test_method_introduction_preserves_the_teaching_contract():
    introduction = MethodIntroduction(
        method_name="配方法",
        student_definition="把二次式整理成完全平方的形式。",
        target_form=r"\((x-a)^2=b\)",
        why_it_helps="开平方后可以转成两个一次方程。",
    )

    assert introduction.method_name == "配方法"
    assert introduction.target_form == r"\((x-a)^2=b\)"


def test_choice_option_supports_optional_legacy_feedback():
    legacy = InteractionOption(option_id="A", label="9")
    diagnostic = InteractionOption(
        option_id="B",
        label=r"\(9\)",
        feedback="对，一次项系数一半的平方是九。",
    )

    assert legacy.feedback is None
    assert diagnostic.feedback.startswith("对")
    assert diagnostic.feedback_audio_url is None


def test_transfer_item_accepts_diagnostic_options():
    item = TransferItem(
        problem_text="解方程 x^2-8x+12=0",
        expected_answer="x=2 或 x=6",
        method_signal="先把一次项系数取一半再平方。",
        options=[
            TransferOption(
                option_id="A",
                label=r"\(x=2\) 或 \(x=6\)",
                canonical_answer="x=2 或 x=6",
                feedback="对，配方后开平方得到这两个解。",
            ),
            TransferOption(
                option_id="B",
                label=r"\(x=-2\) 或 \(x=-6\)",
                canonical_answer="x=-2 或 x=-6",
                feedback="符号发生了变化，再检查平方根分支。",
            ),
            TransferOption(
                option_id="C",
                label=r"\(x=4\)",
                canonical_answer="x=4",
                feedback="开平方会产生两个分支，不能只保留中点。",
            ),
        ],
        correct_option_id="A",
    )

    assert item.correct_option_id == "A"
    assert item.options[0].canonical_answer == "x=2 或 x=6"


def test_transfer_item_keeps_legacy_empty_options_compatible():
    item = TransferItem(
        problem_text="解方程 x+2=0",
        expected_answer="x=-2",
        method_signal="等式两边同时减二。",
    )

    assert item.options == []
    assert item.correct_option_id is None
```

Add validation tests proving duplicate option ids, fewer than three new transfer options,
and a missing `correct_option_id` are rejected.

- [ ] **Step 2: Run the schema tests and verify RED**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_schemas.py -k \
  "method_introduction or choice_option_supports or transfer_item"
```

Expected: failures because the new models and fields do not exist.

- [ ] **Step 3: Implement the schema types**

Add to `app/schemas.py`:

```python
class MethodIntroduction(SchemaModel):
    method_name: NonEmptyString
    student_definition: NonEmptyString
    target_form: NonEmptyString
    why_it_helps: NonEmptyString


class InteractionOption(SchemaModel):
    option_id: NonEmptyString
    label: NonEmptyString
    feedback: Optional[NonEmptyString] = None
    feedback_audio_url: Optional[NonEmptyString] = None


class TransferOption(SchemaModel):
    option_id: NonEmptyString
    label: NonEmptyString
    canonical_answer: NonEmptyString
    feedback: NonEmptyString
```

Extend `TransferItem`:

```python
class TransferItem(SchemaModel):
    problem_text: ProblemText
    expected_answer: NonEmptyString
    method_signal: NonEmptyString
    options: List[TransferOption] = Field(default_factory=list)
    correct_option_id: Optional[NonEmptyString] = None

    @model_validator(mode="after")
    def validate_diagnostic_options(self) -> "TransferItem":
        if not self.options:
            if self.correct_option_id is not None:
                raise ValueError(
                    "legacy transfer without options cannot name a correct option"
                )
            return self
        if not 3 <= len(self.options) <= 4:
            raise ValueError("transfer choices require three or four options")
        option_ids = [option.option_id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("transfer option ids must be unique")
        if self.correct_option_id not in option_ids:
            raise ValueError(
                "correct_option_id must match a transfer option"
            )
        if len({option.label for option in self.options}) != len(self.options):
            raise ValueError("transfer option labels must be unique")
        return self
```

Add `method_introduction: MethodIntroduction` before `math_steps` in `LessonDraft`.
Update `valid_draft()` and direct `LessonDraft`/`RuntimeLesson` fixtures with a complete
introduction and valid three-option transfer choice.

- [ ] **Step 4: Run the selected schema tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Run fixture-dependent tests**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_schemas.py tests/test_generation.py \
  tests/test_compiler.py tests/test_tts_client.py
```

Expected: all tests pass after fixtures adopt the new required `method_introduction`.

- [ ] **Step 6: Commit the schema contract**

```bash
git add app/schemas.py tests/test_schemas.py tests/test_generation.py \
  tests/test_tts_client.py
git commit -m "feat: define method-first choice lessons"
```

### Task 2: Enforce method-first and choice-first generation

**Files:**
- Modify: `app/prompts.py`
- Modify: `app/generation.py`
- Modify: `tests/test_generation.py`

- [ ] **Step 1: Write failing generation-gate tests**

Add:

```python
def test_generation_rejects_required_method_name_mismatch():
    draft = valid_draft()
    draft["method_introduction"]["method_name"] = "公式法"
    client = FakeClient([draft, copy.deepcopy(draft)])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="方法介绍"):
        asyncio.run(service.generate(problem(required_method="complete_the_square")))


@pytest.mark.parametrize("kind", ["expression", "transfer"])
def test_generation_rejects_new_formula_input_interactions(kind):
    draft = valid_draft()
    draft["moments"][0]["interaction"] = {
        "interaction_id": "typed-answer",
        "kind": kind,
        "prompt": "请输入结果。",
        "expected_answer": (
            "(x-3)^2" if kind == "expression" else "x=1 或 x=5"
        ),
        "hints": ["重新观察题目。"],
        "explanation_after_correct": "正确。",
    }
    client = FakeClient([draft, copy.deepcopy(draft)])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="选择或点选"):
        asyncio.run(service.generate(problem()))


def test_generation_rejects_choice_without_diagnostic_feedback():
    draft = valid_draft()
    draft["moments"][0]["interaction"]["options"][0]["feedback"] = None
    client = FakeClient([draft, copy.deepcopy(draft)])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="诊断反馈"):
        asyncio.run(service.generate(problem()))


def test_generation_rejects_invalid_transfer_distractor():
    draft = valid_draft()
    draft["transfer_item"]["options"][1]["canonical_answer"] = (
        draft["transfer_item"]["expected_answer"]
    )
    client = FakeClient([draft, copy.deepcopy(draft)])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="近迁移选项"):
        asyncio.run(service.generate(problem()))
```

Ensure `problem()` accepts overrides or build the required-method `ProblemInput` directly.

- [ ] **Step 2: Run the generation tests and verify RED**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_generation.py -k \
  "method_name_mismatch or formula_input or diagnostic_feedback or invalid_transfer_distractor"
```

Expected: failures because the deterministic quality gates are absent.

- [ ] **Step 3: Add deterministic validation**

In `LessonGenerationService._validate_draft`, add:

```python
required_method_names = {
    "factor": "因式分解法",
    "quadratic_formula": "公式法",
    "complete_the_square": "配方法",
}
required_name = required_method_names.get(problem.required_method)
if (
    required_name is not None
    and draft.method_introduction.method_name != required_name
):
    raise LessonQualityError("讲解的方法介绍与指定方法不一致。")

for interaction in interactions:
    if interaction.kind in {"expression", "transfer"}:
        raise LessonQualityError(
            "新讲解中的数学互动必须使用选择或点选。"
        )
    if interaction.kind == "choice":
        if not 3 <= len(interaction.options) <= 4:
            raise LessonQualityError(
                "选择互动必须提供三到四个诊断选项。"
            )
        if any(option.feedback is None for option in interaction.options):
            raise LessonQualityError(
                "选择互动缺少逐项诊断反馈。"
            )
```

After validating the transfer problem, require options for newly generated drafts and
validate every `canonical_answer` with `MathEngine.answers_equivalent`. Exactly one
option must be equivalent to `expected_answer`, and it must match `correct_option_id`.
Convert all `MathValidationError` details to the public-safe message
`"近迁移选项未通过数学验证。"` without including model output.

- [ ] **Step 4: Strengthen Director, Reviewer, and revision prompts**

Replace the input-field rule in `DIRECTOR_SYSTEM` with explicit requirements:

```text
- method_introduction 在第一次实质性代数变形前解释方法名称、学生定义、目标形式和作用；
- 指定配方法时，第一项教学重点必须是“配方法”，随后解释完全平方目标；
- 板书、互动和总结中的数学片段使用 \( ... \) 或 \[ ... \]；
- narration 使用自然口语说明数学含义，不包含 LaTeX 命令；
- 自动判分互动只使用 choice 或 point_select；
- choice 提供三至四个互异选项，每项都有针对该思路的 feedback；
- transfer_item 提供三至四个 TransferOption，canonical_answer 使用数学引擎可验证的纯文本。
```

Add matching Reviewer `must_fix` criteria and repeat the contract in
`REVISION_SYSTEM`. Remove the Director instruction encouraging `expression`.

- [ ] **Step 5: Run generation tests and verify GREEN**

Run the command from Step 2, followed by:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_generation.py
```

Expected: the selected tests and the complete generation suite pass.

- [ ] **Step 6: Commit the quality gates**

```bash
git add app/prompts.py app/generation.py tests/test_generation.py
git commit -m "feat: enforce method-first diagnostic lessons"
```

### Task 3: Compile the method introduction and near-transfer choice

**Files:**
- Modify: `app/compiler.py`
- Modify: `app/api.py`
- Modify: `tests/test_compiler.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing compiler and privacy tests**

Update the expected beat count and add:

```python
def test_compiler_places_method_introduction_before_solution_moments():
    draft = valid_draft()
    lesson = compile_lesson()
    method_beat = lesson.beats[1]

    assert method_beat.purpose == "先认识方法"
    assert method_beat.layer == "micro_explanation"
    assert method_beat.narration.startswith("今天用配方法")
    assert [
        action.model_dump(exclude_none=True)
        for action in method_beat.board_actions
    ] == [
        {
            "type": "write",
            "target": "method_name",
            "content": "配方法",
        },
        {"type": "focus", "target": "method_name"},
        {
            "type": "write",
            "target": "method_target_form",
            "content": draft["method_introduction"]["target_form"],
        },
    ]
    assert lesson.beats[2].purpose == draft["moments"][0]["purpose"]


def test_compiler_turns_near_transfer_into_a_choice():
    draft = valid_draft()
    transfer = compile_lesson().beats[-1].interaction

    assert transfer.kind == "choice"
    assert transfer.expected_answer == (
        draft["transfer_item"]["correct_option_id"]
    )
    assert [option.option_id for option in transfer.options] == [
        option["option_id"] for option in draft["transfer_item"]["options"]
    ]
    assert all(option.feedback for option in transfer.options)
```

In `tests/test_api.py`, assert public transfer options omit `canonical_answer` while
retaining `option_id`, `label`, and `feedback`.

- [ ] **Step 2: Run compiler/API tests and verify RED**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_compiler.py tests/test_api.py
```

Expected: failures because no method beat exists and transfer still compiles as a text input.

- [ ] **Step 3: Compile the method beat**

In `LessonCompiler.compile`, insert after the original-problem beat:

```python
introduction = draft.method_introduction
beats.append(
    RuntimeBeat(
        beat_id="pending",
        purpose="先认识方法",
        narration=(
            f"今天用{introduction.method_name}。"
            f"{introduction.student_definition}"
            f"{introduction.why_it_helps}"
        ),
        board_actions=[
            BoardAction(
                type="write",
                target="method_name",
                content=introduction.method_name,
            ),
            BoardAction(type="focus", target="method_name"),
            BoardAction(
                type="write",
                target="method_target_form",
                content=introduction.target_form,
            ),
        ],
        layer="micro_explanation",
    )
)
```

The existing runtime clears temporary layers before moving to the next beat, so the first
solution moment returns to the original base board without a new action type.

- [ ] **Step 4: Compile the new near-transfer choice with a legacy fallback**

When `transfer_item.options` is nonempty, map each `TransferOption` to:

```python
InteractionOption(
    option_id=option.option_id,
    label=option.label,
    feedback=option.feedback,
)
```

and compile `Interaction(kind="choice", expected_answer=correct_option_id, ...)`.
When the option list is empty, preserve the current `kind="transfer"` compilation path.

- [ ] **Step 5: Strip canonical answers from the public payload**

In `public_lesson_payload`, after removing `transfer_item.expected_answer`, loop over
`payload["transfer_item"].get("options", [])` and remove `canonical_answer`. Continue
removing each compiled interaction's `expected_answer`.

- [ ] **Step 6: Run compiler/API tests and verify GREEN**

Run the command from Step 2.

Expected: all compiler and API tests pass.

- [ ] **Step 7: Commit compilation and privacy**

```bash
git add app/compiler.py app/api.py tests/test_compiler.py tests/test_api.py
git commit -m "feat: compile method-first choice lessons"
```

### Task 4: Generate and play option-specific feedback audio

**Files:**
- Modify: `app/audio_service.py`
- Modify: `tests/test_tts_client.py`

- [ ] **Step 1: Write a failing audio test**

Change the test lesson's interaction to a choice containing three diagnostic options, then
add:

```python
def test_audio_service_writes_feedback_for_every_choice_option(tmp_path):
    lesson = runtime_lesson_with_choice()
    client = FakeSpeechClient()

    voiced = run(LessonAudioService(client, tmp_path).attach_audio(lesson))
    options = voiced.beats[1].interaction.options

    assert [option.feedback_audio_url for option in options] == [
        "/audio/lesson-001/beat-002-option-1.mp3",
        "/audio/lesson-001/beat-002-option-2.mp3",
        "/audio/lesson-001/beat-002-option-3.mp3",
    ]
    assert [option.feedback for option in options] == client.texts[-3:]
```

Keep the existing hint and correct-feedback test using its current free-text fixture so
legacy audio behavior remains covered.

- [ ] **Step 2: Run the audio test and verify RED**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_tts_client.py::test_audio_service_writes_feedback_for_every_choice_option
```

Expected: failure because option feedback audio is not synthesized.

- [ ] **Step 3: Synthesize option feedback**

Inside `LessonAudioService.attach_audio`, after hint audio generation:

```python
voiced_options = []
for index, option in enumerate(interaction.options, start=1):
    feedback_audio_url = None
    if option.feedback:
        feedback_audio_url = await self._write(
            lesson.lesson_id,
            f"{beat.beat_id}-option-{index}",
            option.feedback,
        )
    voiced_options.append(
        option.model_copy(
            update={"feedback_audio_url": feedback_audio_url}
        )
    )
```

Include `"options": voiced_options` in the interaction copy update. Use the numeric index,
not model-supplied `option_id`, in asset paths.

- [ ] **Step 4: Run the complete audio tests and verify GREEN**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_tts_client.py tests/test_volcengine_tts_client.py
```

Expected: all TTS and audio-service tests pass.

- [ ] **Step 5: Commit option feedback audio**

```bash
git add app/audio_service.py tests/test_tts_client.py
git commit -m "feat: voice diagnostic option feedback"
```

### Task 5: Add a local, safe KaTeX math-text module

**Files:**
- Create: `package.json`
- Create: `package-lock.json`
- Modify: `.gitignore`
- Create: `app/static/vendor/katex/katex.mjs`
- Create: `app/static/vendor/katex/katex.min.css`
- Create: `app/static/vendor/katex/fonts/*`
- Create: `app/static/vendor/katex/LICENSE`
- Create: `app/static/math-text.mjs`
- Create: `tests/math-text.test.mjs`
- Modify: `tests/test_static_pages.py`

- [ ] **Step 1: Write failing pure parsing tests**

Create `tests/math-text.test.mjs`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";

import {
  mathSegments,
  normalizeLegacyMath,
} from "../app/static/math-text.mjs";


test("explicit inline and display delimiters become math segments", () => {
  assert.deepEqual(
    mathSegments("目标是 \\((x-a)^2=b\\)，再看 \\[x^2=4\\]"),
    [
      { kind: "text", value: "目标是 " },
      { kind: "math", value: "(x-a)^2=b", displayMode: false },
      { kind: "text", value: "，再看 " },
      { kind: "math", value: "x^2=4", displayMode: true },
    ],
  );
});


test("legacy equation suffix is isolated without changing Chinese text", () => {
  assert.equal(
    normalizeLegacyMath("解方程 x^2-6x+5=0"),
    "解方程 \\(x^2-6x+5=0\\)",
  );
});


test("plain teaching prose remains plain text", () => {
  assert.deepEqual(
    mathSegments("先观察一次项系数。"),
    [{ kind: "text", value: "先观察一次项系数。" }],
  );
});
```

- [ ] **Step 2: Run the Node test and verify RED**

Run:

```bash
node --test tests/math-text.test.mjs
```

Expected: module-not-found failure because `math-text.mjs` does not exist.

- [ ] **Step 3: Add the package contract and vendor KaTeX**

Create `package.json`:

```json
{
  "name": "shiguang-ai-math-demo",
  "private": true,
  "scripts": {
    "test": "node --test tests/*.test.mjs"
  },
  "dependencies": {
    "katex": "0.18.1"
  }
}
```

Add `node_modules/` to `.gitignore`, then run:

```bash
npm install
mkdir -p app/static/vendor/katex
cp node_modules/katex/dist/katex.mjs app/static/vendor/katex/katex.mjs
cp node_modules/katex/dist/katex.min.css app/static/vendor/katex/katex.min.css
cp -R node_modules/katex/dist/fonts app/static/vendor/katex/fonts
cp node_modules/katex/LICENSE app/static/vendor/katex/LICENSE
```

The checked-in CSS, module, fonts, and license make classroom playback independent of CDN
availability. `package-lock.json` pins the source package.

- [ ] **Step 4: Implement segmentation, fallback, and safe DOM rendering**

Create `app/static/math-text.mjs` exporting:

```javascript
import katex from "./vendor/katex/katex.mjs";

const LEGACY_EQUATION = /([A-Za-z0-9()[\]{}^*+\-÷×/.\s]+=[A-Za-z0-9()[\]{}^*+\-÷×/.\s]+)/;

export function normalizeLegacyMath(value = "") {
  const text = String(value);
  if (/\\\(|\\\[/.test(text)) return text;
  return text.replace(LEGACY_EQUATION, (match) => `\\(${match.trim()}\\)`);
}

export function mathSegments(value = "") {
  const normalized = normalizeLegacyMath(value);
  const pattern = /\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]/g;
  const segments = [];
  let cursor = 0;
  for (const match of normalized.matchAll(pattern)) {
    if (match.index > cursor) {
      segments.push({
        kind: "text",
        value: normalized.slice(cursor, match.index),
      });
    }
    segments.push({
      kind: "math",
      value: match[1] ?? match[2],
      displayMode: match[2] !== undefined,
    });
    cursor = match.index + match[0].length;
  }
  if (cursor < normalized.length) {
    segments.push({ kind: "text", value: normalized.slice(cursor) });
  }
  return segments.length
    ? segments
    : [{ kind: "text", value: normalized }];
}

export function renderMathText(container, value = "") {
  container.replaceChildren();
  for (const segment of mathSegments(value)) {
    if (segment.kind === "text") {
      container.append(document.createTextNode(segment.value));
      continue;
    }
    const span = document.createElement("span");
    span.className = segment.displayMode ? "math-display" : "math-inline";
    try {
      katex.render(segment.value, span, {
        displayMode: segment.displayMode,
        throwOnError: false,
        trust: false,
        strict: "warn",
      });
    } catch {
      span.textContent = segment.value;
      span.classList.add("math-fallback");
    }
    container.append(span);
  }
}
```

During GREEN, adjust only the legacy-equation regular expression if the explicit tests reveal
whitespace capture at the Chinese boundary; do not add a general natural-language parser.

- [ ] **Step 5: Run the math-text tests and verify GREEN**

Run:

```bash
npm test
```

Expected: the new math-text tests and existing runtime-core tests pass.

- [ ] **Step 6: Add static asset contract tests**

In `tests/test_static_pages.py`, assert the local module and assets are served:

```python
assert client.get("/static/math-text.mjs").status_code == 200
assert client.get("/static/vendor/katex/katex.mjs").status_code == 200
assert client.get("/static/vendor/katex/katex.min.css").status_code == 200
```

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_static_pages.py
```

Expected: the module and assets are served.

- [ ] **Step 7: Commit the local renderer**

Commit the module, tests, package metadata, and vendored distribution:

```bash
git add .gitignore package.json package-lock.json app/static/math-text.mjs \
  app/static/vendor/katex tests/math-text.test.mjs tests/test_static_pages.py
git commit -m "feat: add local math rendering module"
```

### Task 6: Render formulas and diagnostic feedback throughout the classroom

**Files:**
- Modify: `app/static/lesson.html`
- Modify: `app/static/lesson.js`
- Modify: `app/static/styles.css`
- Modify: `app/static/runtime-core.mjs`
- Modify: `tests/runtime-core.test.mjs`
- Modify: `tests/test_static_pages.py`

- [ ] **Step 1: Write failing runtime/UI contract tests**

Add to `tests/runtime-core.test.mjs`:

```javascript
test("new generated interactions never require a formula keyboard", () => {
  assert.equal(classifyInteractionControl({ kind: "choice" }), "options");
  assert.equal(classifyInteractionControl({ kind: "point_select" }), "board");
});
```

In `tests/test_static_pages.py`, require `lesson.js` to import and call
`renderMathText`, and require the option click path to pass the selected option object:

```python
assert "/static/vendor/katex/katex.min.css" in lesson_html
assert 'import { renderMathText } from "./math-text.mjs"' in source
assert "renderMathText(dom.problem" in source
assert "renderMathText(content" in source
assert "renderMathText(heading" in source
assert "renderMathText(button" in source
assert "submitInteraction(interaction, option.option_id, option" in source
```

- [ ] **Step 2: Run UI contract tests and verify RED**

Run:

```bash
node --test tests/runtime-core.test.mjs &&
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_static_pages.py
```

Expected: static assertions fail because `lesson.js` still uses `textContent`.

- [ ] **Step 3: Load KaTeX CSS and integrate the renderer**

Add before `styles.css` in `lesson.html`:

```html
<link rel="stylesheet" href="/static/vendor/katex/katex.min.css">
```

Import `renderMathText` at the top of `lesson.js`. Replace direct `textContent` writes for
the problem, learning goal, board content, comparison values, narration, interaction heading,
option labels, hints, and feedback with `renderMathText(node, value)`. Keep control labels and
screen-reader status messages as plain text when they contain no math.

For board updates, compare the source string through `content.dataset.source` instead of
`content.textContent`, because KaTeX changes the DOM text:

```javascript
if (content.dataset.source !== value.content) {
  content.dataset.source = value.content || "";
  renderMathText(content, value.content || humanizeTarget(target));
}
```

- [ ] **Step 4: Use selected-option feedback and audio**

Change the option click handler to call:

```javascript
submitInteraction(interaction, option.option_id, option, {
  controls,
  feedback,
  hint,
  continueButton,
});
```

Update `submitInteraction` to accept `selectedOption`. For an incorrect response, render
`selectedOption.feedback` when present, otherwise keep the existing staged hint. For a correct
response, render selected-option feedback before falling back to
`interaction.explanation_after_correct`. Play `selectedOption.feedback_audio_url` before the
legacy hint/correct audio fallback.

Text-input and board-point calls pass `null` as `selectedOption`, preserving old lesson behavior.

- [ ] **Step 5: Style formulas without changing the full-screen layout**

Add narrowly scoped rules:

```css
.math-inline .katex {
  font-size: 1.05em;
}

.board-content .katex {
  color: var(--board-ink);
  font-size: 1.22em;
}

.math-display {
  display: block;
  width: 100%;
  margin: 0.3em 0;
  text-align: center;
}

.interaction-option .katex {
  font-size: 1.16em;
}

.math-fallback {
  font-family: "STIX Two Math", "Times New Roman", serif;
}
```

Do not introduce a side panel or alter the 16:10 landscape classroom regions.

- [ ] **Step 6: Run JavaScript and static-page tests and verify GREEN**

Run:

```bash
npm test &&
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_static_pages.py
```

Expected: all Node and static-page tests pass.

- [ ] **Step 7: Commit the classroom integration**

```bash
git add app/static/lesson.html app/static/lesson.js app/static/styles.css \
  app/static/runtime-core.mjs tests/runtime-core.test.mjs \
  tests/test_static_pages.py
git commit -m "feat: render formulas in choice lessons"
```

### Task 7: Document, regress, and verify the real lesson

**Files:**
- Modify: `README.md`
- Modify: `scripts/smoke_live.py`
- Test: all Python and Node tests

- [ ] **Step 1: Extend the live smoke assertions**

After lesson generation in `scripts/smoke_live.py`, assert:

```python
assert lesson["beats"][1]["purpose"] == "先认识方法"
assert lesson["beats"][1]["layer"] == "micro_explanation"
assert lesson["beats"][-1]["interaction"]["kind"] == "choice"
assert all(
    option.get("feedback")
    for beat in lesson["beats"]
    if (beat.get("interaction") or {}).get("kind") == "choice"
    for option in beat["interaction"]["options"]
)
```

Keep secrets loaded only from `.env`; do not print request headers, keys, raw model payloads, or
private reference material.

- [ ] **Step 2: Update README behavior and test instructions**

Document:

- method introduction before the first mathematical operation;
- locally rendered KaTeX formulas;
- diagnostic choices and option-specific spoken feedback;
- choice-based near transfer;
- `npm install` only when refreshing vendored KaTeX assets;
- standard commands `pytest -q tests` and `npm test`.

- [ ] **Step 3: Run focused regression suites**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests/test_schemas.py tests/test_generation.py \
  tests/test_compiler.py tests/test_api.py tests/test_tts_client.py \
  tests/test_static_pages.py &&
npm test
```

Expected: all selected Python and Node tests pass.

- [ ] **Step 4: Run complete verification**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
pytest -q tests &&
python -m compileall -q app tests &&
npm test &&
git diff --check &&
git status -sb
```

Expected: zero Python/Node failures, compileall exits 0, no whitespace errors, and only the
intentional README/smoke changes remain before the final commit.

- [ ] **Step 5: Commit documentation and smoke coverage**

```bash
git add README.md scripts/smoke_live.py
git commit -m "docs: explain method-first choice lessons"
```

- [ ] **Step 6: Run the real provider smoke**

With the local server running under `conda activate general`, execute:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh &&
conda activate general &&
python scripts/smoke_live.py
```

Expected: a live lesson id, method-introduction beat, diagnostic choice interactions, local
audio URLs, and no printed credentials.

- [ ] **Step 7: Verify in the horizontal Pad classroom**

Open the generated lesson at `http://127.0.0.1:8000/lesson/<lesson-id>` and verify:

1. the first teaching emphasis is “配方法”;
2. the definition, target form, and usefulness are visible and spoken before algebraic changes;
3. original problem, board formulas, interaction options, feedback, and near transfer use KaTeX;
4. no newly generated interaction presents a formula input field;
5. two different wrong options produce different feedback and allow retry;
6. the correct option advances;
7. the micro-explanation layer returns to the original base board;
8. the entire experience remains full-screen landscape, not split into columns.

Record the lesson id and exact automated verification counts in the final handoff. State that this
is runtime and browser verification, not evidence of student learning transfer.
