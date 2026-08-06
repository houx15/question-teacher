# Light Classroom and Semantic Emphasis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dark classroom with a warm light Pad experience and prevent useless whole-formula enclosure when the board contains only one teaching object.

**Architecture:** Keep the existing fullscreen DOM and runtime. Re-theme the classroom through scoped CSS tokens and component surfaces, add a deterministic annotation guard to the shared board reducer, and strengthen Director/Reviewer prompts so new lessons choose semantic local targets before reaching that guard.

**Tech Stack:** HTML, CSS, browser-native JavaScript modules, Node test runner, Python/FastAPI tests.

---

### Task 1: Reject low-information single-object enclosure

**Files:**
- Modify: `tests/runtime-core.test.mjs`
- Modify: `app/static/runtime-core.mjs`

- [ ] **Step 1: Write the failing reducer tests**

Add two tests. The first starts with one `kind: "object"` entry and asserts that `circle` and `box` do not create annotations. The second starts with two objects and asserts that a `circle` remains available for distinction.

```javascript
test("single board object ignores whole-object enclosure", () => {
  const initial = new Map([
    ["equation", {
      kind: "object",
      target: "equation",
      content: "x² - 6x = -5",
      annotations: [],
    }],
  ]);

  const circled = applyBoardAction(initial, {
    type: "annotate",
    target: "equation",
    annotation: "circle",
  });
  const boxed = applyBoardAction(circled, {
    type: "annotate",
    target: "equation",
    annotation: "box",
  });

  assert.deepEqual(boxed.get("equation").annotations, []);
});

test("multiple board objects retain enclosure for distinction", () => {
  const initial = new Map([
    ["left", { kind: "object", target: "left", content: "x+1", annotations: [] }],
    ["right", { kind: "object", target: "right", content: "2", annotations: [] }],
  ]);

  const result = applyBoardAction(initial, {
    type: "annotate",
    target: "left",
    annotation: "circle",
  });

  assert.equal(result.get("left").annotations[0].type, "circle");
});
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
node --test tests/runtime-core.test.mjs
```

Expected: the single-object enclosure test fails because current code appends both annotations.

- [ ] **Step 3: Implement the minimal runtime guard**

In `applyBoardAction`, count real board objects before appending an annotation:

```javascript
function visibleObjectCount(board) {
  return [...board.values()].filter((value) => (
    value?.kind === "object" && value.masked !== true
  )).length;
}

case "annotate": {
  const annotation = action.annotation || "highlight";
  const isUselessEnclosure = (
    (annotation === "circle" || annotation === "box")
    && visibleObjectCount(board) <= 1
  );
  if (isUselessEnclosure) break;
  // preserve the existing annotation append
}
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
node --test tests/runtime-core.test.mjs
```

Expected: all runtime tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/static/runtime-core.mjs tests/runtime-core.test.mjs
git commit -m "fix: suppress useless whole-formula enclosure"
```

### Task 2: Encode the teaching rule in generation and review

**Files:**
- Modify: `tests/test_generation.py`
- Modify: `app/prompts.py`

- [ ] **Step 1: Write failing prompt contract assertions**

Extend the prompt contract test to require all three semantic rules:

```python
assert "只有一个" in DIRECTOR_SYSTEM
assert "circle" in DIRECTOR_SYSTEM
assert "局部语义对象" in DIRECTOR_SYSTEM
assert "无信息增益" in REVIEWER_SYSTEM
assert "整式圈注" in REVIEWER_SYSTEM
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_generation.py
```

Expected: the new assertions fail because the current prompts only say that a mentioned object may be circled.

- [ ] **Step 3: Add precise Director, Reviewer, and Revision constraints**

Replace the generic marking instruction with:

```text
重点动作必须指向对理解有帮助的局部语义对象。画面只有一个公式或板书对象时，
禁止用 circle 或 box 包围整个对象；需要强调内部的系数、符号、运算或条件时，
先将该局部写成独立 target，再使用 focus、underline、arrow 或短 label。
circle/box 只用于多个对象间的区分、回指或比较。
```

Add to the Reviewer:

```text
把没有信息增益的整式圈注、为制造动画而添加的标记列为 must_fix。
```

Add a compact equivalent constraint to `REVISION_SYSTEM`.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_generation.py
```

Expected: all generation tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/prompts.py tests/test_generation.py
git commit -m "feat: require semantic teaching emphasis"
```

### Task 3: Re-theme the fullscreen classroom as warm light paper

**Files:**
- Modify: `tests/test_static_pages.py`
- Modify: `app/static/lesson.html`
- Modify: `app/static/styles.css`

- [ ] **Step 1: Write failing light-theme contract assertions**

In the static page test, assert the lesson theme color and scoped light classroom tokens:

```python
assert '<meta name="theme-color" content="#f4efe5">' in lesson_html
assert "--classroom-canvas: #f4efe5;" in styles.text
assert "--board-surface: #fbfaf6;" in styles.text
assert "--board-ink: #203047;" in styles.text
assert "--classroom-panel: #fffdf8;" in styles.text
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_static_pages.py
```

Expected: assertions fail because the lesson page still declares the dark theme and no light classroom tokens.

- [ ] **Step 3: Add scoped classroom color tokens**

Add tokens to `:root` without changing the authoring-page colors:

```css
--classroom-canvas: #f4efe5;
--classroom-panel: #fffdf8;
--board-surface: #fbfaf6;
--board-surface-deep: #f0ede5;
--board-ink: #203047;
--board-ink-soft: #657286;
--classroom-line: #d8d2c6;
--classroom-reason: #4f8b7b;
--classroom-focus: #e9b949;
```

Change the lesson theme meta value to `#f4efe5`.

- [ ] **Step 4: Re-theme every classroom surface**

Update the classroom-only selectors from `.lesson-body` through the state, overlay, rotate and responsive sections:

- use `--classroom-canvas` for the page;
- use `--classroom-panel` for shell, top bar, route strip, controls and cards;
- use `--board-surface` plus a subtle warm paper texture for the board;
- use `--board-ink` for formulae and headings;
- use `--board-ink-soft` for secondary copy;
- keep `--classroom-focus` as the single active emphasis color;
- change masks to pale graphite hatching;
- use light borders and restrained shadows instead of black glow;
- keep all layout dimensions and fullscreen behavior unchanged.

- [ ] **Step 5: Run focused tests**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_static_pages.py
node --test tests/runtime-core.test.mjs
```

Expected: both suites pass.

- [ ] **Step 6: Commit**

```bash
git add app/static/lesson.html app/static/styles.css tests/test_static_pages.py
git commit -m "style: introduce warm light classroom"
```

### Task 4: Full verification and visual acceptance

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run complete automated verification**

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q
python -m compileall -q app scripts tests
node --test tests/runtime-core.test.mjs
git diff --check
git status -sb
```

Expected: all Python and Node tests pass, compilation and diff checks exit zero, and only intentional changes are present.

- [ ] **Step 2: Run the Demo in the browser**

Start the app with the configured environment and inspect a 1280×800 landscape viewport. Verify:

1. warm light page, shell, board, top bar, subtitles and controls;
2. readable ink contrast;
3. no whole-formula circle when it is the only board object;
4. multi-object enclosure still renders when pedagogically valid;
5. interaction overlays, temporary layer return, audio controls and portrait rotation remain usable.

- [ ] **Step 3: Commit any verified visual corrections**

If browser inspection reveals a concrete issue, add a focused regression assertion where possible, make the smallest correction, rerun Task 4 Step 1, and commit:

```bash
git add app/static/styles.css app/static/lesson.html tests
git commit -m "fix: refine light classroom contrast"
```
