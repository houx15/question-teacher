# Adaptive Structured-Board Teaching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `TeachingProgression`, natural display/spoken teaching scripts, diagnostic wrong-answer support, and a 16px continuous structured board so generated lessons explicitly teach why each step is taken and adapt explanation depth without blocking the main lesson.

**Architecture:** Extend the existing private preparation pipeline with a versioned `TeachingProgression` between `ReasoningTrajectory` and the final classroom artifacts. The Interaction Designer produces diagnostic structure before the final Teaching Agent call, so the Teaching Agent remains the sole owner of all student-facing main and remediation language. The Classroom Director compiles that language into whitelisted step, board, emphasis, support, and scroll actions; the existing compiler, Volcengine TTS, SQLite persistence, private generation record, and full-screen runtime remain the delivery boundary.

**Tech Stack:** Python 3.9, FastAPI, Pydantic v2, SQLite, the existing OpenAI-compatible JSON client, pytest/pytest-asyncio, vanilla JavaScript with `node:test`, KaTeX, and Volcengine TTS.

**Design spec:** `docs/superpowers/specs/2026-08-14-adaptive-structured-board-teaching-design.md`

---

## Delivery rules

- Work in the current repository and branch. Do not create a worktree.
- Activate the approved Python environment for every Python command:

  ```bash
  source /opt/anaconda3/etc/profile.d/conda.sh
  conda activate general
  ```

- Keep Python 3.9-compatible typing: use `Optional`, `List`, `Dict`, `Tuple`, and `Union`, never `X | Y`.
- Do not touch or commit the untracked `.superpowers/` directory.
- Keep the service stopped until Task 12's authorized end-to-end verification.
- Use strict TDD: write one behavior, observe the intended RED, implement the minimum production change, rerun focused tests, then commit.
- Do not delete active security, reference-safety, provenance, concurrency, atomic persistence, or cache-busting tests to make the new path pass.
- Never write API keys, tokens, raw reference prose, private correct answers, or review findings into public lesson JSON or logs.
- Run `git diff --check` before every commit.

## Dependency refinement

The approved design makes the Teaching Agent the final language owner. To enforce that technically, the private dependency order becomes:

```text
SolutionTrace
→ ReasoningTrajectory
→ TeachingProgression
→ InteractionPlan (diagnostic intent only)
→ TeachingScript (main language + correct/wrong response language)
→ PerformanceScore
→ SimulationReport + LessonReviewDecision
```

The Interaction Designer does not write final feedback sentences. It names the diagnostic target, error code, diagnosis, desired explanation depth, and resume point. The Teaching Agent receives that structure and writes all student-facing language. This is the implementation-level meaning of “最后需要 Teaching Agent 将讲解组织成更顺溜的人话”.

## Stable boundaries

The following contracts remain stable:

- `POST /api/lessons/generate` still accepts `ProblemInput`.
- Generation success still returns a lesson ID and does not auto-navigate.
- `GET /api/lessons/{lesson_id}` remains the student-facing lesson endpoint.
- Existing lesson IDs remain readable without migration or regeneration.
- The reference answer, reference explanation, frozen route, and typed math bridge remain the authority boundary.
- The change does not introduce a general solver, grader, or free-form formula input.
- Volcengine TTS is invoked only after preparation, review, adaptation, and compilation succeed.
- Lesson runtime, private `GenerationRecord`, audio, and lesson row remain atomically paired.
- Public generation errors remain generic; private diagnostics remain content-free and allowlisted.

The public runtime gains only bounded optional fields required for structured-board playback. Old saved lessons omit them and continue through the current legacy rendering path.

## Target file map

Create:

- `app/math_speech.py` — deterministic display-math to Chinese classroom speech and alignment checks.
- `app/teaching_progression_validation.py` — focused validation for progression structure and coverage.
- `app/static/structured-board.mjs` — pure structured-board state reducer and selectors.
- `tests/test_math_speech.py` — pronunciation and fail-closed alignment tests.
- `tests/test_teaching_progression_validation.py` — progression coverage and ordering tests.
- `tests/structured-board.test.mjs` — step lifecycle, support, and scroll-selection tests.

Modify:

- `app/preparation_models.py`
- `app/preparation_prompts.py`
- `app/preparation_pipeline.py`
- `app/preparation_validation.py`
- `app/pedagogy_rubric.py`
- `app/prepared_lesson_adapter.py`
- `app/schemas.py`
- `app/compiler.py`
- `app/generation.py`
- `app/generation_integrity.py`
- `app/audio_manifest.py`
- `app/audio_service.py`
- `app/api.py`
- `app/store.py`
- `app/static/runtime-core.mjs`
- `app/static/cue-player.mjs` only if support-cue pause/cancel cannot reuse its existing sequence contract
- `app/static/generate.js`
- `app/static/index.html`
- `app/static/lesson.js`
- `app/static/lesson.html`
- `app/static/styles.css`
- `tests/preparation_fakes.py`
- `tests/generation_fakes.py`
- `tests/test_preparation_models.py`
- `tests/test_preparation_prompts.py`
- `tests/test_preparation_pipeline.py`
- `tests/test_preparation_validation.py`
- `tests/test_prepared_lesson_adapter.py`
- `tests/test_schemas.py`
- `tests/test_compiler.py`
- `tests/test_generation.py`
- `tests/test_api.py`
- `tests/test_store.py`
- `tests/test_tts_client.py`
- `tests/test_static_pages.py`
- `tests/runtime-core.test.mjs`
- `tests/cue-player.test.mjs`
- `tests/fixtures/pedagogy_golden_cases.json`
- `tests/test_pedagogy_evaluation.py`
- `scripts/run_pedagogy_evaluation.py`
- `scripts/smoke_live.py`
- `README.md`

## Spec coverage map

| Approved design requirement | Implemented and verified by |
| --- | --- |
| First-class private `TeachingProgression` | Tasks 1–3 |
| Teaching Agent owns final natural language | Tasks 2 and 4 |
| Separate `display_text` and `spoken_text` | Tasks 4, 6, 9, and 11 |
| Correct/wrong answers change depth and then continue | Tasks 4, 5, 9, and 11 |
| Clause-bound problem emphasis and board actions | Tasks 6, 8, and 9 |
| Single-column colored step structure | Tasks 8–10 |
| 16px body, formula hierarchy, automatic scroll | Tasks 9, 10, and 12 |
| Student simulation and non-compensable review | Tasks 3 and 12 |
| Private provenance, atomic persistence, lesson-ID replay | Tasks 7, 11, and 12 |
| Parameter-root five-step acceptance and 18-case breadth | Task 12 |
| Old lesson compatibility and cache-busting | Tasks 1, 5–7, 9–10, and 12 |

---

### Task 1: Add the first-class TeachingProgression contracts

**Files:**

- Modify: `app/preparation_models.py:20-560`
- Modify: `tests/test_preparation_models.py`

- [ ] **Step 1: Write failing model tests**

  Add builders and tests that require ordered steps, stable IDs, valid phases, at least one guiding question, exact local uniqueness, bounded board summaries, and first-class inclusion in `PreparedLesson`:

  ```python
  def teaching_progression_payload():
      return {
          "steps": [
              {
                  "step_id": "teaching-step-001",
                  "sequence_index": 0,
                  "episode_ids": ["episode-001"],
                  "phase": "construct",
                  "student_problem": "方程的根代表什么？",
                  "why_now": "先把题目的关键事实变成可执行条件。",
                  "evidence_target_ids": ["problem-focus-001"],
                  "guiding_questions": ["把一个数代入方程后会发生什么？"],
                  "knowledge_anchor": "方程的根代入后等式成立",
                  "checkpoint": None,
                  "reveal": "令x=2n",
                  "math_action": "把根代入关于x的方程",
                  "directory_question": "根代表什么？",
                  "directory_label": "第一步：理解方程的根",
                  "board_summary": ["方程的根 → 代入后等式成立"],
                  "error_tip": "不要把根代给m或n",
                  "transition_question": "这个方程是关于谁的？",
                  "must_teach_refs": ["must-teach-root"],
              }
          ]
      }


  def test_teaching_progression_requires_contiguous_step_order():
      payload = teaching_progression_payload()
      payload["steps"][0]["sequence_index"] = 1

      with pytest.raises(ValidationError, match="contiguous"):
          TeachingProgression.model_validate(payload)


  def test_prepared_lesson_contains_teaching_progression():
      payload = prepared_lesson().model_dump(mode="python")
      payload["teaching_progression"] = teaching_progression_payload()

      lesson = PreparedLesson.model_validate(payload)

      assert lesson.teaching_progression.steps[0].directory_label == (
          "第一步：理解方程的根"
      )
  ```

- [ ] **Step 2: Run the focused test and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_models.py -k "teaching_progression or prepared_lesson_contains"'
  ```

  Expected: collection or assertions fail because `TeachingProgression` and `PreparedLesson.teaching_progression` do not exist.

- [ ] **Step 3: Implement the model vocabulary**

  Add these exact Python 3.9-compatible models:

  ```python
  TeachingPhase = Literal["construct", "explore", "execute", "check"]


  class ProgressionCheckpoint(SchemaModel):
      diagnostic_goal: NonEmptyString
      misconception_ids: List[GeneratedId] = Field(
          default_factory=list,
          max_length=MAX_DETAIL_ITEMS,
      )


  class TeachingProgressionStep(SchemaModel):
      step_id: GeneratedId
      sequence_index: int = Field(ge=0)
      episode_ids: List[GeneratedId] = Field(
          min_length=1,
          max_length=MAX_PREPARATION_ITEMS,
      )
      phase: TeachingPhase
      student_problem: NonEmptyString
      why_now: NonEmptyString
      evidence_target_ids: List[GeneratedId] = Field(
          default_factory=list,
          max_length=MAX_DETAIL_ITEMS,
      )
      guiding_questions: List[NonEmptyString] = Field(
          min_length=1,
          max_length=MAX_DETAIL_ITEMS,
      )
      knowledge_anchor: NonEmptyString
      checkpoint: Optional[ProgressionCheckpoint] = None
      reveal: NonEmptyString
      math_action: NonEmptyString
      directory_question: NonEmptyString
      directory_label: GeneratedLabelText
      board_summary: List[NarrativeBoardContent] = Field(
          min_length=1,
          max_length=8,
      )
      error_tip: NonEmptyString
      transition_question: NonEmptyString
      must_teach_refs: List[GeneratedId] = Field(
          min_length=1,
          max_length=MAX_DETAIL_ITEMS,
      )


  class TeachingProgression(SchemaModel):
      steps: List[TeachingProgressionStep] = Field(
          min_length=1,
          max_length=MAX_PREPARATION_ITEMS,
      )

      @model_validator(mode="after")
      def validate_steps(self) -> "TeachingProgression":
          _require_unique([item.step_id for item in self.steps], "teaching step ids")
          if [item.sequence_index for item in self.steps] != list(range(len(self.steps))):
              raise ValueError("teaching step indexes must be contiguous starting at zero")
          return self
  ```

  Extend `ArtifactType`, `PreparedLesson`, and artifact-history bounds:

  ```python
  ArtifactType = Literal[
      "solution_trace",
      "reasoning_trajectory",
      "teaching_progression",
      "interaction_plan",
      "teaching_script",
      "performance_score",
      "simulation_report",
  ]

  class PreparedLesson(SchemaModel):
      rubric_version: NonEmptyString
      solution_trace: SolutionTrace
      reasoning_trajectory: ReasoningTrajectory
      teaching_progression: Optional[TeachingProgression] = None
      interaction_plan: InteractionPlan
      teaching_script: TeachingScript
      performance_score: PerformanceScore
      simulation_report: SimulationReport
      review: LessonReviewDecision
      repair_count: int = Field(ge=0)
      artifact_history: List[ArtifactRevision] = Field(
          min_length=5,
          max_length=MAX_ARTIFACT_HISTORY_ITEMS,
      )
  ```

  `teaching_progression` remains optional only for historical rubric-0.1 private records. Current-rubric preparation and storage validators must require it and require the exact seven-artifact history; a new record may never omit it.

- [ ] **Step 4: Run the model suite and confirm GREEN**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_models.py'
  ```

  Expected: all model tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add app/preparation_models.py tests/test_preparation_models.py
  git diff --check
  git commit -m "feat: model structured teaching progression"
  ```

---

### Task 2: Generate progression and diagnostic intent before final language

**Files:**

- Modify: `app/preparation_prompts.py:40-670`
- Modify: `app/preparation_pipeline.py:28-1020`
- Modify: `app/api.py:55-90`
- Modify: `app/static/generate.js:20-40`
- Modify: `app/static/index.html` generate-script version
- Modify: `scripts/smoke_live.py:20-180`
- Modify: `tests/preparation_fakes.py`
- Modify: `tests/test_preparation_prompts.py`
- Modify: `tests/test_preparation_pipeline.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_pages.py`

- [ ] **Step 1: Write failing prompt and call-order tests**

  Require the exact role-call order and reference boundary:

  ```python
  @pytest.mark.asyncio
  async def test_pipeline_builds_progression_and_interaction_before_final_script():
      pipeline, client = pipeline_with_approved_responses()

      run = await pipeline.prepare_with_audit(
          problem(), frozen_route(), focus_targets()
      )

      assert [call.role for call in client.calls] == [
          "reference_analyst",
          "teaching_designer",
          "teaching_designer",
          "interaction_designer",
          "script_teacher",
          "classroom_director",
          "student_simulator",
          "lesson_reviewer",
      ]
      assert run.prepared_lesson.teaching_progression.steps


  def test_final_script_prompt_receives_progression_and_interaction_without_raw_reference():
      marker = "RAW_REFERENCE_ONLY_MARKER"
      prompt = teaching_script_prompt(
          teaching_progression(),
          interaction_plan(),
      )

      assert '"teaching_progression"' in prompt
      assert '"interaction_plan"' in prompt
      assert marker not in prompt
  ```

- [ ] **Step 2: Run focused tests and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_prompts.py tests/test_preparation_pipeline.py -k "progression or interaction_before_final_script"'
  ```

  Expected: failures show missing progression prompt/system/pipeline stage and the old script-before-interaction order.

- [ ] **Step 3: Add the progression prompt and final-language prompt inputs**

  Add one additional call under the existing `teaching_designer` role:

  ```python
  TEACHING_PROGRESSION_SYSTEM = "\n".join(
      (
          "你是教学推进设计师。只输出符合 Schema 的 TeachingProgression。",
          "每一步必须先定义学生此刻的问题和 why_now，再定义动作与结论。",
          "步骤标题只能在思路形成后揭示，不得提前剧透完整解法。",
          "每个 must_teach 必须被一个步骤引用；不要写最终教师台词。",
      )
  )


  def teaching_progression_prompt(
      reasoning_trajectory: ReasoningTrajectory,
      problem_targets: _ProblemTargets,
      repair: Optional[_InputDict] = None,
  ) -> str:
      payload = {
          "reasoning_trajectory": _artifact_payload(
              reasoning_trajectory,
              ReasoningTrajectory,
              "reasoning_trajectory",
          ),
          "problem_targets": _problem_targets_projection(problem_targets),
      }
      return _prompt_envelope(
          "把 ReasoningTrajectory 组织为可审核的 TeachingProgression。",
          _with_repair(payload, repair),
      )


  def teaching_script_prompt(
      teaching_progression: TeachingProgression,
      interaction_plan: InteractionPlan,
      repair: Optional[_InputDict] = None,
  ) -> str:
      return _prompt_envelope(
          "为主线和每个互动结果写自然、顺畅、可朗读的最终 TeachingScript。",
          _with_repair(
              {
                  "teaching_progression": _artifact_payload(
                      teaching_progression,
                      TeachingProgression,
                      "teaching_progression",
                  ),
                  "interaction_plan": _artifact_payload(
                      interaction_plan,
                      InteractionPlan,
                      "interaction_plan",
                  ),
              },
              repair,
          ),
      )
  ```

  Update `_PREPARED_ARTIFACT_TYPES` and capability projection so progression is accepted as a private artifact and no new arbitrary layout capability is exposed.

- [ ] **Step 4: Insert the request-scoped pipeline stage**

  Add `teaching_progression` to `PreparationState`, snapshots, active versions, and `_continue_preparation`:

  ```python
  await self._create_teaching_progression(
      state,
      problem_focus_targets,
      on_stage,
  )
  await self._create_interaction_plan(state, on_stage)
  await self._create_teaching_script(state, on_stage)
  await self._create_performance_score(
      state,
      problem_focus_targets,
      on_stage,
  )
  ```

  `_create_teaching_progression` must emit `设计教学推进`, use `TEACHING_PROGRESSION_SYSTEM`, validate the exact model, accept it as `artifact_type="teaching_progression"`, and record `responsible_role="teaching_designer"`.

  Change `_create_interaction_plan` inputs to `TeachingProgression` only. Change `_create_teaching_script` inputs to `TeachingProgression + InteractionPlan`. The final Teaching Agent is therefore the last writer of student-facing language.

- [ ] **Step 5: Update fakes and assert content isolation**

  Map both `TEACHING_DESIGNER_SYSTEM` and `TEACHING_PROGRESSION_SYSTEM` to the role queue `teaching_designer`; give the fake queue two responses in order. Add assertions that raw reference text appears only in the already-approved grounder/auditor/reference-analyst boundary and never in progression, interaction, or final script calls.

- [ ] **Step 6: Run focused pipeline and prompt suites**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_prompts.py tests/test_preparation_pipeline.py tests/test_reference_safety.py'
  ```

  Expected: all tests pass; call order is deterministic and request-local under reversed concurrent completion.

- [ ] **Step 7: Update public progress and smoke grammar**

  Add the exact internal/public stage and preserve monotonic order:

  ```python
  _PUBLIC_GENERATION_STAGES.update(
      {"设计教学推进": "正在设计课堂推进"}
  )

  _PUBLIC_STAGE_ORDER = (
      "正在理解题目",
      "正在核对题目材料",
      "正在整理参考解析",
      "正在设计解题思维轨迹",
      "正在设计课堂推进",
      "正在设计互动",
      "正在编写讲稿",
      "正在编排板书与高亮",
      "正在审核和优化课程",
      "正在编译课程",
      "正在生成语音",
      "正在保存课程",
      "课程已生成",
  )
  ```

  Add the matching non-technical explanation in `generate.js`: “把思路组织成学生能够一步步跟上的课堂结构。” Update `scripts/smoke_live.py` to recognize the exact initial grammar `TRACE → TRAJECTORY → PROGRESSION → INTERACTION → SCRIPT → PERFORMANCE → SIMULATION → REVIEW`; repair suffixes begin at the reviewed artifact and preserve the same dependency order. Keep structural retries max two per logical call, post-repair reviewer fresh-context allowance max four consecutive reviewer calls, and eight repair cycles.

  Add tests that the real pipeline emits the new order, repair callbacks never regress a public job stage, concurrent jobs do not share progress state, valid smoke traces pass, and swapped `INTERACTION/SCRIPT` traces fail.

  Because `generate.js` changes, bump its version in `index.html`; keep the unversioned home shell `Cache-Control: no-cache` and assert the new versioned response does not receive `no-cache` or `no-store`.

- [ ] **Step 8: Run progress and smoke tests**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_api.py tests/test_static_pages.py -k "progress or smoke"'
  ```

  Expected: all tests pass with the new monotonic stage and exact model-call grammar.

- [ ] **Step 9: Commit**

  ```bash
  git add app/preparation_prompts.py app/preparation_pipeline.py app/api.py app/static/generate.js app/static/index.html scripts/smoke_live.py tests/preparation_fakes.py tests/test_preparation_prompts.py tests/test_preparation_pipeline.py tests/test_api.py tests/test_static_pages.py
  git diff --check
  git commit -m "feat: prepare explicit teaching progression"
  ```

---

### Task 3: Validate progression coverage and repair dependencies

**Files:**

- Create: `app/teaching_progression_validation.py`
- Create: `tests/test_teaching_progression_validation.py`
- Modify: `app/preparation_validation.py:35-1700`
- Modify: `app/preparation_pipeline.py:100-1220`
- Modify: `tests/test_preparation_validation.py`
- Modify: `tests/test_preparation_pipeline.py`

- [ ] **Step 1: Write failing deterministic coverage tests**

  Cover all of these independent failures:

  - episode omitted or duplicated across progression steps;
  - `must_teach` omitted, duplicated, or attached to a step that does not own its episode;
  - unknown problem evidence target;
  - step order contradicts episode order;
  - empty/generic `why_now` or duplicate directory labels;
  - a board summary that repeats the sole visible object without adding meaning;
  - interaction checkpoint points to an unknown misconception ID;
  - repair of `teaching_progression` invalidates interaction, script, performance, and simulation, but preserves trace and trajectory.

  Canonical tests:

  ```python
  def test_progression_covers_each_must_teach_exactly_once():
      progression = teaching_progression()

      validate_teaching_progression(
          progression,
          reasoning_trajectory(),
          focus_targets(),
      )


  def test_progression_rejects_missing_why_for_a_math_step():
      progression = teaching_progression(update={"steps.0.why_now": "然后计算"})

      with pytest.raises(
          TeachingProgressionValidationError,
          match="progression_why_not_explanatory",
      ):
          validate_teaching_progression(
              progression,
              reasoning_trajectory(),
              focus_targets(),
          )
  ```

- [ ] **Step 2: Run the new test file and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_teaching_progression_validation.py'
  ```

  Expected: collection fails because `app.teaching_progression_validation` does not exist.

- [ ] **Step 3: Implement the focused validator**

  Use exact set and order comparisons; do not use fuzzy LLM scoring:

  ```python
  class TeachingProgressionValidationError(ValueError):
      def __init__(self, code: str, artifact_id: str) -> None:
          super().__init__("%s:%s" % (code, artifact_id))
          self.code = code
          self.artifact_id = artifact_id


  def validate_teaching_progression(
      progression: TeachingProgression,
      trajectory: ReasoningTrajectory,
      problem_targets: List[ProblemFocusTarget],
  ) -> None:
      episode_ids = [item.episode_id for item in trajectory.episodes]
      covered_episodes = [
          episode_id
          for step in progression.steps
          for episode_id in step.episode_ids
      ]
      if covered_episodes != episode_ids:
          raise TeachingProgressionValidationError(
              "progression_episode_coverage_invalid",
              "teaching_progression",
          )

      expected_must_teach = {
          item.must_teach_id: episode.episode_id
          for episode in trajectory.episodes
          for item in episode.must_teach
      }
      actual_refs = [
          item
          for step in progression.steps
          for item in step.must_teach_refs
      ]
      if len(actual_refs) != len(set(actual_refs)) or set(actual_refs) != set(expected_must_teach):
          raise TeachingProgressionValidationError(
              "progression_must_teach_coverage_invalid",
              "teaching_progression",
          )
  ```

  Add explicit checks for evidence target allowlists, ownership, label uniqueness, non-generic reasons, and bounded display content using the existing `math_content` helpers.

- [ ] **Step 4: Make artifact repair order authoritative**

  Replace role-only repair routing with artifact routing because one Teaching Designer owns both trajectory and progression:

  ```python
  ARTIFACT_DEPENDENCY_ORDER = (
      "solution_trace",
      "reasoning_trajectory",
      "teaching_progression",
      "interaction_plan",
      "teaching_script",
      "performance_score",
      "simulation_report",
  )


  def earliest_repair_artifact(findings: List[ReviewFinding]) -> ArtifactType:
      material = [item for item in findings if item.severity != "polish"]
      if not material:
          raise RuntimeError("no material review finding to route")
      return min(
          material,
          key=lambda item: ARTIFACT_DEPENDENCY_ORDER.index(item.artifact_type),
      ).artifact_type
  ```

  `_repair_from` must rebuild the selected artifact and every dependency to its right. The exact initial history is seven artifacts in dependency order. Each repair cycle appends one exact contiguous suffix ending in `simulation_report`; version numbers increase only for rebuilt artifacts.

- [ ] **Step 5: Run validation, pipeline, and history tests**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_teaching_progression_validation.py tests/test_preparation_validation.py tests/test_preparation_pipeline.py'
  ```

  Expected: all tests pass, including eight-repair bound, fresh-review convergence, reversed concurrency, exact active/issued versions, and all provider/deterministic failure points.

- [ ] **Step 6: Commit**

  ```bash
  git add app/teaching_progression_validation.py app/preparation_validation.py app/preparation_pipeline.py tests/test_teaching_progression_validation.py tests/test_preparation_validation.py tests/test_preparation_pipeline.py
  git diff --check
  git commit -m "feat: validate teaching progression dependencies"
  ```

---

### Task 4: Give the Teaching Agent final display, speech, and branch language

**Files:**

- Create: `app/math_speech.py`
- Create: `tests/test_math_speech.py`
- Modify: `app/preparation_models.py:220-332`
- Modify: `app/preparation_prompts.py:100-600`
- Modify: `app/preparation_validation.py:330-820`
- Modify: `tests/test_preparation_models.py`
- Modify: `tests/test_preparation_prompts.py`
- Modify: `tests/test_preparation_validation.py`

- [ ] **Step 1: Write failing math-speech tests**

  Require deterministic classroom readings and fail-closed unsupported syntax:

  ```python
  @pytest.mark.parametrize(
      ("display", "spoken"),
      (
          (r"$m-n$", "m 减 n"),
          (r"$n\ne0$", "n 不等于零"),
          (r"$(2n)^2$", "二 n 整体的平方"),
          (r"$\frac{1}{2}$", "二分之一"),
          (r"$-4(m-n)+2=0$", "负四乘括号 m 减 n 括号加二等于零"),
      ),
  )
  def test_math_display_has_deterministic_chinese_speech(display, spoken):
      assert display_math_to_spoken(display) == spoken


  def test_unknown_control_command_fails_closed():
      with pytest.raises(MathSpeechError, match="unsupported_math_speech"):
          display_math_to_spoken(r"$\htmlClass{secret}{x}$")
  ```

- [ ] **Step 2: Run the new test and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_math_speech.py'
  ```

  Expected: collection fails because `app.math_speech` does not exist.

- [ ] **Step 3: Implement bounded recursive pronunciation**

  Implement a 500-character bounded tokenizer that reuses the strict-math trust boundary and supports the tested vocabulary:

  ```python
  class MathSpeechError(ValueError):
      pass


  _OPERATORS = {
      "=": "等于",
      "+": "加",
      "-": "减",
      r"\ne": "不等于",
      r"\cdot": "乘",
      r"\times": "乘",
      ">": "大于",
      "<": "小于",
      r"\ge": "大于等于",
      r"\le": "小于等于",
  }


  def display_math_to_spoken(value: str) -> str:
      expression = _extract_single_math_expression(value)
      if len(expression) > 500 or not is_strict_math_expression(expression):
          raise MathSpeechError("unsupported_math_speech")
      spoken = _speak_expression(expression)
      if not spoken or contains_math_markup(spoken):
          raise MathSpeechError("unsupported_math_speech")
      return _normalize_spaces(spoken)


  def validate_display_spoken_alignment(
      display_text: str,
      spoken_text: str,
  ) -> None:
      expected = [
          display_math_to_spoken(fragment)
          for fragment in extract_math_fragments(display_text)
      ]
      normalized_spoken = _normalize_spaces(spoken_text)
      if any(item not in normalized_spoken for item in expected):
          raise MathSpeechError("display_spoken_math_mismatch")
  ```

  `_speak_expression` must recursively handle `\frac{numerator}{denominator}`, square/cube exponents, parentheses, integers, single-letter variables, approved functions, and the exact operators above. `extract_math_fragments` must recognize the repository's supported `$...$`, `\(...\)`, and `\[...\]` delimiters without accepting malformed nesting. It must reject DOM/URL commands, malformed braces, and unknown multi-letter controls.

- [ ] **Step 4: Extend final script models**

  Make every main and support clause step-addressable and split display from speech:

  ```python
  ExplanationDepth = Literal["brief", "conceptual", "worked"]


  class ScriptClause(SchemaModel):
      clause_id: GeneratedId
      episode_id: GeneratedId
      lesson_step_id: Optional[GeneratedId] = None
      pedagogical_function: PedagogicalFunction
      display_text: Optional[NarrativeBoardContent] = None
      spoken_text: CueSpokenText
      math_references: List[GeneratedMathAnswer] = Field(
          default_factory=list,
          max_length=MAX_MATH_REFERENCE_ITEMS,
      )
      learner_gain: NonEmptyString
      answer_exposure: bool
      must_teach_refs: List[GeneratedId] = Field(
          default_factory=list,
          max_length=MAX_DETAIL_ITEMS,
      )


  class ResponseScript(SchemaModel):
      response_id: GeneratedId
      interaction_id: GeneratedId
      option_id: GeneratedId
      classification: Literal["correct", "incorrect"]
      error_code: Optional[GeneratedId] = None
      depth: ExplanationDepth
      clauses: List[ScriptClause] = Field(min_length=1, max_length=8)


  class TeachingScript(SchemaModel):
      title: NonEmptyString
      learning_goal: NonEmptyString
      method_rationale: NonEmptyString
      method_introduction: MethodIntroduction
      opening_clause_ids: List[GeneratedId] = Field(min_length=1)
      method_introduction_clause_ids: List[GeneratedId] = Field(min_length=1)
      clauses: List[ScriptClause] = Field(min_length=1)
      response_scripts: List[ResponseScript] = Field(default_factory=list)
      closing_summary_clause_ids: List[GeneratedId] = Field(min_length=1)
  ```

  The two optional clause fields exist only to deserialize historical rubric-0.1 private records. Current-rubric script validation requires both on every main and response clause.

- [ ] **Step 5: Validate final language ownership and depth**

  Add deterministic rules:

  - every main clause references an existing progression step;
  - main clauses cover progression steps in order;
  - every `must_teach` is referenced by at least one main clause in its owning step;
  - every interaction option has exactly one response script;
  - correct response uses `depth="brief"`;
  - incorrect response depth matches the interaction intent and contains more explanatory content than the correct response;
  - display content passes generated-display validation;
  - spoken content contains no math markup or control syntax;
  - every display math fragment is represented by deterministic speech or an exact server-approved equivalent.

  Update `SCRIPT_TEACHER_SYSTEM` to require natural simplified Chinese, varied transitions, short questions, no internal field names, no generic “首先/其次/然后” chains, and no deletion of progression `must_teach` items.

- [ ] **Step 6: Run speech, prompt, model, and validation tests**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_math_speech.py tests/test_preparation_models.py tests/test_preparation_prompts.py tests/test_preparation_validation.py'
  ```

  Expected: all tests pass, including `m-n` pronounced with “减” and malformed math rejected before TTS.

- [ ] **Step 7: Commit**

  ```bash
  git add app/math_speech.py app/preparation_models.py app/preparation_prompts.py app/preparation_validation.py tests/test_math_speech.py tests/test_preparation_models.py tests/test_preparation_prompts.py tests/test_preparation_validation.py
  git diff --check
  git commit -m "feat: author natural adaptive teaching language"
  ```

---

### Task 5: Make interactions diagnostic and non-blocking after support

**Files:**

- Modify: `app/preparation_models.py:280-335`
- Modify: `app/preparation_prompts.py:110-610`
- Modify: `app/preparation_validation.py:390-980`
- Modify: `app/schemas.py:627-705`
- Modify: `app/prepared_lesson_adapter.py:285-360`
- Modify: `app/static/runtime-core.mjs:480-690`
- Modify: `app/static/lesson.js:850-1150`
- Modify: `tests/test_preparation_models.py`
- Modify: `tests/test_preparation_validation.py`
- Modify: `tests/test_prepared_lesson_adapter.py`
- Modify: `tests/test_schemas.py`
- Modify: `tests/runtime-core.test.mjs`

- [ ] **Step 1: Write failing private interaction tests**

  Require structural diagnosis without final prose:

  ```python
  def test_wrong_option_declares_error_and_deeper_support():
      option = PlannedInteractionOption(
          option_id="option-square-error",
          display_text=r"$2n^2$",
          canonical_answer="2n^2",
          misconception="只给n平方",
          error_code="square-distribution-error",
          remediation_depth="worked",
      )

      assert option.error_code == "square-distribution-error"
      assert option.remediation_depth == "worked"


  def test_new_interactions_must_resume_by_continuing():
      payload = planned_interaction().model_dump(mode="python")
      payload["resume_policy"] = "retry"

      with pytest.raises(
          PreparationValidationError,
          match="interaction_resume_policy_invalid",
      ):
          validate_interaction_plan(
              InteractionPlan.model_validate({"interactions": [payload], "transfer_item": transfer_item()}),
              teaching_progression(),
          )
  ```

- [ ] **Step 2: Run focused tests and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_models.py tests/test_preparation_validation.py -k "error_code or resume_policy or remediation_depth"'
  ```

  Expected: fields and validators are missing.

- [ ] **Step 3: Extend diagnostic intent models**

  Keep existing fields required for historical private-record parsing, but make the new path authoritative through deterministic validation:

  ```python
  class PlannedInteractionOption(SchemaModel):
      option_id: GeneratedId
      display_text: NonEmptyString
      canonical_answer: NonEmptyString
      misconception: Optional[NonEmptyString] = None
      error_code: Optional[GeneratedId] = None
      remediation_depth: Optional[Literal["conceptual", "worked"]] = None


  class PlannedInteraction(SchemaModel):
      interaction_id: GeneratedId
      episode_id: Optional[GeneratedId] = None
      teaching_step_id: Optional[GeneratedId] = None
      after_clause_id: Optional[GeneratedId] = None
      diagnostic_target: NonEmptyString
      diagnostic_kind: DiagnosticKind
      why_pause: Optional[NonEmptyString] = None
      prompt: NonEmptyString
      options: List[PlannedInteractionOption] = Field(min_length=3, max_length=4)
      correct_option_id: GeneratedId
      correct_feedback: Optional[NonEmptyString] = None
      incorrect_feedback_by_option: Dict[GeneratedId, NonEmptyString] = Field(
          default_factory=dict
      )
      hint: Optional[NonEmptyString] = None
      resume_clause_id: Optional[GeneratedId] = None
      resume_step_id: Optional[GeneratedId] = None
      resume_policy: Literal["continue", "retry"] = "retry"
      concealed_targets: List[GeneratedId] = Field(default_factory=list)
  ```

  The legacy feedback/hint/clause fields remain optional compatibility inputs and are ignored by the new adapter path. New-generation validation requires `teaching_step_id`, `why_pause`, `resume_step_id`, `resume_policy="continue"`, a correct option with no error code/depth, every wrong option with a unique error code and explicit depth, and checkpoint ownership by the same progression step.

- [ ] **Step 4: Add bounded support cues to public interaction options**

  Extend the optional public contract without exposing the correct answer. Define the bounded support cue before `InteractionOption`, avoiding a forward reference to the later `RuntimeSyncCue` class:

  ```python
  class SupportSyncCue(SchemaModel):
      cue_id: NonEmptyString
      display_text: Optional[NarrativeBoardContent] = None
      spoken_text: NonEmptyString
      lead_actions: List[SyncVisualAction] = Field(default_factory=list, max_length=6)
      start_actions: List[SyncVisualAction] = Field(default_factory=list, max_length=8)
      end_actions: List[SyncVisualAction] = Field(default_factory=list, max_length=6)
      audio_url: Optional[NonEmptyString] = None


  class InteractionOption(SchemaModel):
      option_id: NonEmptyString
      label: NonEmptyString
      feedback: Optional[NonEmptyString] = None
      feedback_audio_url: Optional[NonEmptyString] = None
      support_cues: List[SupportSyncCue] = Field(default_factory=list, max_length=8)


  class Interaction(SchemaModel):
      # existing fields stay unchanged
      advance_after_response: bool = False
  ```

  `prepared_lesson_adapter` maps the final Teaching Agent's `ResponseScript` for each option into `support_cues`; it sets `advance_after_response=True` only for the new structured path. Legacy interactions retain `False` and their retry behavior.

- [ ] **Step 5: Change runtime outcome semantics**

  Add `advanceAfterResponse` to `recordAnswer`:

  ```javascript
  recordAnswer(result = {}) {
    const interaction = this.current()?.interaction;
    const classification = result.classification || "wrong";
    const canContinue = (
      classification === "correct"
      || interaction?.kind === "transfer"
      || interaction?.advance_after_response === true
    );
    this.answers.set(this.current().beat_id, { classification, canContinue });
    return { classification, canContinue };
  }
  ```

  In `submitInteraction`, play the selected option's support cue sequence, apply its visual actions, then close the interaction and advance. Do not re-enable options or recursively call `submitInteraction` when `advance_after_response=true`. Keep the exact legacy retry branch for old lessons.

  Update `resolveInteractionPresentation` so a wrong classification with `advance_after_response=true` returns `advanceMode="automatic"` after support playback; a legacy wrong classification continues to return `advanceMode="retry"`.

- [ ] **Step 6: Run private, adapter, schema, and runtime tests**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_models.py tests/test_preparation_validation.py tests/test_prepared_lesson_adapter.py tests/test_schemas.py'
  node --test --test-name-pattern="adaptive|interaction" tests/runtime-core.test.mjs tests/structured-board.test.mjs
  ```

  Expected: a wrong answer plays deeper support and continues once; correct answer plays brief support and continues; legacy interactions still require retry.

- [ ] **Step 7: Commit**

  ```bash
  git add app/preparation_models.py app/preparation_prompts.py app/preparation_validation.py app/schemas.py app/prepared_lesson_adapter.py app/static/runtime-core.mjs app/static/lesson.js tests/test_preparation_models.py tests/test_preparation_validation.py tests/test_prepared_lesson_adapter.py tests/test_schemas.py tests/runtime-core.test.mjs
  git diff --check
  git commit -m "feat: adapt explanation depth after interactions"
  ```

---

### Task 6: Compile step-aware board performances

**Files:**

- Modify: `app/preparation_models.py:334-400`
- Modify: `app/preparation_prompts.py:120-630`
- Modify: `app/preparation_validation.py:980-1340`
- Modify: `app/schemas.py:524-642`
- Modify: `app/prepared_lesson_adapter.py:1-680`
- Modify: `app/compiler.py:1-382`
- Modify: `tests/test_preparation_models.py`
- Modify: `tests/test_preparation_validation.py`
- Modify: `tests/test_prepared_lesson_adapter.py`
- Modify: `tests/test_compiler.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing step-action tests**

  Require one reveal and one completion boundary per teaching step, exact step ownership for board lines, and paired support open/close actions:

  ```python
  def test_performance_reveals_and_completes_each_step_once():
      score = structured_performance_score()

      validate_performance_score(
          score,
          focus_targets(),
          teaching_progression(),
          teaching_script(),
          interaction_plan(),
      )


  @pytest.mark.parametrize(
      "mutation",
      ("missing_reveal", "duplicate_complete", "wrong_step_line", "unclosed_support"),
  )
  def test_step_performance_rejects_invalid_lifecycle(mutation):
      score = mutate_structured_score(structured_performance_score(), mutation)

      with pytest.raises(PreparationValidationError):
          validate_performance_score(
              score,
              focus_targets(),
              teaching_progression(),
              teaching_script(),
              interaction_plan(),
          )
  ```

- [ ] **Step 2: Run focused performance tests and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_validation.py tests/test_schemas.py -k "step_performance or structured_action"'
  ```

  Expected: missing step metadata/action types cause failures.

- [ ] **Step 3: Extend the private board vocabulary**

  Define the shared literal in `app/schemas.py` next to `LessonLayer`, then import it into `app/preparation_models.py`:

  ```python
  BoardLineRole = Literal[
      "knowledge_anchor",
      "working",
      "summary",
      "error_tip",
      "support",
  ]


  class PerformanceBoardObject(SchemaModel):
      board_object_id: GeneratedId
      content: NarrativeBoardContent
      layer: Literal["base", "micro_explanation", "comparison"] = "base"
      teaching_step_id: Optional[GeneratedId] = None
      line_role: Optional[BoardLineRole] = None
  ```

  The new path requires both optional fields. They remain optional only so historical private records can still deserialize.

- [ ] **Step 4: Extend whitelisted runtime actions**

  Add bounded optional metadata and five new action types:

  ```python
  class SyncVisualAction(SchemaModel):
      surface: Literal["problem", "board"]
      type: Literal[
          "write", "transform", "focus", "emphasize", "annotate",
          "fade", "reveal", "clear_focus",
          "reveal_step_header", "complete_step", "scroll_to_step",
          "open_supporting_explanation", "close_supporting_explanation",
      ]
      target: GeneratedId
      content: Optional[NarrativeBoardContent] = None
      teaching_step_id: Optional[GeneratedId] = None
      step_label: Optional[GeneratedLabelText] = None
      board_role: Optional[BoardLineRole] = None
      # retain existing source/relation/annotation/emphasis/persistence fields
  ```

  Validation rules:

  - step actions are board-only;
  - `reveal_step_header` requires matching `teaching_step_id` and `step_label`;
  - `complete_step` and `scroll_to_step` require matching `teaching_step_id` and no unrelated payload;
  - support open/close require `teaching_step_id` and a support target;
  - writes on the new path require `teaching_step_id` and `board_role`;
  - problem actions reject all step metadata.

  Extend both authored and runtime main cues so display, speech, and step identity survive compilation:

  ```python
  class NarrativeSyncCue(SchemaModel):
      cue_id: GeneratedId
      teaching_step_id: Optional[GeneratedId] = None
      display_text: Optional[NarrativeBoardContent] = None
      spoken_text: CueSpokenText
      lead_actions: List[SyncVisualAction] = Field(default_factory=list, max_length=6)
      start_actions: List[SyncVisualAction] = Field(default_factory=list, max_length=8)
      end_actions: List[SyncVisualAction] = Field(default_factory=list, max_length=6)


  class RuntimeSyncCue(SchemaModel):
      cue_id: NonEmptyString
      teaching_step_id: Optional[GeneratedId] = None
      display_text: Optional[NarrativeBoardContent] = None
      spoken_text: NonEmptyString
      lead_actions: List[SyncVisualAction] = Field(default_factory=list)
      start_actions: List[SyncVisualAction] = Field(default_factory=list)
      end_actions: List[SyncVisualAction] = Field(default_factory=list)
      audio_url: Optional[NonEmptyString] = None
  ```

  Current-rubric adapter validation requires `teaching_step_id` and `display_text`; old lesson drafts and runtime lessons may omit them.

- [ ] **Step 5: Bind actions to exact clauses and response clauses**

  `PerformanceScore` must cover main `ScriptClause` IDs and nested response-clause IDs. For each step:

  1. problem emphasis may lead the first thinking clause;
  2. `reveal_step_header` occurs after the step's guiding question and before its first conclusion/write;
  3. writes append knowledge anchors and working lines in clause order;
  4. `complete_step` follows the last main clause in the step;
  5. every incorrect response begins with `open_supporting_explanation`, contains at least one `support` write, and ends with `close_supporting_explanation`;
  6. `scroll_to_step` is emitted at step activation and after support close.

  Keep the existing prohibition on non-discriminating emphasis, sole-object emphasis, early visual actions, unsafe layers, and cross-layer target leakage.

- [ ] **Step 6: Compile without moving actions across clauses**

  Extend `prepared_lesson_to_draft_with_provenance` so each action-bearing clause remains isolated in a runtime cue. Preserve authored layer, step metadata, response support cues, and exact action timing. `LessonCompiler` must copy step actions verbatim after Pydantic validation and must not synthesize additional student-visible board text.

- [ ] **Step 7: Run focused backend suites**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_models.py tests/test_preparation_validation.py tests/test_prepared_lesson_adapter.py tests/test_compiler.py tests/test_schemas.py'
  ```

  Expected: all tests pass, including legacy runtime fixture byte equivalence for lessons without structured-step metadata.

- [ ] **Step 8: Commit**

  ```bash
  git add app/preparation_models.py app/preparation_prompts.py app/preparation_validation.py app/schemas.py app/prepared_lesson_adapter.py app/compiler.py tests/test_preparation_models.py tests/test_preparation_validation.py tests/test_prepared_lesson_adapter.py tests/test_compiler.py tests/test_schemas.py
  git diff --check
  git commit -m "feat: compile step-aware board performances"
  ```

---

### Task 7: Preserve structured semantics through generation and persistence

**Files:**

- Modify: `app/preparation_models.py:487-560`
- Modify: `app/generation.py:240-500`
- Modify: `app/generation_integrity.py`
- Modify: `app/store.py`
- Modify: `app/api.py`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_store.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing provenance and mutation tests**

  Require exact clause identity beyond spoken text:

  ```python
  def test_generation_record_preserves_step_display_and_actions():
      bundle = generate_structured_bundle()
      record = bundle.record

      assert record.cue_provenance[0].lesson_step_id == "teaching-step-001"
      assert record.cue_provenance[0].display_text == "方程的根代入后等式仍成立。"
      assert record.cue_provenance[0].spoken_text == "方程的根代入以后，等式仍然成立。"


  @pytest.mark.parametrize(
      "mutation",
      ("step_id", "display_text", "support_action", "step_action", "response_binding"),
  )
  def test_bundle_rejects_structured_runtime_mutation(mutation):
      with pytest.raises(GenerationIntegrityError):
          validate_generated_pair(*mutated_structured_pair(mutation))
  ```

- [ ] **Step 2: Run generation/store/API tests and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_generation.py tests/test_store.py tests/test_api.py -k "structured or step_display or response_binding"'
  ```

  Expected: missing provenance fields and incomplete semantic comparisons fail.

- [ ] **Step 3: Extend private cue provenance**

  ```python
  class RuntimeCueProvenanceRecord(SchemaModel):
      episode_id: GeneratedId
      lesson_step_id: Optional[GeneratedId] = None
      clause_id: GeneratedId
      original_performance_cue_id: GeneratedId
      runtime_cue_id: GeneratedId
      display_text: Optional[NarrativeBoardContent] = None
      spoken_text: CueSpokenText
      response_id: Optional[GeneratedId] = None
  ```

  Build it only through the authoritative adapter factory. Direct construction and caller-supplied episode/step/text/action mappings remain rejected. Optional step/display fields preserve historical rubric-0.1 records; current-rubric bundle validation requires both.

- [ ] **Step 4: Validate the full generated pair**

  Freeze and defensively copy `ProblemInput`, validation report, adapted draft, progression, and runtime baseline before compiler invocation. Compare the returned runtime lesson to the trusted deterministic compile baseline, excluding only `lesson_id` and all audio URL fields. The comparison includes:

  - cue IDs, step IDs, display/spoken text;
  - lead/start/end actions;
  - interaction option/support-cue binding;
  - step lifecycle actions and layers;
  - problem focus targets;
  - validation report.

- [ ] **Step 5: Preserve historical reads and atomic writes**

  New private records require the current rubric and complete seven-artifact history. Historical records with older rubric versions keep their internally consistent older shape and old runtime lesson. New structured records must fail closed if progression, provenance, support cues, or runtime semantics disagree. No SQLite column migration is required because the private record remains JSON in the existing five-column table.

- [ ] **Step 6: Run generation, persistence, and API suites**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_generation.py tests/test_store.py tests/test_api.py'
  ```

  Expected: all tests pass, including empty SQLite initialization, duplicate lesson/generation IDs, corruption fail-closed, no audio on rejected generation, cancellation propagation, and restart-safe private record retrieval.

- [ ] **Step 7: Commit**

  ```bash
  git add app/preparation_models.py app/generation.py app/generation_integrity.py app/store.py app/api.py tests/test_generation.py tests/test_store.py tests/test_api.py
  git diff --check
  git commit -m "feat: persist structured teaching provenance"
  ```

---

### Task 8: Add a pure continuous-board state reducer

**Files:**

- Create: `app/static/structured-board.mjs`
- Create: `tests/structured-board.test.mjs`
- Modify: `app/static/runtime-core.mjs`
- Modify: `tests/runtime-core.test.mjs`
- Modify: `package.json`

- [ ] **Step 1: Write failing board lifecycle tests**

  ```javascript
  import test from "node:test";
  import assert from "node:assert/strict";
  import {
    applyStructuredBoardAction,
    emptyStructuredBoard,
    stepForScroll,
  } from "../app/static/structured-board.mjs";

  test("step grows from questioning to active to completed", () => {
    let state = emptyStructuredBoard();
    state = applyStructuredBoardAction(state, {
      type: "reveal_step_header",
      teaching_step_id: "teaching-step-001",
      step_label: "第一步：理解方程的根",
      target: "teaching-step-001",
      surface: "board",
    });
    state = applyStructuredBoardAction(state, {
      type: "write",
      teaching_step_id: "teaching-step-001",
      board_role: "knowledge_anchor",
      target: "board-root-meaning",
      content: "方程的根 → 代入后等式成立",
      surface: "board",
    });
    state = applyStructuredBoardAction(state, {
      type: "complete_step",
      teaching_step_id: "teaching-step-001",
      target: "teaching-step-001",
      surface: "board",
    });

    assert.equal(state.steps.get("teaching-step-001").status, "completed");
    assert.equal(state.steps.get("teaching-step-001").lines.length, 1);
  });

  test("support closes without deleting the main step", () => {
    const state = structuredBoardWithCompletedMainLine();
    const opened = applyStructuredBoardAction(state, openSupportAction());
    const closed = applyStructuredBoardAction(opened, closeSupportAction());

    assert.equal(closed.steps.get("teaching-step-001").support, null);
    assert.equal(closed.steps.get("teaching-step-001").lines.length, 1);
  });
  ```

- [ ] **Step 2: Run the new Node test and confirm RED**

  ```bash
  node --test tests/structured-board.test.mjs
  ```

  Expected: module-not-found failure.

- [ ] **Step 3: Implement immutable structured-board state**

  ```javascript
  export function emptyStructuredBoard() {
    return { steps: new Map(), activeStepId: null, requestedScrollStepId: null };
  }

  export function applyStructuredBoardAction(current, action) {
    const state = cloneStructuredBoard(current);
    if (!action?.teaching_step_id) return state;
    const stepId = action.teaching_step_id;
    const step = state.steps.get(stepId) || {
      stepId,
      label: "",
      status: "questioning",
      lines: [],
      support: null,
    };

    if (action.type === "reveal_step_header") {
      step.label = action.step_label;
      step.status = "active";
      state.activeStepId = stepId;
    } else if (action.type === "write") {
      const line = {
        target: action.target,
        content: action.content,
        role: action.board_role,
      };
      if (step.support && action.board_role === "support") step.support.lines.push(line);
      else step.lines.push(line);
    } else if (action.type === "complete_step") {
      step.status = "completed";
    } else if (action.type === "open_supporting_explanation") {
      step.status = "supporting";
      step.support = { target: action.target, lines: [] };
    } else if (action.type === "close_supporting_explanation") {
      step.support = null;
      step.status = "active";
    } else if (action.type === "scroll_to_step") {
      state.requestedScrollStepId = stepId;
    }
    state.steps.set(stepId, step);
    return state;
  }
  ```

  Clone maps, step objects, line arrays, and support arrays before mutation. Unknown, malformed, or unsupported actions return an unchanged clone. Never use `innerHTML` or accept model CSS classes.

- [ ] **Step 4: Integrate with existing visual state**

  `runtime-core.mjs` keeps problem emphasis and legacy board maps unchanged. Add `structuredBoard` to `emptyVisualState`; route actions with `teaching_step_id` through `applyStructuredBoardAction`; route old actions through the existing board reducer. Old lessons therefore remain byte-for-byte on the legacy path.

- [ ] **Step 5: Run all JavaScript reducer tests**

  ```bash
  npm test
  ```

  Expected: the new structured-board tests and all existing runtime/cue/math tests pass.

- [ ] **Step 6: Commit**

  ```bash
  git add app/static/structured-board.mjs app/static/runtime-core.mjs tests/structured-board.test.mjs tests/runtime-core.test.mjs package.json
  git diff --check
  git commit -m "feat: reduce continuous board step state"
  ```

---

### Task 9: Render synchronized display text, support cues, and auto-scroll

**Files:**

- Modify: `app/schemas.py:1200-1218`
- Modify: `app/static/cue-player.mjs`
- Modify: `app/static/lesson.js:1-1200`
- Modify: `tests/cue-player.test.mjs`
- Modify: `tests/runtime-core.test.mjs`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing display/speech and scroll tests**

  ```javascript
  test("cue displays display_text while audio uses spoken_text", async () => {
    const harness = cueHarness();
    await harness.player.playBeat({
      sync_cues: [{
        cue_id: "cue-target-sign",
        display_text: "我们要求的是 $m-n$。",
        spoken_text: "我们要求的是 m 减 n。",
        lead_actions: [],
        start_actions: [],
        end_actions: [],
        audio_url: "/audio/cue-target-sign.mp3",
      }],
    });

    assert.equal(harness.displayed[0], "我们要求的是 $m-n$。");
    assert.equal(harness.audioSources[0], "/audio/cue-target-sign.mp3");
  });

  test("support sequence restores the active main step and requests scroll", async () => {
    const result = await runSupportSequence(wrongOptionWithSupportCues());

    assert.equal(result.stepStatus, "active");
    assert.equal(result.requestedScrollStepId, "teaching-step-003");
    assert.equal(result.advanceCount, 1);
  });
  ```

- [ ] **Step 2: Run focused Node tests and confirm RED**

  ```bash
  node --test tests/cue-player.test.mjs tests/runtime-core.test.mjs tests/structured-board.test.mjs
  ```

  Expected: current cue payload has no `display_text`, support cues do not run as a sequence, and no scroll request is consumed.

- [ ] **Step 3: Extend runtime cue display contract**

  The `RuntimeSyncCue` compatibility fields were added in Task 6. Assert here that legacy cues with `display_text=None` display `spoken_text`, while current-rubric cues always provide `display_text` and `teaching_step_id`.

- [ ] **Step 4: Reuse CuePlayer for response support**

  Add a bounded `playCueSequence(cues, token)` method that uses the existing pause/resume/cancel and stale-token guards. It must apply lead/start/end actions once, stop immediately when the beat token changes, and settle if audio errors or times out. Do not create a second independent timer implementation.

- [ ] **Step 5: Render structured sections and consume scroll requests**

  In `lesson.js`:

  - show `cue.display_text || cue.spoken_text` in the narration area;
  - create one `<section class="lesson-step">` per structured step;
  - create its header only when `reveal_step_header` arrives;
  - append/update lines by semantic target without reordering old lines;
  - render support inside the active step and remove it on close;
  - after render, call `scrollIntoView({behavior: "smooth", block: "center"})` only for the current requested step;
  - suppress smooth scrolling while replay snapshot restoration is in progress;
  - keep focus, aria-live, and keyboard behavior from the existing interaction path.

- [ ] **Step 6: Run all JavaScript and schema tests**

  ```bash
  npm test
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_schemas.py tests/test_static_pages.py'
  ```

  Expected: all tests pass; old cues display spoken text; new cues display rendered formulas while TTS uses natural speech.

- [ ] **Step 7: Commit**

  ```bash
  git add app/schemas.py app/static/cue-player.mjs app/static/lesson.js tests/cue-player.test.mjs tests/runtime-core.test.mjs tests/test_schemas.py
  git diff --check
  git commit -m "feat: play synchronized structured board cues"
  ```

---

### Task 10: Replace the classroom board with the 16px continuous Pad layout

**Files:**

- Modify: `app/static/lesson.html`
- Modify: `app/static/styles.css:1040-1450`
- Modify: `app/static/lesson.js` import version
- Modify: `app/static/lesson.html` script version
- Modify: `tests/test_static_pages.py`

- [ ] **Step 1: Write failing static layout and cache tests**

  ```python
  def test_lesson_shell_uses_single_continuous_structured_board():
      html = lesson_html()
      css = lesson_css()

      assert 'id="structured-board"' in html
      assert 'class="board-directory"' not in html
      assert ".lesson-step" in css
      assert "font-size: 16px" in css


  def test_structured_board_module_cache_chain_is_versioned(client):
      html = client.get("/lesson/example").text
      assert "lesson.js?v=20260814-1" in html
      lesson_js = client.get("/static/lesson.js?v=20260814-1").text
      assert "structured-board.mjs?v=20260814-1" in lesson_js
  ```

- [ ] **Step 2: Run static tests and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_static_pages.py -k "continuous_structured_board or structured_board_module_cache"'
  ```

  Expected: new container, CSS selectors, and version chain are missing.

- [ ] **Step 3: Implement the single-column shell**

  Replace the active board regions with this stable structure while retaining legacy containers behind the legacy runtime path:

  ```html
  <section class="chalkboard-frame" aria-label="讲解板书">
    <div id="structured-board" class="structured-board" hidden></div>
    <div id="legacy-board" class="legacy-board">
      <div id="base-board" class="board-region"></div>
      <div id="layer-stage" class="board-region" hidden></div>
    </div>
  </section>
  ```

- [ ] **Step 4: Implement the approved Pad typography**

  Use exact core styles:

  ```css
  .structured-board {
    height: 100%;
    overflow-y: auto;
    padding: 24px 28px 32px;
    font-size: 16px;
    line-height: 1.65;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
  }

  .lesson-step {
    margin: 0 0 22px;
    scroll-margin-block: 28vh;
  }

  .lesson-step__header {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 15px;
    font-weight: 650;
  }

  .lesson-step__line {
    margin: 8px 0 0 18px;
    font-size: 16px;
  }

  .lesson-step__line[data-board-role="working"] .katex-display,
  .lesson-step__line[data-board-role="summary"] .katex-display {
    font-size: 22px;
  }

  .lesson-step[data-status="completed"] {
    color: var(--board-ink-soft);
  }

  .lesson-step__support {
    margin: 10px 0 0 18px;
    padding: 10px 12px;
    border-left: 3px solid #e8a23a;
    background: #fff8e8;
  }
  ```

  Use colored state dots inside the header; do not add a separate left directory. Ensure 1024×768 and 1280×800 both preserve the full-screen classroom controls without horizontal overflow.

- [ ] **Step 5: Complete the cache-busting chain**

  Bump only changed lesson assets. The unversioned lesson HTML remains `Cache-Control: no-cache`; versioned JS/CSS/module responses must not receive `no-cache` or `no-store`.

- [ ] **Step 6: Run static and JavaScript tests**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_static_pages.py'
  npm test
  node --check app/static/lesson.js
  ```

  Expected: all tests pass and the version chain is complete.

- [ ] **Step 7: Commit**

  ```bash
  git add app/static/lesson.html app/static/styles.css app/static/lesson.js tests/test_static_pages.py
  git diff --check
  git commit -m "feat: render continuous tablet teaching board"
  ```

---

### Task 11: Voice all main and support cues with deterministic manifests

**Files:**

- Modify: `app/audio_manifest.py`
- Modify: `app/audio_service.py`
- Modify: `app/generation_integrity.py`
- Modify: `tests/test_tts_client.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing support-audio and pronunciation tests**

  ```python
  @pytest.mark.asyncio
  async def test_audio_service_voices_main_and_support_cues_from_spoken_text(tmp_path):
      lesson = structured_runtime_lesson()
      service, fake_tts = audio_service(tmp_path)

      voiced = await service.attach_audio(lesson)

      assert "我们要求的是 m 减 n。" in fake_tts.texts
      assert "二 n 整体的平方等于四 n 的平方。" in fake_tts.texts
      support = voiced.beats[0].interaction.options[1].support_cues
      assert all(cue.audio_url for cue in support)


  def test_audio_manifest_rejects_cross_option_support_url():
      lesson = voiced_structured_runtime_lesson()
      payload = lesson.model_dump(mode="python")
      payload["beats"][0]["interaction"]["options"][1]["support_cues"][0][
          "audio_url"
      ] = payload["beats"][0]["interaction"]["options"][2]["support_cues"][0][
          "audio_url"
      ]

      with pytest.raises(ValueError, match="support cue audio manifest mismatch"):
          validate_lesson_audio_manifest(RuntimeLesson.model_validate(payload))
  ```

- [ ] **Step 2: Run TTS/API tests and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_tts_client.py tests/test_api.py -k "support_cue or deterministic_pronunciation"'
  ```

  Expected: support cues are not traversed or voiced and manifest validation ignores them.

- [ ] **Step 3: Define deterministic support asset IDs**

  ```python
  def support_cue_asset_id(
      beat_id: str,
      interaction_id: str,
      option_id: str,
      cue_id: str,
  ) -> str:
      return validated_audio_asset_id(
          "support-%s-%s-%s-%s"
          % (beat_id, interaction_id, option_id, cue_id)
      )
  ```

  Preflight every derived asset ID before creating the lesson audio directory. Reject duplicates and any collision with main cue, beat, hint, option-feedback, or correct-feedback assets.

- [ ] **Step 4: Voice support cues under the existing bounded concurrency**

  Include support cues in the same lesson-wide semaphore and cancellation group as main cues. Always synthesize `cue.spoken_text`; never send `display_text` or raw math markup to Volcengine. On any provider failure or cancellation, cancel remaining tasks and remove only the directory exclusively created by this request.

- [ ] **Step 5: Validate post-audio semantic neutrality**

  Generation integrity may differ only in these audio fields:

  - beat `audio_url`;
  - main cue `audio_url`;
  - hint/correct/option feedback audio URLs;
  - nested support cue `audio_url`.

  All display text, spoken text, step IDs, interactions, actions, and resume behavior must remain identical to the pre-TTS snapshot.

- [ ] **Step 6: Run full audio, API, and integrity suites**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_tts_client.py tests/test_api.py tests/test_generation.py'
  ```

  Expected: all tests pass, including missing/external/cross-lesson/swapped URLs, cleanup timeout, cancellation, no-overwrite ownership, and exact Volcengine request text.

- [ ] **Step 7: Commit**

  ```bash
  git add app/audio_manifest.py app/audio_service.py app/generation_integrity.py tests/test_tts_client.py tests/test_api.py
  git diff --check
  git commit -m "feat: voice adaptive teaching support cues"
  ```

---

### Task 12: Upgrade the rubric, golden evidence, and end-to-end acceptance

**Files:**

- Modify: `app/pedagogy_rubric.py`
- Modify: `app/preparation_prompts.py`
- Modify: `app/preparation_validation.py`
- Modify: `tests/fixtures/pedagogy_golden_cases.json`
- Modify: `scripts/run_pedagogy_evaluation.py`
- Modify: `tests/test_pedagogy_evaluation.py`
- Modify: `tests/test_preparation_pipeline.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_pages.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing hard-gate tests**

  Add exact criterion IDs and tests:

  ```python
  EXPECTED_NEW_CRITERIA = {
      "step_purpose_and_transition",
      "display_speech_math_alignment",
      "adaptive_explanation_depth",
      "continuous_board_structure",
      "current_step_visible",
  }


  def test_rubric_contains_structured_teaching_hard_gates():
      payload = rubric_payload()
      ids = {item["criterion_id"] for item in payload["criteria"]}

      assert EXPECTED_NEW_CRITERIA <= ids


  @pytest.mark.parametrize(
      "failure",
      (
          "missing_root_meaning",
          "missing_about_x_reason",
          "missing_square_warning",
          "missing_nonzero_condition",
          "missing_opposite_relation",
          "minus_not_spoken",
          "wrong_support_not_deeper",
          "step_never_completed",
      ),
  )
  def test_parameter_root_lesson_fails_each_required_gate(failure):
      artifacts = mutate_parameter_root_prepared_lesson(failure)

      with pytest.raises(PreparationValidationError):
          validate_prepared_lesson(artifacts)
  ```

- [ ] **Step 2: Run rubric/golden tests and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_validation.py tests/test_pedagogy_evaluation.py -k "structured_teaching or parameter_root_lesson"'
  ```

  Expected: criteria and evidence metrics are absent.

- [ ] **Step 3: Publish rubric version 0.2**

  Add the five criteria as hard requirements. Keep `current_emphasis_correct` and `learner_follows_why` non-compensable. Reviewer findings must cite an exact artifact ID and concrete evidence; approval is impossible if simulation fails any step's attention, reason, execution, continuation, display/speech, support-depth, or visibility ability.

- [ ] **Step 4: Strengthen the parameter-root golden case**

  Extend `parameter_root_01` with exact reviewable expectations:

  ```json
  {
    "required_step_labels": [
      "第一步：理解方程的根",
      "第二步：代入方程",
      "第三步：展开并化简",
      "第四步：利用n不等于0约去n",
      "第五步：整理出m-n"
    ],
    "required_spoken_forms": [
      {"display": "m-n", "spoken_contains": "m 减 n"},
      {"display": "n≠0", "spoken_contains": "n 不等于零"}
    ],
    "required_error_codes": [
      "substitution-variable-error",
      "square-distribution-error",
      "nonzero-condition-error",
      "opposite-expression-error"
    ]
  }
  ```

  Keep all 18 existing case IDs and breadth tags. Add optional structured metrics to every successful run: step coverage, must-teach-to-script coverage, must-teach-to-board coverage, display/speech alignment, diagnostic branch coverage, and step lifecycle coverage. Metrics must retain the existing exact status/count/ratio invariants.

- [ ] **Step 5: Add a deterministic end-to-end fixture test**

  Run the actual pipeline, adapter, compiler, audio fake, atomic store, GET endpoint, and interaction evaluator using the approved parameter-root responses. Assert:

  - five steps in exact order;
  - structured step metadata and bounded actions;
  - `m-n` display with spoken “m 减 n”;
  - correct substitution response is brief;
  - square-error response contains a worked intermediate line;
  - wrong response continues exactly once;
  - generated lesson and private record survive a new store instance;
  - public GET strips expected answers, error diagnostics, review findings, and private progression.

- [ ] **Step 6: Update README operating instructions**

  Document:

  - the new private chain and final Teaching Agent ownership;
  - old lesson compatibility;
  - how to run focused Python/JS tests;
  - how to run the Golden Set comparison;
  - the exact authorized live smoke inputs;
  - that the service remains stopped unless the operator starts it for acceptance.

- [ ] **Step 7: Run all automated verification**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q'
  npm test
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && python -m compileall -q app scripts tests'
  node --check app/static/lesson.js
  git diff --check
  ```

  Expected: every command exits 0; no test is deleted; `.superpowers/` remains the only pre-existing untracked path.

- [ ] **Step 8: Run authorized live acceptance**

  Start the service only after automated GREEN. Generate this exact lesson through the real OpenAI-compatible model and Volcengine TTS:

  ```text
  question: 若$2n$ ($n\ne 0$)是关于 x的方程 $x^2-2mx+2n=0$的根，则m-n的值为
  answer: $\frac{1}{2}$
  explanation: 因为 $2n(n\ne 0)$ 是关于x的方程$x^2-2mx+2n=0$的解
  所以 $4n^2-4mn+2n=0$
  所以$4n-4m+2=0$
  所以$m-n=\frac{1}{2}$
  ```

  Verify in a real browser at both 1280×800 and 1024×768:

  1. body text computes to 16px;
  2. no separate left directory is present;
  3. five colored step headers appear progressively;
  4. current step scrolls into the central visible area;
  5. problem highlights appear only when the corresponding speech begins;
  6. `m-n` audio audibly includes “减”;
  7. the square-error option opens a worked support section, plays support audio, closes it, and continues;
  8. correct response is shorter than the wrong support path;
  9. previous step summaries remain reviewable and their emphasis is softened;
  10. reload by lesson ID starts without regeneration and preserves all audio URLs.

  Save the new lesson ID and two acceptance screenshots outside public runtime JSON. Stop the service after acceptance.

- [ ] **Step 9: Commit**

  ```bash
  git add app/pedagogy_rubric.py app/preparation_prompts.py app/preparation_validation.py tests/fixtures/pedagogy_golden_cases.json scripts/run_pedagogy_evaluation.py tests/test_pedagogy_evaluation.py tests/test_preparation_pipeline.py tests/test_api.py tests/test_static_pages.py README.md
  git diff --check
  git commit -m "test: verify adaptive structured teaching quality"
  ```

---

## Final completion gate

Do not declare the feature complete until all of these are true:

- every new artifact is request-scoped, versioned, private, and repairable;
- every student-visible sentence is owned by the final Teaching Agent;
- display and speech math are separately validated;
- wrong-answer support is deeper, bounded, non-blocking, and returns to the correct main location;
- board step headers, lines, support, emphasis, and scrolling are generated only through whitelisted actions;
- old lesson IDs remain playable;
- new records persist exact progression and cue provenance;
- no incomplete lesson/audio/private-record pair can be opened;
- the parameter-root sample passes the five-step human review;
- all 18 Golden Set cases remain structurally valid;
- full Python, JavaScript, compile, syntax, cache, persistence, and browser checks are GREEN;
- the service is stopped after live acceptance.
