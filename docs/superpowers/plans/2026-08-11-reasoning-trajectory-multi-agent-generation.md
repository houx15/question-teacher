# Reasoning-Trajectory Multi-Agent Lesson Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current one-pass narrative/material generation and fixed two-revision review with a traceable preparation pipeline that identifies each mathematical decision, designs an interleaved reasoning trajectory, writes a clause-addressable teaching script, binds interactions and board actions to exact clauses, and admits a lesson only after evidence-based review converges.

**Architecture:** Keep the current public generation API, lesson ID, compiler, Volcengine TTS, SQLite runtime cache, and full-screen classroom. Insert a deterministic `LessonPreparationPipeline` after the verified/grounded teaching route and before `LessonCompiler`. Seven bounded LLM roles exchange validated Pydantic artifacts; deterministic validators enforce IDs, references, coverage, privacy, and visual legality; the orchestrator routes review findings back to the earliest responsible role and invalidates only downstream artifacts. The existing compiler receives an adapted `LessonDraft`, so saved old lessons remain playable.

**Tech Stack:** Python 3.9, FastAPI, Pydantic v2, SQLite, existing OpenAI-compatible JSON client, pytest/pytest-asyncio, vanilla JavaScript runtime tests, KaTeX, Volcengine TTS.

---

## Delivery rules

- Work in the current repository and branch. Do not create a worktree.
- Activate the approved Python environment for every Python command:

  ```bash
  source /opt/anaconda3/etc/profile.d/conda.sh
  conda activate general
  ```

- Use Python 3.9-compatible typing (`Optional`, `List`, `Dict`, `Union`); do not introduce `X | Y` annotations.
- Do not touch the untracked `.superpowers/` directory.
- Keep the server stopped until the final browser-acceptance task.
- Use test-driven development: add one failing behavior at a time, verify the failure is caused by the missing behavior, implement the minimum production change, rerun the focused tests, then commit.
- Never put API keys, access tokens, raw provider payloads, or private answer feedback into logs or public lesson payloads.
- Every commit must pass `git diff --check` before it is created.

## Stable boundaries

The following existing contracts remain public and unchanged:

- `POST /api/lessons/generate` accepts `ProblemInput`.
- `GET /api/jobs/{job_id}` exposes the same job contract.
- Successful generation still returns a lesson ID and does not auto-navigate.
- `GET /api/lessons/{lesson_id}` continues to strip answers, private feedback, and validation records.
- `RuntimeLesson`, Volcengine cue audio, and the full-screen classroom remain the playback format.
- Existing saved lessons are not migrated or regenerated.
- Reference answer, reference explanation, and the existing frozen teaching route remain the mathematical fact boundary; this work does not add a general solver, grader, or correction product.

The new private chain is:

```text
ProblemInput + FrozenTeachingRoute
  -> SolutionTrace
  -> ReasoningTrajectory
  -> TeachingScript
  -> InteractionPlan
  -> PerformanceScore
  -> SimulationReport + LessonReviewDecision
  -> PreparedLesson
  -> LessonDraft
  -> existing LessonCompiler
  -> RuntimeLesson + generation record
```

## Target file map

Create:

- `app/preparation_models.py`
- `app/preparation_prompts.py`
- `app/preparation_validation.py`
- `app/math_content.py`
- `app/preparation_pipeline.py`
- `app/prepared_lesson_adapter.py`
- `app/pedagogy_rubric.py`
- `tests/preparation_fakes.py`
- `tests/test_preparation_models.py`
- `tests/test_preparation_prompts.py`
- `tests/test_preparation_validation.py`
- `tests/test_math_content.py`
- `tests/test_preparation_pipeline.py`
- `tests/test_prepared_lesson_adapter.py`
- `tests/fixtures/pedagogy_golden_cases.json`
- `scripts/run_pedagogy_evaluation.py`
- `tests/test_pedagogy_evaluation.py`

Modify:

- `app/generation.py`
- `app/compiler.py`
- `app/store.py`
- `app/api.py`
- `app/prompts.py` only to delete obsolete imports after migration
- `tests/generation_fakes.py`
- `tests/test_generation.py`
- `tests/test_generation_agents.py`
- `tests/test_compiler.py`
- `tests/test_store.py`
- `tests/test_api.py`
- `README.md`

The classroom HTML, CSS, cue player, runtime reducer, and math renderer are not changed unless final browser verification exposes a regression.

---

## Task 1: Add the private preparation artifact contracts

**Files:**

- Create: `app/preparation_models.py`
- Create: `tests/test_preparation_models.py`

- [ ] **Step 1: Write failing schema and reference tests**

  Cover all of these behaviors explicitly:

  - every artifact rejects unknown fields;
  - IDs are non-empty and unique in their local artifact;
  - `ReasoningEpisode.mode` accepts exactly `understand`, `plan`, `explore`, `execute`, `monitor`, `revise`, `reflect`;
  - `TeachingScript.spoken_text` is the only explanatory-narration source and rejects blank text; interaction prompts and feedback come only from `InteractionPlan`;
  - `InteractionPlan` allows zero interactions and never requires an artificial fixed count;
  - every interaction option has one unique ID and exactly one option matches `correct_option_id`;
  - review severity and responsible-role values are closed literals;
  - `PreparedLesson` carries the rubric version and complete artifact history.

  Use builders in the test file rather than large inline dictionaries. The canonical test should read:

  ```python
  def test_reasoning_trajectory_accepts_interleaved_modes() -> None:
      trajectory = make_trajectory(
          modes=["understand", "plan", "execute", "monitor", "revise"]
      )

      assert [episode.mode for episode in trajectory.episodes] == [
          "understand",
          "plan",
          "execute",
          "monitor",
          "revise",
      ]


  def test_interaction_plan_does_not_force_an_interaction() -> None:
      plan = InteractionPlan(interactions=[], transfer_item=make_transfer_item())

      assert plan.interactions == []
  ```

- [ ] **Step 2: Run the focused test and confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_models.py'
  ```

  Expected: collection fails because `app.preparation_models` does not exist.

- [ ] **Step 3: Implement the complete model vocabulary**

  Use `SchemaModel` from `app.schemas` so Pydantic behavior remains consistent. Define the following exact public names and fields:

  ```python
  EvidenceStatus = Literal["quoted", "derived", "inferred", "verified_route"]
  ReasoningMode = Literal[
      "understand", "plan", "explore", "execute", "monitor", "revise", "reflect"
  ]
  TrajectoryType = Literal["planned", "exploratory", "hybrid"]
  PedagogicalFunction = Literal[
      "focus", "question", "explain", "decide", "execute",
      "observe", "correct", "transition", "review", "summarize"
  ]
  DiagnosticKind = Literal["conception", "execution"]
  ArtifactType = Literal[
      "solution_trace", "reasoning_trajectory", "teaching_script",
      "interaction_plan", "performance_score", "simulation_report"
  ]
  ResponsibleRole = Literal[
      "reference_analyst", "teaching_designer", "script_teacher",
      "interaction_designer", "classroom_director"
  ]
  RoleName = Literal[
      "reference_analyst", "teaching_designer", "script_teacher",
      "interaction_designer", "classroom_director", "student_simulator",
      "lesson_reviewer"
  ]

  class SourceAnchor(SchemaModel):
      source_kind: Literal["problem", "answer", "solution", "verified_route"]
      source_id: GeneratedId
      excerpt: NonEmptyString

  class SolutionTraceStep(SchemaModel):
      source_step_id: GeneratedId
      source_anchor: SourceAnchor
      state_before: NonEmptyString
      mathematical_action: NonEmptyString
      justification: NonEmptyString
      state_after: NonEmptyString
      new_information: NonEmptyString
      assumption_ids_used: List[GeneratedId] = Field(default_factory=list)
      omitted_reasoning: List[NonEmptyString] = Field(default_factory=list)
      evidence_status: EvidenceStatus

  class TraceAssumption(SchemaModel):
      assumption_id: GeneratedId
      content: NonEmptyString
      source_anchor: SourceAnchor

  class SolutionTrace(SchemaModel):
      task_target: NonEmptyString
      reference_conclusion: NonEmptyString
      assumptions: List[TraceAssumption] = Field(default_factory=list)
      source_steps: List[SolutionTraceStep] = Field(min_length=1)
      audit_notes: List[NonEmptyString] = Field(default_factory=list)

  class MustTeachItem(SchemaModel):
      must_teach_id: GeneratedId
      content: NonEmptyString
      why_it_matters: NonEmptyString

  class ReasoningEpisode(SchemaModel):
      episode_id: GeneratedId
      sequence_index: int = Field(ge=0)
      mode: ReasoningMode
      source_step_ids: List[GeneratedId] = Field(min_length=1)
      learner_state_before: NonEmptyString
      attention_targets: List[NonEmptyString] = Field(min_length=1)
      thinking_question: NonEmptyString
      decision: NonEmptyString
      decision_reason: NonEmptyString
      mathematical_action: NonEmptyString
      action_justification: NonEmptyString
      result: NonEmptyString
      result_meaning: NonEmptyString
      transition_reason: NonEmptyString
      must_teach: List[MustTeachItem] = Field(min_length=1)
      likely_misconceptions: List[NonEmptyString] = Field(default_factory=list)
      interaction_intent: Optional[NonEmptyString] = None
      visual_intent: Optional[NonEmptyString] = None

  class ReasoningTrajectory(SchemaModel):
      trajectory_type: TrajectoryType
      lesson_purpose: NonEmptyString
      episodes: List[ReasoningEpisode] = Field(min_length=1)
      method_summary: NonEmptyString
      error_summary: NonEmptyString

  class ScriptClause(SchemaModel):
      clause_id: GeneratedId
      episode_id: GeneratedId
      pedagogical_function: PedagogicalFunction
      spoken_text: CueSpokenText
      math_references: List[GeneratedMathAnswer] = Field(default_factory=list)
      learner_gain: NonEmptyString
      answer_exposure: bool
      must_teach_refs: List[GeneratedId] = Field(default_factory=list)

  class TeachingScript(SchemaModel):
      title: NonEmptyString
      learning_goal: NonEmptyString
      method_rationale: NonEmptyString
      method_introduction: MethodIntroduction
      opening_clause_ids: List[GeneratedId] = Field(min_length=1)
      method_introduction_clause_ids: List[GeneratedId] = Field(min_length=1)
      clauses: List[ScriptClause] = Field(min_length=1)
      closing_summary_clause_ids: List[GeneratedId] = Field(min_length=1)

  class PlannedInteractionOption(SchemaModel):
      option_id: GeneratedId
      display_text: NonEmptyString
      canonical_answer: NonEmptyString
      misconception: Optional[NonEmptyString] = None

  class PlannedInteraction(SchemaModel):
      interaction_id: GeneratedId
      episode_id: GeneratedId
      after_clause_id: GeneratedId
      diagnostic_target: NonEmptyString
      diagnostic_kind: DiagnosticKind
      prompt: NonEmptyString
      options: List[PlannedInteractionOption] = Field(min_length=3, max_length=4)
      correct_option_id: GeneratedId
      correct_feedback: NonEmptyString
      incorrect_feedback_by_option: Dict[GeneratedId, NonEmptyString]
      hint: NonEmptyString
      resume_clause_id: GeneratedId
      concealed_targets: List[GeneratedId] = Field(default_factory=list)

  class InteractionPlan(SchemaModel):
      interactions: List[PlannedInteraction] = Field(default_factory=list, max_length=3)
      transfer_item: GeneratedTransferItem

  class ClauseBoundVisualAction(SchemaModel):
      clause_id: GeneratedId
      action: SyncVisualAction

  class PerformanceCue(SchemaModel):
      cue_id: GeneratedId
      clause_ids: List[GeneratedId] = Field(min_length=1)
      lead_actions: List[ClauseBoundVisualAction] = Field(default_factory=list)
      start_actions: List[ClauseBoundVisualAction] = Field(default_factory=list)
      end_actions: List[ClauseBoundVisualAction] = Field(default_factory=list)

  class PerformanceBoardObject(SchemaModel):
      board_object_id: GeneratedId
      content: NarrativeBoardContent
      layer: Literal["base", "micro_explanation", "comparison"] = "base"

  class OverlayTransition(SchemaModel):
      transition_id: GeneratedId
      after_clause_id: GeneratedId
      action: Literal["enter", "return"]
      layer: Literal["micro_explanation", "comparison"]

  class PerformanceScore(SchemaModel):
      cues: List[PerformanceCue] = Field(min_length=1)
      board_objects: List[PerformanceBoardObject] = Field(default_factory=list)
      overlay_transitions: List[OverlayTransition] = Field(default_factory=list)

  class EpisodeSimulationResult(SchemaModel):
      episode_id: GeneratedId
      learner_profile: NonEmptyString
      can_identify_attention_target: bool
      can_explain_decision: bool
      can_execute_action: bool
      can_use_result_to_continue: bool
      evidence: List[NonEmptyString] = Field(min_length=1)

  class SimulationReport(SchemaModel):
      episode_results: List[EpisodeSimulationResult] = Field(min_length=1)
      interaction_results: List[NonEmptyString] = Field(default_factory=list)
      end_of_lesson_recall: NonEmptyString
      blocking_findings: List[NonEmptyString] = Field(default_factory=list)

  class ReviewFinding(SchemaModel):
      finding_id: GeneratedId
      severity: Literal["blocking", "material", "polish"]
      artifact_type: ArtifactType
      artifact_id: GeneratedId
      criterion: NonEmptyString
      evidence: NonEmptyString
      responsible_role: ResponsibleRole
      requested_change: NonEmptyString
      invalidated_downstream_artifacts: List[ArtifactType] = Field(default_factory=list)

  class LessonReviewDecision(SchemaModel):
      status: Literal["approved", "revision_required", "failed"]
      findings: List[ReviewFinding] = Field(default_factory=list)
      retained_artifacts: List[ArtifactType] = Field(default_factory=list)
      approval_summary: NonEmptyString

  class ArtifactRevision(SchemaModel):
      artifact_type: ArtifactType
      version: int = Field(ge=1)
      responsible_role: ResponsibleRole
      finding_ids: List[GeneratedId] = Field(default_factory=list)

  class PreparedLesson(SchemaModel):
      rubric_version: NonEmptyString
      solution_trace: SolutionTrace
      reasoning_trajectory: ReasoningTrajectory
      teaching_script: TeachingScript
      interaction_plan: InteractionPlan
      performance_score: PerformanceScore
      simulation_report: SimulationReport
      review: LessonReviewDecision
      repair_count: int = Field(ge=0)
      artifact_history: List[ArtifactRevision] = Field(min_length=5)

  class RoleCallRecord(SchemaModel):
      role: RoleName
      input_artifact_versions: Dict[ArtifactType, int] = Field(default_factory=dict)
      output_artifact_type: Optional[ArtifactType] = None
      output_artifact_version: Optional[int] = Field(default=None, ge=1)
      duration_ms: int = Field(ge=0)
      retry_count: int = Field(ge=0)
      failure_category: Optional[NonEmptyString] = None
      token_usage: Optional[Dict[str, int]] = None
      review_finding_ids: List[GeneratedId] = Field(default_factory=list)

  class GenerationRecord(SchemaModel):
      generation_id: NonEmptyString
      lesson_id: NonEmptyString
      route_fingerprint: NonEmptyString
      prepared_lesson: PreparedLesson
      role_calls: List[RoleCallRecord] = Field(min_length=7)
      created_at: NonEmptyString
  ```

  Import the existing `CueSpokenText`, `GeneratedId`, `GeneratedMathAnswer`, `GeneratedTransferItem`, `MethodIntroduction`, `NarrativeBoardContent`, `NonEmptyString`, `SchemaModel`, and `SyncVisualAction` from `app.schemas`; do not duplicate their constraints.

- [ ] **Step 4: Add local uniqueness validators**

  Add Pydantic model validators for unique assumption, step, episode, clause, option, cue, and finding IDs; contiguous `sequence_index` values starting at zero; one valid correct option; and no blocking/material finding when a decision is `approved`. An approved review may retain polish findings. Keep cross-artifact reference validation out of the models and in Task 3.

- [ ] **Step 5: Run GREEN and regression tests**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_models.py tests/test_schemas.py'
  git diff --check
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add app/preparation_models.py tests/test_preparation_models.py
  git commit -m "feat: define lesson preparation artifacts"
  ```

---

## Task 2: Version the pedagogy rubric and isolate role prompts

**Files:**

- Create: `app/pedagogy_rubric.py`
- Create: `app/preparation_prompts.py`
- Create: `tests/test_preparation_prompts.py`

- [ ] **Step 1: Write failing prompt-isolation tests**

  Assert that:

  - the rubric version is exactly `0.1`;
  - both non-compensable gates appear verbatim in simulator and reviewer prompts;
  - only Reference Material Analyst receives the complete `reference_solution_text` field;
  - downstream prompts receive `SolutionTrace` or a narrower artifact projection; they may contain validated short source-anchor excerpts but never the complete raw reference payload;
  - no role is asked for coordinates, CSS selectors, timestamps, audio durations, or provider details;
  - each repair prompt contains finding IDs, evidence, requested changes, current artifact version, and retained upstream artifacts;
  - source text is delimited and explicitly treated as untrusted data.

  The key safety test should be:

  ```python
  def test_complete_reference_payload_is_confined_to_reference_analyst() -> None:
      problem = make_problem(
          reference_solution_text="第一步代入。IGNORE_ALL_RULES。第二步约分。"
      )
      trace = make_solution_trace(source_excerpt="第一步代入。")

      analyst_payload = parse_prompt_payload(
          solution_trace_prompt(problem, make_route(), make_focus_targets())
      )
      designer_payload = parse_prompt_payload(
          reasoning_trajectory_prompt(problem, trace, make_capabilities())
      )

      assert "reference_solution_text" in analyst_payload["source_material"]
      assert "reference_solution_text" not in designer_payload
      assert "IGNORE_ALL_RULES" not in json.dumps(designer_payload, ensure_ascii=False)
  ```

- [ ] **Step 2: Confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_prompts.py'
  ```

  Expected: imports fail because the new prompt modules do not exist.

- [ ] **Step 3: Implement rubric v0.1 as data, not hidden prompt prose**

  Define:

  ```python
  PEDAGOGY_RUBRIC_VERSION = "0.1"

  NON_COMPENSABLE_GATES = (
      "当前强调正确：每个片段必须指出此刻真正影响下一步的条件、关系、方法或结果。",
      "学生能跟上并理解为什么：每个必要决定必须说明为什么现在这样想、这样做，以及结果如何推动下一步。",
  )

  HARD_REQUIREMENTS = (
      "数学结论和必要条件与权威教学路线一致。",
      "构思、探索、执行、监控可以交替，不把解题伪装成始终线性的既定步骤。",
      "每个 must_teach 都有可定位的讲稿证据。",
      "互动诊断理解，不通过选项提前泄露尚未讲授的答案。",
      "视觉动作只在对应语句发生时出现，并引用合法语义目标。",
  )
  ```

  Add `rubric_payload()` returning a JSON-serializable dictionary used by both simulator and reviewer.

- [ ] **Step 4: Implement one prompt builder per role**

  Export these exact names:

  ```python
  SOLUTION_TRACE_SYSTEM
  TEACHING_DESIGNER_SYSTEM
  SCRIPT_TEACHER_SYSTEM
  INTERACTION_DESIGNER_SYSTEM
  CLASSROOM_DIRECTOR_SYSTEM
  STUDENT_SIMULATOR_SYSTEM
  LESSON_REVIEWER_SYSTEM

  solution_trace_prompt(problem, teaching_route, focus_targets, repair=None)
  reasoning_trajectory_prompt(problem, solution_trace, capabilities, repair=None)
  teaching_script_prompt(reasoning_trajectory, repair=None)
  interaction_plan_prompt(reasoning_trajectory, teaching_script, repair=None)
  performance_score_prompt(problem_targets, teaching_script, interaction_plan, capabilities, repair=None)
  student_simulation_prompt(reasoning_trajectory, teaching_script, interaction_plan, performance_score)
  lesson_review_prompt(prepared_artifacts, simulation_report, reviewer_context_id)
  ```

  Every builder must serialize structured data with `json.dumps(payload, ensure_ascii=False, sort_keys=True)` and produce an explicit JSON-only response request. Role systems contain these boundaries:

  ```text
  Reference Material Analyst: distinguish quoted, derived, inferred, and verified-route evidence; never silently repair the reference answer.
  Teaching Designer: design the learner's actual reasoning order; preserve mathematical dependencies; explain why attention moves at every transition.
  Script Teacher: write only words a student should hear; preserve every must_teach item; do not design visuals or timestamps.
  Interaction Designer: add a choice only when it diagnoses conception or execution; exactly one correct option; do not reveal future answers.
  Classroom Director: bind semantic actions to exact clause IDs; do not alter spoken text; do not emit pixels, selectors, or milliseconds.
  Student Simulator: evaluate whether a novice can identify the focus, explain the decision, execute it, and use the result to continue.
  Lesson Reviewer: cite evidence, assign the earliest responsible role, and approve only when no blocking or material findings remain.
  ```

- [ ] **Step 5: Run GREEN and commit**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_prompts.py'
  git diff --check
  git add app/pedagogy_rubric.py app/preparation_prompts.py tests/test_preparation_prompts.py
  git commit -m "feat: define pedagogy rubric and role prompts"
  ```

---

## Task 3: Enforce cross-artifact traceability deterministically

**Files:**

- Create: `app/preparation_validation.py`
- Create: `app/math_content.py`
- Modify: `app/generation.py`
- Modify: `app/schemas.py`
- Create: `tests/test_preparation_validation.py`
- Create: `tests/test_math_content.py`

- [ ] **Step 1: Write one failing test per invalid edge**

  Test the following edge failures and their stable error codes:

  | Invalid edge | Error code |
  |---|---|
  | trace conclusion differs from frozen route | `trace_conclusion_mismatch` |
  | trace step cites missing assumption | `trace_assumption_missing` |
  | episode cites missing trace step | `episode_source_missing` |
  | a trace step is unused without an audit note | `trace_step_uncovered` |
  | `must_teach` has no script clause evidence | `must_teach_uncovered` |
  | clause cites missing episode | `clause_episode_missing` |
  | interaction cites missing or earlier-incompatible clause | `interaction_clause_invalid` |
  | interaction correct option is exposed in concealed target | `interaction_answer_leakage` |
  | cue skips, duplicates, or reorders clauses | `cue_clause_coverage_invalid` |
  | cue mixes non-adjacent clauses | `cue_clause_nonadjacent` |
  | visual action cites a clause outside its cue | `visual_clause_invalid` |
  | visual action target is unknown at that cue | `visual_target_invalid` |
  | an action appears before the clause that introduces its content | `visual_action_too_early` |
  | spoken text contains LaTeX delimiters or internal highlight syntax | `spoken_markup_invalid` |
  | choice labels normalize to the same KaTeX expression | `choice_formula_duplicate` |
  | board content contains unsupported mixed formula markup | `board_formula_invalid` |
  | approved review still has blocking/material findings | `review_approval_invalid` |

  Also test the representative parameter-root trace through the full coverage matrix.

- [ ] **Step 2: Confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_validation.py'
  ```

- [ ] **Step 3: Implement a structured validator error**

  ```python
  class PreparationValidationError(ValueError):
      def __init__(self, code: str, artifact_id: str, detail: str) -> None:
          super().__init__(detail)
          self.code = code
          self.artifact_id = artifact_id
          self.detail = detail
  ```

  Export:

  ```python
  validate_solution_trace(trace, teaching_route) -> None
  validate_reasoning_trajectory(trajectory, trace) -> None
  validate_teaching_script(script, trajectory) -> None
  validate_interaction_plan(plan, trajectory, script) -> None
  validate_performance_score(score, problem_targets, script, plan) -> None
  validate_simulation_report(report, trajectory, plan) -> None
  validate_review_decision(decision) -> None
  validate_prepared_lesson(prepared, teaching_route, problem_targets) -> None
  blocking_signature(decision) -> str
  ```

- [ ] **Step 4: Implement `must_teach` coverage without fuzzy substring guessing**

  Treat each `MustTeachItem.must_teach_id` as the stable edge. Validation requires every ID one or more times in `ScriptClause.must_teach_refs`, and the covering clauses must belong to the same episode. Reject missing IDs, cross-episode references, and references to nonexistent IDs. Never use fuzzy prose matching as the acceptance mechanism.

- [ ] **Step 5: Reuse current semantic-action legality**

  Extract the current choice-label normalization, answer-leak normalization, math-markup detection, and generated formula checks from `app/generation.py`/`app/schemas.py` into `app/math_content.py`. Keep thin imports at the old call sites so legacy behavior is unchanged. Call the shared helpers from preparation validation; do not create a second regex dialect. Likewise, extract or call current semantic-action legality rather than copying its allowlists. Add regression tests in `tests/test_math_content.py` and `tests/test_generation_agents.py` for inline `$x=1$`, display `$$x=1$$`, `\(x=1\)`, `\[x=1\]`, escaped backslashes, KaTeX-equivalent option labels, and internal target/highlight tokens. Prove that natural spoken Chinese stays markup-free while formulas remain in `math_references`, option display labels, or semantic board content.

- [ ] **Step 6: Run GREEN and commit**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_models.py tests/test_preparation_validation.py tests/test_math_content.py tests/test_generation_agents.py'
  git diff --check
  git add app/preparation_models.py app/preparation_validation.py app/math_content.py tests/test_preparation_models.py tests/test_preparation_validation.py tests/test_math_content.py app/generation.py app/schemas.py tests/test_generation_agents.py
  git commit -m "feat: validate preparation traceability"
  ```

---

## Task 4: Generate and validate SolutionTrace and ReasoningTrajectory

**Files:**

- Create: `app/preparation_pipeline.py`
- Create: `tests/preparation_fakes.py`
- Create: `tests/test_preparation_pipeline.py`

- [ ] **Step 1: Build a role-aware fake client**

  Implement a FIFO fake keyed by system-prompt identity rather than by call index alone:

  ```python
  class PreparationFakeClient:
      def __init__(self, responses_by_role: Dict[str, List[object]]) -> None:
          self.responses_by_role = responses_by_role
          self.calls: List[RecordedCall] = []

      async def complete_json(self, system: str, user: str) -> object:
          role = role_for_system(system)
          self.calls.append(RecordedCall(role=role, system=system, user=user))
          return self.responses_by_role[role].pop(0)
  ```

  Keep it independent from the legacy `tests/generation_fakes.py` until Task 8.

- [ ] **Step 2: Write failing early-pipeline tests**

  Cover:

  - stages run in the order `reference_analyst -> teaching_designer`;
  - raw reference text reaches only the analyst call;
  - each role gets one schema retry on invalid JSON/schema output;
  - a second invalid structure raises `invalid_structure` with the role name;
  - deterministic trace validation fails before the designer is called;
  - a trajectory containing `plan -> execute -> monitor -> revise -> execute` is accepted;
  - a trajectory may be planned, exploratory, or hybrid and is not forced to contain every mode;
  - the parameter-root problem includes the four indispensable moves: substitute the root, connect to the target relation, use `n != 0`, and return to `m - n`.

- [ ] **Step 3: Confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_pipeline.py -k "trace or trajectory or structure"'
  ```

- [ ] **Step 4: Implement the early orchestrator**

  Define:

  ```python
  from dataclasses import dataclass, field

  class PreparationFailure(RuntimeError):
      def __init__(self, category: str, role: str, detail: str) -> None:
          super().__init__(detail)
          self.category = category
          self.role = role
          self.detail = detail

  @dataclass
  class PreparationState:
      solution_trace: Optional[SolutionTrace] = None
      reasoning_trajectory: Optional[ReasoningTrajectory] = None
      teaching_script: Optional[TeachingScript] = None
      interaction_plan: Optional[InteractionPlan] = None
      performance_score: Optional[PerformanceScore] = None
      simulation_report: Optional[SimulationReport] = None
      review: Optional[LessonReviewDecision] = None
      versions: Dict[str, int] = field(default_factory=dict)
      history: List[ArtifactRevision] = field(default_factory=list)

  class LessonPreparationPipeline:
      MAX_STRUCTURE_ATTEMPTS = 2
      MAX_REPAIR_CYCLES = 8
  ```

  Export the async method `prepare(self, problem: ProblemInput, teaching_route: FrozenTeachingRoute, problem_focus_targets: List[ProblemFocusTarget], on_stage: Optional[StageCallback] = None) -> PreparedLesson`. Its initial implementation constructs `PreparationState`, calls the trace stage, calls the trajectory stage, then continues into the downstream methods added in Tasks 5 and 6.

  Add a private `_complete_model(role, system, prompt, model_type)` that catches provider errors separately from JSON/Pydantic errors. It retries structure once, never retries deterministic content validation as a structure error, and appends a `RoleCallRecord` containing artifact versions, duration, retry count, safe failure category, token usage when supplied, and routed finding IDs. The record must not contain prompts, source text, provider payloads, or credentials.

- [ ] **Step 5: Add the first two stage methods**

  Implement `_create_solution_trace` and `_create_reasoning_trajectory`. Validate each result immediately. Increment artifact versions only after a valid result exists. Emit these internal stages exactly:

  ```text
  整理参考解析
  设计解题思维轨迹
  ```

- [ ] **Step 6: Run GREEN and commit**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_pipeline.py -k "trace or trajectory or structure"'
  git diff --check
  git add app/preparation_pipeline.py tests/preparation_fakes.py tests/test_preparation_pipeline.py
  git commit -m "feat: prepare solution trace and reasoning trajectory"
  ```

---

## Task 5: Generate script, interaction, and clause-bound performance score

**Files:**

- Modify: `app/preparation_pipeline.py`
- Create: `app/prepared_lesson_adapter.py`
- Modify: `app/schemas.py`
- Modify: `app/compiler.py`
- Modify: `tests/test_preparation_pipeline.py`
- Create: `tests/test_prepared_lesson_adapter.py`
- Modify: `tests/test_schemas.py`
- Modify: `tests/test_compiler.py`

- [ ] **Step 1: Write failing downstream-stage tests**

  Prove:

  - Script Teacher covers every `must_teach_id` and cannot reorder episode dependencies;
  - the complete ordered concatenation of script clauses is the only explanatory-narration source; interaction prompt/feedback audio is derived only from `InteractionPlan` and fixed runtime navigation phrases remain unchanged;
  - Interaction Designer may return zero interactions;
  - each accepted interaction diagnoses either conception or execution and has exactly one correct option;
  - Classroom Director preserves clause order, uses every clause exactly once, and cannot change spoken text;
  - not every cue needs a highlight;
  - a highlight on the screen's only element without a discriminating purpose is rejected as `non_discriminating_emphasis`;
  - a board equation can be emphasized or faded after it is introduced;
  - a problem target is highlighted only in the cue whose clause discusses it;
  - an overlay can enter, teach one point, and return before the main board continues.

- [ ] **Step 2: Confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_pipeline.py -k "script or interaction or performance" tests/test_prepared_lesson_adapter.py'
  ```

- [ ] **Step 3: Implement three downstream stage methods**

  Add `_create_teaching_script`, `_create_interaction_plan`, and `_create_performance_score`. Each stage calls its deterministic validator immediately and emits:

  ```text
  编写讲稿
  设计互动
  编排板书与高亮
  ```

  Do not synthesize timestamps. `PerformanceCue.clause_ids` must be a contiguous slice of script order. The adapter concatenates only those exact clause texts.

- [ ] **Step 4: Implement the adapter to the existing `LessonDraft`**

  Export `prepared_lesson_to_draft(problem: ProblemInput, prepared: PreparedLesson, teaching_route: FrozenTeachingRoute, verified_math_steps: Optional[List[MathStep]] = None) -> LessonDraft`. Require a non-empty `verified_math_steps` list in symbolic mode and reject it in grounded modes, matching the current `LessonDraft.math_steps` invariant without importing the private `_VerifiedMathRoute` class into the adapter.

  The adapter must:

  - map one interaction-bearing episode to one `LessonMoment`;
  - merge only adjacent, interaction-free episodes when cue order remains intact;
  - preserve the episode-to-runtime mapping in the private generation record;
  - create each existing narrative sync cue from the ordered `clause_ids` and exact `spoken_text` concatenation;
  - convert planned choices into existing runtime `Interaction` and `InteractionOption` objects without exposing the correct option publicly;
  - reuse the frozen teaching route, script-authored method introduction, supplied verified math steps in symbolic mode, problem focus targets, and transfer item expected by `LessonDraft`;
  - leave actual audio URLs empty for the current audio service to fill.

- [ ] **Step 5: Preserve episode IDs through compilation**

  Keep episode-to-runtime mapping solely in the private generation record for this release. Do not add a new public/runtime beat field. Test that the existing `RuntimeLesson` JSON contract does not change.

- [ ] **Step 6: Make new opening, method, and summary speech clause-authoritative**

  Extend `LessonDraft` with optional `opening_sync_cues`, `method_introduction_sync_cues`, and `summary_sync_cues`. Update `LessonCompiler` to use these authored cues when present and retain its existing synthesized opening/method/summary behavior when they are absent. The adapter must populate all three from `TeachingScript` clause-ID groups, and each resulting cue must preserve the exact clause text and unwrapped `SyncVisualAction` values from `ClauseBoundVisualAction`. Add regression tests proving a legacy `LessonDraft` compiles equivalently while a prepared draft introduces no explanatory narration outside `TeachingScript.clauses`; interaction prompts/feedback remain sourced from `InteractionPlan`.

- [ ] **Step 7: Run focused GREEN and compiler regression**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_pipeline.py -k "script or interaction or performance" tests/test_prepared_lesson_adapter.py tests/test_schemas.py tests/test_compiler.py'
  git diff --check
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add app/preparation_pipeline.py app/prepared_lesson_adapter.py app/schemas.py app/compiler.py tests/test_preparation_pipeline.py tests/test_prepared_lesson_adapter.py tests/test_schemas.py tests/test_compiler.py
  git commit -m "feat: compile clause-bound teaching performance"
  ```

---

## Task 6: Add student simulation, evidence review, and targeted repair convergence

**Files:**

- Modify: `app/preparation_pipeline.py`
- Modify: `app/preparation_validation.py`
- Modify: `tests/test_preparation_pipeline.py`
- Modify: `tests/test_preparation_validation.py`

- [ ] **Step 1: Write failing simulator and review tests**

  Require the simulator to report, for every episode, whether a novice can:

  - identify the current attention target;
  - explain the decision;
  - execute the mathematical action;
  - use the result to continue.

  Require the reviewer to:

  - cite a concrete artifact ID and evidence for every finding;
  - reject when either non-compensable gate fails;
  - assign the earliest responsible role;
  - never rewrite artifacts directly;
  - approve only with no blocking/material findings.

- [ ] **Step 2: Write the targeted-repair state-machine tests**

  Cover each route independently:

  | Responsible role | Retain | Rebuild |
  |---|---|---|
  | `reference_analyst` | frozen teaching route | trace and all downstream artifacts |
  | `teaching_designer` | trace | trajectory and all downstream artifacts |
  | `script_teacher` | trace, trajectory | script and all downstream artifacts |
  | `interaction_designer` | trace, trajectory, script | interaction, performance, simulation, review |
  | `classroom_director` | trace, trajectory, script, interaction | performance, simulation, review |

  Also prove:

  - multiple findings select the earliest responsible role in dependency order;
  - unaffected upstream version numbers remain unchanged;
  - any repaired artifact increments exactly once;
  - simulator and reviewer always rerun after a change;
  - there is no fixed “two revisions then accept” behavior;
  - eight unresolved repair cycles raise `review_not_converged` and never compile;
  - the same blocking signature twice switches to a fresh reviewer context;
  - the fresh reviewer returning the same signature raises `review_not_converged` immediately;
  - a polish-only review can approve, while blocking/material findings cannot.

- [ ] **Step 3: Confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_pipeline.py -k "simulation or review or repair or converge"'
  ```

- [ ] **Step 4: Implement simulation and review stages**

  Add `_simulate_student` and `_review_lesson`. Emit one public-safe internal stage:

  ```text
  模拟学生并审核课程
  ```

  `blocking_signature` must hash only canonical `(severity, artifact_type, artifact_id, criterion, responsible_role)` tuples. Do not include free-form wording, ordering, or provider-generated finding IDs.

- [ ] **Step 5: Implement dependency-aware repair**

  Add:

  ```python
  ROLE_ORDER = {
      "reference_analyst": 0,
      "teaching_designer": 1,
      "script_teacher": 2,
      "interaction_designer": 3,
      "classroom_director": 4,
  }

  def earliest_responsible_role(findings: List[ReviewFinding]) -> ResponsibleRole:
      material = [finding for finding in findings if finding.severity != "polish"]
      return min(material, key=lambda finding: ROLE_ORDER[finding.responsible_role]).responsible_role
  ```

  Add the async method `_repair_from(self, role: ResponsibleRole, state: PreparationState, findings: List[ReviewFinding], context: PreparationContext) -> None`. Implement it as an explicit five-branch dispatch in `ROLE_ORDER`: call the selected role with its repair payload, clear only artifacts downstream of that role, regenerate those cleared artifacts in dependency order, then rerun simulation and review. Reject an unknown role before mutating state.

  Every repair prompt contains only the current responsible artifact, retained upstream artifacts, and routed findings. It must not give a downstream role authority to alter upstream mathematics or pedagogy.

- [ ] **Step 6: Build `PreparedLesson` only after approval**

  The final path must call `validate_prepared_lesson` and require `review.status == "approved"`. A `failed` reviewer decision or exhausted convergence budget raises `PreparationFailure(category="review_not_converged", role="lesson_reviewer", detail="课程审核未收敛。")`.

- [ ] **Step 7: Run GREEN and commit**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_preparation_pipeline.py tests/test_preparation_validation.py'
  git diff --check
  git add app/preparation_pipeline.py app/preparation_validation.py tests/test_preparation_pipeline.py tests/test_preparation_validation.py
  git commit -m "feat: converge lessons through targeted review"
  ```

---

## Task 7: Replace the legacy narrative/material loop in `LessonGenerationService`

**Files:**

- Modify: `app/generation.py`
- Modify: `app/prompts.py`
- Modify: `tests/generation_fakes.py`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_generation_agents.py`

- [ ] **Step 1: Replace obsolete behavioral expectations with new failures**

  Remove or rewrite tests that assert:

  - `MAX_REVISIONS = 2`;
  - Director creates a monolithic `NarrativeDraft` before material generation;
  - Reviewer rewrites the whole narrative;
  - every revision regenerates all materials.

  Add integration tests asserting:

  - verified and grounded teaching routes still run before preparation;
  - `LessonPreparationPipeline` receives the frozen route and semantic problem targets;
  - approved prepared output goes through the adapter and current compiler;
  - rejected/non-converged output never reaches the compiler or TTS;
  - existing public `generate()` still returns `RuntimeLesson`;
  - validation report records rubric version, artifact versions, repair count, route fingerprint, and final review status, but not raw private artifacts;
  - the raw reference solution is visible only to the current reference audit/grounding and new reference-analysis stages, never script/interaction/director/simulator/reviewer.

- [ ] **Step 2: Confirm RED against the old service**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_generation.py tests/test_generation_agents.py'
  ```

  Expected: new stage order and preparation dependency assertions fail; old two-revision tests are gone.

- [ ] **Step 3: Inject and call the preparation pipeline**

  Extend the existing constructor with the final optional argument `preparation_pipeline: Optional[LessonPreparationPipeline] = None`. Assign `self.preparation_pipeline = preparation_pipeline or LessonPreparationPipeline(client)` after the existing dependencies; preserve every existing default and argument order.

  After `FrozenTeachingRoute` and `problem_focus_targets` exist:

  ```python
  prepared = await self.preparation_pipeline.prepare(
      problem,
      teaching_route,
      problem_focus_targets,
      on_stage=on_stage,
  )
  draft = prepared_lesson_to_draft(
      problem,
      prepared,
      teaching_route,
      verified_math_steps=(
          verified_route.thaw().math_steps
          if verified_route is not None
          else None
      ),
  )
  ```

  Delete the calls to `_create_validated_narrative`, `_create_validated_materials`, `_review`, and `_revise` from the active path. Keep legacy private methods for one commit only if focused tests still require them; delete them and their unused prompt constants before this task's final commit.

- [ ] **Step 4: Preserve `generate()` and add a private-record result for API use**

  Define `GeneratedLessonBundle` exactly as follows:

  ```python
  class GeneratedLessonBundle(SchemaModel):
      lesson: RuntimeLesson
      generation_record: GenerationRecord
  ```

  Move the full active generation implementation into `generate_bundle(self, problem: ProblemInput, on_stage: Optional[StageCallback] = None) -> GeneratedLessonBundle`. It constructs `GenerationRecord` from the approved `PreparedLesson`, compiles the runtime, and returns both. Keep `generate(self, problem: ProblemInput, on_stage: Optional[StageCallback] = None) -> RuntimeLesson` as a one-line compatibility wrapper returning `(await self.generate_bundle(problem, on_stage=on_stage)).lesson`.

  `GenerationRecord` contains the original private artifacts, rubric version, route fingerprint, review result, artifact history, and generation ID. This keeps direct callers and existing fakes compatible while allowing the API to persist private evidence.

- [ ] **Step 5: Update fakes without making production prompt order implicit**

  Compose `PreparationFakeClient` into the existing generation fake. Existing route/audit fixtures remain in `tests/generation_fakes.py`; new role fixtures live in `tests/preparation_fakes.py`. Tests should assert role names and artifacts rather than a brittle total call count.

- [ ] **Step 6: Run GREEN and commit**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_generation.py tests/test_generation_agents.py tests/test_preparation_pipeline.py tests/test_compiler.py'
  git diff --check
  git add app/generation.py app/prompts.py tests/generation_fakes.py tests/test_generation.py tests/test_generation_agents.py
  git commit -m "refactor: generate lessons from prepared reasoning"
  ```

---

## Task 8: Persist private generation records atomically with lessons

**Files:**

- Modify: `app/store.py`
- Modify: `app/api.py`
- Modify: `tests/test_store.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing persistence tests**

  Prove:

  - `save_lesson(lesson, generation_record=record)` inserts runtime and record in one transaction;
  - the second table has exactly `lesson_id`, `generation_id`, `rubric_version`, `record_json`, `created_at`;
  - `lesson_id` is both primary key and foreign key to `lessons(lesson_id)` with cascade delete semantics;
  - a record insertion failure rolls back the lesson insertion and does not populate the memory cache;
  - duplicate lesson IDs preserve the current exact `ValueError` behavior;
  - corrupt or mismatched private record JSON fails closed;
  - reopening the store after restart returns the same private record through a private method;
  - `get_lesson` and public APIs work for old databases without a generation-record row;
  - no public lesson payload contains the generation record.

- [ ] **Step 2: Confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_store.py -k "generation_record or atomic" tests/test_api.py -k "generation_record"'
  ```

- [ ] **Step 3: Add the table inside the existing transaction**

  Use this schema:

  ```sql
  CREATE TABLE IF NOT EXISTS lesson_generation_records (
      lesson_id TEXT PRIMARY KEY,
      generation_id TEXT NOT NULL UNIQUE,
      rubric_version TEXT NOT NULL,
      record_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY (lesson_id) REFERENCES lessons(lesson_id) ON DELETE CASCADE
  )
  ```

  Change the compatible signature to `save_lesson(self, lesson: RuntimeLesson, generation_record: Optional[GenerationRecord] = None) -> None`.

  Insert the optional record after the lesson insert and before the single commit. Cache only after the transaction succeeds. Add `get_generation_record(lesson_id) -> Optional[GenerationRecord]` as a server-private method.

- [ ] **Step 4: Make `run_generation` prefer `generate_bundle` without breaking fakes**

  Use capability detection only at the boundary:

  ```python
  if callable(getattr(generator, "generate_bundle", None)):
      bundle = await generator.generate_bundle(problem, on_stage=report_stage)
      lesson = bundle.lesson
      generation_record = bundle.generation_record
  else:
      lesson = await generator.generate(problem, on_stage=report_stage)
      generation_record = None

  lesson = await audio_service.attach_audio(lesson, on_stage=report_stage)
  store.save_lesson(lesson, generation_record=generation_record)
  ```

  This preserves current API fakes and allows audio failure to prevent both runtime and private record persistence.

- [ ] **Step 5: Run GREEN, full store/API regression, and commit**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_store.py tests/test_api.py'
  git diff --check
  git add app/store.py app/api.py tests/test_store.py tests/test_api.py
  git commit -m "feat: persist private lesson preparation records"
  ```

---

## Task 9: Expose stable user-facing preparation progress and failure categories

**Files:**

- Modify: `app/api.py`
- Modify: `app/generation.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_generation.py`

- [ ] **Step 1: Write failing stage and safe-error tests**

  Assert the internal-to-public mapping:

  ```python
  {
      "整理参考解析": "正在整理参考解析",
      "设计解题思维轨迹": "正在设计解题思维轨迹",
      "编写讲稿": "正在编写讲稿",
      "设计互动": "正在设计互动",
      "编排板书与高亮": "正在编排板书与高亮",
      "模拟学生并审核课程": "正在审核和优化课程",
      "正在编译课堂": "正在编译课堂",
      "正在生成讲解语音": "正在生成讲解语音",
      "正在保存课程": "正在保存课程",
  }
  ```

  Repeated repair stages must not flood the job feed with internal role details. All preparation failures remain the generic public message `课程生成失败，请稍后重试。`; `LessonInputError` keeps its current user-correctable messages.

- [ ] **Step 2: Confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_api.py -k "stage or error"'
  ```

- [ ] **Step 3: Implement stage mapping and structured internal diagnostics**

  Extend `_PUBLIC_GENERATION_STAGES` with the exact mapping above. Add structured internal preparation categories to the generation record:

  ```text
  provider_error
  invalid_structure
  reference_trace_failed
  reasoning_design_failed
  review_not_converged
  compile_failed
  tts_failed
  persistence_failed
  ```

  Do not place provider payloads, keys, raw prompts, answer text, or private feedback in user-visible errors.

  In `run_generation`, call `report_stage("正在保存课程")` immediately before `store.save_lesson(lesson, generation_record=generation_record)`. Do not emit completion or expose the lesson ID until that atomic save succeeds.

- [ ] **Step 4: Run GREEN and commit**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_api.py tests/test_generation.py'
  git diff --check
  git add app/api.py app/generation.py tests/test_api.py tests/test_generation.py
  git commit -m "feat: report lesson preparation progress safely"
  ```

---

## Task 10: Add the 18-case pedagogy golden set and repeatable comparison runner

**Files:**

- Create: `tests/fixtures/pedagogy_golden_cases.json`
- Create: `scripts/run_pedagogy_evaluation.py`
- Create: `tests/test_pedagogy_evaluation.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing fixture-contract tests**

  Require exactly 18 unique case IDs and coverage tags for:

  ```text
  concept_condition_conversion
  algebra_execution
  equation_parameter
  method_selection
  text_only_geometry
  function_relationship
  omitted_condition
  exploration_or_revision
  concept_overlay
  no_forced_interaction
  no_forced_emphasis
  ```

  Every case must contain:

  ```json
  {
    "case_id": "parameter_root_01",
    "problem": {
      "problem_text": "若2n（n不等于0）是方程x^2-2mx+2n=0的根，求m-n。",
      "reference_answer": "1/2",
      "reference_solution_text": "将x=2n代入，得到4n^2-4mn+2n=0；由n不等于0，整理得4n-4m+2=0，所以m-n=1/2。",
      "lesson_length": "standard"
    },
    "coverage_tags": ["equation_parameter", "omitted_condition"],
    "trace_anchors": ["是根意味着代入", "约去n前使用n不等于0", "结果重新连接m-n"],
    "required_reasoning_modes": ["plan", "execute", "monitor"],
    "required_must_teach": ["目标只需要m与n的关系", "含字母因子约分前确认非零"],
    "typical_misconceptions": ["直接除以n却不检查非零条件", "只算代入而不解释为什么"],
    "required_board_states": ["x=2n代入原方程", "提取2n后的关系式", "m-n=1/2"],
    "acceptable_excerpt_patterns": ["先看目标", "因为n不等于0"],
    "unacceptable_excerpt_patterns": ["直接把n约掉", "答案显然是二分之一"]
  }
  ```

  The fixture validates metadata and teacher-authored expectations; it does not claim model success or student learning.

- [ ] **Step 2: Confirm RED**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_pedagogy_evaluation.py'
  ```

- [ ] **Step 3: Author the 18 reviewed cases**

  Include the parameter-root demonstration as case 1. Select the other 17 from junior-math text-only categories in the design spec. For each case, write teacher-reviewable anchors and failure examples; do not fabricate automatic correctness scores.

- [ ] **Step 4: Implement a deterministic evaluation runner**

  CLI:

  ```bash
  python scripts/run_pedagogy_evaluation.py \
    --rubric-version 0.1 \
    --runs-per-case 3 \
    --output-dir /tmp/ai-math-pedagogy-v01
  ```

  The runner must:

  - require an explicit output directory and never overwrite a non-empty run directory;
  - generate each case three times through the real service when credentials are configured;
  - save private records and public runtime summaries separately;
  - calculate deterministic contract metrics: generation success, hard-gate review status, must-teach coverage, clause/action binding, schema/runtime pass, duration, and call count;
  - create blinded A/B pair records for teacher comparison without labeling candidate version;
  - never infer teacher preference or learning effect;
  - redact provider credentials and private feedback from logs.

  Put all network execution behind an explicit `RUN_INTEGRATION=1`; unit tests use fakes and a temporary directory.

- [ ] **Step 5: Document quality-evaluation boundaries**

  Add a README section containing:

  - how to run contract tests offline;
  - how to run three live generations per case;
  - how to compare two rubric/prompt versions blindly;
  - the distinction among automatic review, teacher preference, and student learning evidence;
  - the approved conda environment command.

- [ ] **Step 6: Run GREEN and commit**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q tests/test_pedagogy_evaluation.py'
  git diff --check
  git add tests/fixtures/pedagogy_golden_cases.json scripts/run_pedagogy_evaluation.py tests/test_pedagogy_evaluation.py README.md
  git commit -m "test: add pedagogy golden evaluation set"
  ```

---

## Task 11: Verify the representative problem end to end

**Files:**

- Modify only files required by defects found during verification.

- [ ] **Step 1: Run all offline Python and JavaScript tests**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q'
  npm test
  node --check app/static/generate.js
  node --check app/static/lesson.js
  git diff --check
  git status --short
  ```

  Expected: all tests pass; the only pre-existing untracked path is `.superpowers/`.

- [ ] **Step 2: Run one credentialed generation for the approved parameter-root problem**

  Use exactly:

  ```text
  question: 若$2n$ ($n\ne 0$)是关于 x的方程 $x^2-2mx+2n=0$的根，则m-n的值为
  answer: $\frac{1}{2}$
  explanation: 因为 $2n(n\ne 0)$ 是关于x的方程$x^2-2mx+2n=0$的解
               所以 $4n^2-4mn+2n=0$
               所以$4n-4m+2=0$
               所以$m-n=\frac{1}{2}$
  ```

  Confirm the private record proves these eight points before opening the classroom:

  1. “是根” is interpreted as substituting `x = 2n`.
  2. The lesson first connects the target to finding a relation between `m` and `n`.
  3. Substitution and expansion are executed correctly.
  4. The common factor is noticed before division.
  5. `n != 0` is explicitly recalled at the division decision.
  6. The result is reconnected to `m - n`.
  7. At least one interaction diagnoses a real decision or condition.
  8. The final summary states both “root means substitute” and “check nonzero before cancelling a variable factor”.

- [ ] **Step 3: Start the service only for browser acceptance**

  Start it using the existing README command in the `general` environment. Record the process ID so it can be stopped after verification. Do not change credentials or write them into tracked files.

- [ ] **Step 4: Verify the full-screen horizontal classroom**

  At a 1280×800 viewport, verify and capture evidence that:

  - real Volcengine speech plays;
  - the problem highlight appears with the corresponding spoken clause, not at lesson load;
  - key derivation steps appear on the board as they are spoken;
  - an already emphasized key point becomes subdued rather than globally green;
  - not every cue has an unnecessary highlight;
  - no lone formula is circled without a discriminating teaching purpose;
  - board emphasis can target an already-written equation;
  - any overlay returns to the preserved main board;
  - a choice interaction blocks progress and resumes correctly;
  - KaTeX renders all formulas with no raw highlight syntax or rendering warning;
  - pause, resume, replay, previous beat, and interaction feedback remain correct.

- [ ] **Step 5: Verify durable reopening by ID**

  Record the generated lesson ID, stop the service, restart it, and open the same lesson ID. Verify playback and private interaction grading survive restart. Then stop the service again.

- [ ] **Step 6: Fix only observed defects using focused RED/GREEN loops**

  For every defect, first add the narrowest automated regression test. If a visual browser-only defect cannot be expressed in the current unit harness, add a fixture-based runtime contract test and retain the screenshot as manual evidence. Do not broaden the classroom design during this task.

- [ ] **Step 7: Run final verification and commit any fixes**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q'
  npm test
  node --check app/static/generate.js
  node --check app/static/lesson.js
  git diff --check
  git status --short
  ```

  If verification required code changes:

  ```bash
  git add <only-the-files-changed-by-the-fix>
  git commit -m "fix: stabilize prepared lesson playback"
  ```

---

## Task 12: Independent specification and code-quality review

**Files:**

- Modify only files required by actionable review findings.

- [ ] **Step 1: Request a specification-compliance review**

  The reviewer must compare implementation against:

  - `docs/superpowers/specs/2026-08-11-reasoning-trajectory-multi-agent-generation-design.md`
  - this plan
  - the representative parameter-root acceptance criteria

  Require evidence for role isolation, traceability, targeted invalidation, no fixed-round approval, public privacy, and old-lesson compatibility.

- [ ] **Step 2: Request a code-quality review**

  Focus on Python 3.9 compatibility, Pydantic contracts, async/provider failure separation, SQLite atomicity, concurrency safety, prompt privacy, duplicated validation logic, and avoidable complexity.

- [ ] **Step 3: Resolve each actionable finding with TDD**

  For each accepted finding, add a failing regression test, implement the smallest fix, and rerun the focused suite. Do not implement speculative suggestions that are outside the approved design.

- [ ] **Step 4: Run the complete verification suite one final time**

  ```bash
  bash -lc 'source /opt/anaconda3/etc/profile.d/conda.sh && conda activate general && pytest -q'
  npm test
  node --check app/static/generate.js
  node --check app/static/lesson.js
  git diff --check
  git status --short
  ```

- [ ] **Step 5: Commit final review fixes, if any**

  ```bash
  git add <only-reviewed-files>
  git commit -m "fix: address prepared lesson review findings"
  ```

## Definition of done

Implementation is complete only when all statements below are supported by fresh evidence:

- One public submission still produces one saved lesson ID.
- The server creates and persists all five preparation artifacts plus simulation and review evidence before compilation.
- Mathematical facts remain anchored to the problem, answer, reference explanation, or frozen verified route.
- Reasoning can alternate among planning, exploration, execution, monitoring, and revision.
- Every `must_teach` item has an exact script-clause reference.
- Every visual action is bound to the clause being spoken and a legal semantic target.
- Interactions diagnose conception or execution and are not forced where they add no value.
- Both core teaching gates are non-compensable.
- No fixed number of revisions can force approval.
- Repeated no-progress review fails safely as `review_not_converged`.
- Only invalidated downstream artifacts are rebuilt.
- Current compiler, Volcengine TTS, formula rendering, full-screen runtime, pause/replay, and saved-course reopening still work.
- Old saved lessons remain readable without a generation record.
- Private artifacts and correct answers remain absent from public lesson payloads.
- The 18-case golden set and three-run comparison tool are reproducible but do not overclaim student learning.
