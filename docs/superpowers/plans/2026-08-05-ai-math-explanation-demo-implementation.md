# AI Math Explanation Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live OpenAI-compatible, mathematically validated, full-screen interactive junior-math explanation Demo for linear and quadratic equations.

**Architecture:** A FastAPI service coordinates one Lesson Director role, one independent Reviewer role, a deterministic SymPy math engine, and an in-memory lesson runtime. The Director creates a coherent whole lesson, the Reviewer evaluates the whole lesson, the Director revises when required, and a deterministic compiler turns the approved manuscript into semantic board actions and interactive beats for a horizontal Pad classroom.

**Tech Stack:** Python 3.9, FastAPI, Pydantic 2, SymPy, httpx, pytest, native HTML/CSS/JavaScript, Node 20 built-in test runner.

---

## File map

```text
.
├── .env.example                         # OpenAI-compatible service settings
├── .gitignore                           # Secrets and generated caches
├── README.md                            # Setup, run, test, and Demo walkthrough
├── requirements.txt                     # Minimal runtime dependencies
├── app/
│   ├── __init__.py
│   ├── config.py                        # Environment configuration
│   ├── main.py                          # FastAPI app and static-page routes
│   ├── api.py                           # Health, generation, job, lesson, interaction APIs
│   ├── schemas.py                       # Problem, manuscript, review, runtime schemas
│   ├── math_engine.py                   # Parsing, solving, equivalence and answer checks
│   ├── llm_client.py                    # OpenAI-compatible JSON client
│   ├── prompts.py                       # Director, Reviewer and revision prompts
│   ├── generation.py                    # Two-role orchestration and quality loop
│   ├── compiler.py                      # Manuscript to runtime beats
│   ├── store.py                         # In-memory jobs and lessons
│   └── static/
│       ├── index.html                   # Minimal generation page
│       ├── lesson.html                  # Full-screen classroom shell
│       ├── styles.css                   # Distinct horizontal-Pad visual system
│       ├── generate.js                  # Form submission and progress polling
│       ├── runtime-core.mjs             # Pure lesson state machine
│       └── lesson.js                    # Board, layers, interactions and controls
├── scripts/
│   └── smoke_live.py                    # Optional real-endpoint smoke test
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   └── demo_cases.json              # Six regression questions
    ├── test_api.py
    ├── test_compiler.py
    ├── test_config.py
    ├── test_generation.py
    ├── test_llm_client.py
    ├── test_math_engine.py
    ├── test_schemas.py
    ├── test_static_pages.py
    └── runtime-core.test.mjs
```

## Task 1: Project bootstrap and configuration

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `app/main.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
# tests/test_config.py
from app.config import Settings


def test_settings_reports_missing_model_configuration(monkeypatch):
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.model_configured is False
    assert settings.missing_model_settings == [
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ]


def test_settings_normalizes_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "demo-model")

    settings = Settings.from_env()

    assert settings.chat_completions_url == "https://example.test/v1/chat/completions"
    assert settings.model_configured is True
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_config.py
```

Expected: collection fails because `app.config` does not exist.

- [ ] **Step 3: Add minimal dependencies and configuration**

```text
# requirements.txt
fastapi>=0.128,<1
uvicorn>=0.30,<1
pydantic>=2.10,<3
sympy>=1.13,<2
httpx>=0.28,<1
```

```dotenv
# .env.example
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=replace-me
OPENAI_MODEL=replace-me
OPENAI_TIMEOUT_SECONDS=90
```

```python
# app/config.py
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Settings:
    openai_base_url: Optional[str]
    openai_api_key: Optional[str]
    openai_model: Optional[str]
    openai_timeout_seconds: float = 90.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL"),
            openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90")),
        )

    @property
    def missing_model_settings(self) -> List[str]:
        values = {
            "OPENAI_BASE_URL": self.openai_base_url,
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_MODEL": self.openai_model,
        }
        return [name for name, value in values.items() if not value]

    @property
    def model_configured(self) -> bool:
        return not self.missing_model_settings

    @property
    def chat_completions_url(self) -> str:
        if not self.openai_base_url:
            return ""
        return f"{self.openai_base_url.rstrip('/')}/chat/completions"
```

Create empty `app/__init__.py` and `tests/__init__.py`, a FastAPI instance in `app/main.py`, and ignore `.env`, `__pycache__/`, `.pytest_cache/`, `.DS_Store`, and `*.pyc`.

- [ ] **Step 4: Install the one missing runtime package**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
python -m pip install "uvicorn>=0.30,<1"
```

Expected: `uvicorn` installs into the `general` environment.

- [ ] **Step 5: Run the tests**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_config.py
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore .env.example requirements.txt app/__init__.py app/config.py app/main.py tests/__init__.py tests/test_config.py
git commit -m "chore: bootstrap FastAPI demo"
```

## Task 2: Define flexible lesson and runtime contracts

**Files:**
- Create: `app/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write schema tests**

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError

from app.schemas import BoardAction, Interaction, LessonMoment, ProblemInput


def test_problem_requires_question_and_reference_answer():
    problem = ProblemInput(
        problem_text="用配方法解方程：x^2-6x+5=0",
        reference_answer="x=1 或 x=5",
        required_method="complete_the_square",
    )
    assert problem.lesson_length == "standard"


def test_board_action_uses_semantic_targets_not_coordinates():
    action = BoardAction(
        type="annotate",
        target="linear_coefficient",
        content="一次项系数",
        annotation="circle",
    )
    assert action.target == "linear_coefficient"
    assert "x" not in action.model_dump()
    assert "y" not in action.model_dump()


def test_interaction_rejects_missing_expected_answer():
    with pytest.raises(ValidationError):
        Interaction(
            interaction_id="predict-term",
            kind="expression",
            prompt="两边应该同时加多少？",
        )


def test_moment_can_freely_combine_board_layer_and_interaction():
    moment = LessonMoment(
        purpose="理解为什么补 9",
        narration="先看一次项系数 -6，它的一半是 -3。",
        board_actions=[
            BoardAction(type="focus", target="linear_coefficient"),
            BoardAction(
                type="annotate",
                target="linear_coefficient",
                annotation="circle",
            ),
        ],
        layer="micro_explanation",
        interaction=Interaction(
            interaction_id="predict-term",
            kind="expression",
            prompt="两边应该同时加多少？",
            expected_answer="9",
            hints=["先求 -6 的一半，再平方。"],
        ),
    )
    assert moment.layer == "micro_explanation"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_schemas.py` after activating `general`.

Expected: import fails because `app.schemas` does not exist.

- [ ] **Step 3: Implement the schemas**

Define these Pydantic models in `app/schemas.py`:

```python
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ProblemInput(BaseModel):
    problem_text: str = Field(min_length=3)
    reference_answer: str = Field(min_length=1)
    required_method: Optional[
        Literal["factor", "quadratic_formula", "complete_the_square"]
    ] = None
    lesson_length: Literal["concise", "standard"] = "standard"


class MathStep(BaseModel):
    purpose: str
    operation: str
    state_before: List[str]
    state_after: List[str]
    reason: str


class BoardAction(BaseModel):
    type: Literal[
        "write",
        "transform",
        "focus",
        "annotate",
        "compare",
        "mask",
        "reveal",
        "fade",
        "pause",
        "clear",
    ]
    target: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    annotation: Optional[Literal["circle", "box", "underline", "arrow", "bracket", "label"]] = None
    relation_target: Optional[str] = None


class InteractionOption(BaseModel):
    option_id: str
    label: str


class Interaction(BaseModel):
    interaction_id: str
    kind: Literal["point_select", "choice", "expression", "free_text", "transfer"]
    prompt: str
    expected_answer: str
    options: List[InteractionOption] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)
    explanation_after_correct: str = ""


class LessonMoment(BaseModel):
    purpose: str
    narration: str
    board_actions: List[BoardAction] = Field(default_factory=list)
    layer: Literal["base", "micro_explanation", "comparison", "interaction"] = "base"
    interaction: Optional[Interaction] = None


class TransferItem(BaseModel):
    problem_text: str
    expected_answer: str
    method_signal: str


class LessonDraft(BaseModel):
    title: str
    learning_goal: str
    opening: str
    method_rationale: str
    math_steps: List[MathStep]
    moments: List[LessonMoment]
    summary: str
    transfer_item: TransferItem


class ReviewDecision(BaseModel):
    status: Literal["approved", "revision_required"]
    overall_assessment: str
    must_fix: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def revisions_need_reasons(self):
        if self.status == "revision_required" and not self.must_fix:
            raise ValueError("revision_required must include must_fix")
        return self


class RuntimeBeat(BaseModel):
    beat_id: str
    purpose: str
    narration: str
    board_actions: List[BoardAction]
    layer: str
    interaction: Optional[Interaction] = None
    next_beat_id: Optional[str] = None


class RuntimeLesson(BaseModel):
    lesson_id: str
    problem: ProblemInput
    title: str
    learning_goal: str
    beats: List[RuntimeBeat]
    summary: str
    transfer_item: TransferItem
    validation_report: Dict[str, object]


class GenerationJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    lesson_id: Optional[str] = None
    error: Optional[str] = None
```

- [ ] **Step 4: Run schema tests**

Run: `pytest -q tests/test_schemas.py` after activating `general`.

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: define lesson runtime contracts"
```

## Task 3: Build the deterministic SymPy math engine

**Files:**
- Create: `app/math_engine.py`
- Create: `tests/test_math_engine.py`

- [ ] **Step 1: Write failing math tests**

```python
# tests/test_math_engine.py
import pytest

from app.math_engine import MathEngine, MathValidationError
from app.schemas import MathStep


def test_reference_answer_matches_independent_solution():
    engine = MathEngine()
    report = engine.validate_problem("x^2-6x+5=0", "x=1 或 x=5")
    assert report.solution_strings == ["1", "5"]


def test_reference_answer_conflict_is_rejected():
    engine = MathEngine()
    with pytest.raises(MathValidationError, match="参考答案"):
        engine.validate_problem("2x+3=7", "x=3")


def test_equivalent_branch_step_passes():
    engine = MathEngine()
    step = MathStep(
        purpose="开平方",
        operation="split_plus_minus",
        state_before=["(x-3)^2=4"],
        state_after=["x-3=2", "x-3=-2"],
        reason="平方等于 4 时底数可能为 2 或 -2",
    )
    engine.validate_step(step)


def test_non_equivalent_step_fails():
    engine = MathEngine()
    step = MathStep(
        purpose="错误移项",
        operation="subtract_both_sides",
        state_before=["2x+3=7"],
        state_after=["2x=10"],
        reason="错误示例",
    )
    with pytest.raises(MathValidationError, match="解集不一致"):
        engine.validate_step(step)


def test_expression_answers_are_compared_mathematically():
    engine = MathEngine()
    assert engine.expressions_equivalent("(x-3)^2", "x^2-6x+9")


def test_solution_answers_can_use_different_order():
    engine = MathEngine()
    assert engine.answers_equivalent("x=5 或 x=1", "x=1 或 x=5")
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_math_engine.py` after activating `general`.

Expected: import fails because `app.math_engine` does not exist.

- [ ] **Step 3: Implement parsing, solving and validation**

Implement:

```python
# app/math_engine.py
import re
from dataclasses import dataclass
from typing import List

from sympy import Eq, FiniteSet, S, Symbol, simplify, solveset
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from app.schemas import MathStep


class MathValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ProblemValidation:
    solution_strings: List[str]


class MathEngine:
    def __init__(self):
        self.x = Symbol("x", real=True)
        self.transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

    def parse_expression(self, text: str):
        normalized = (
            text.strip()
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("²", "^2")
        )
        try:
            return parse_expr(
                normalized,
                local_dict={"x": self.x},
                transformations=self.transformations,
                evaluate=True,
            )
        except (SyntaxError, TypeError, ValueError) as exc:
            raise MathValidationError(f"无法解析表达式：{text}") from exc

    def parse_equation(self, text: str):
        if text.count("=") != 1:
            raise MathValidationError(f"无法解析方程：{text}")
        left, right = text.split("=", 1)
        return Eq(self.parse_expression(left), self.parse_expression(right))

    def solution_set(self, equations: List[str]):
        result = S.EmptySet
        for text in equations:
            result = result.union(
                solveset(self.parse_equation(text), self.x, domain=S.Reals)
            )
        return result

    def _reference_set(self, reference_answer: str):
        pieces = re.findall(
            r"x\\s*=\\s*([^或,，;；\\s]+)",
            reference_answer.replace("±", " 或 x=-"),
        )
        if not pieces:
            raise MathValidationError("无法解析参考答案")
        return FiniteSet(*[self.parse_expression(piece) for piece in pieces])

    def validate_problem(self, problem_text: str, reference_answer: str):
        solution = self.solution_set([problem_text.split("：")[-1].strip()])
        if not isinstance(solution, FiniteSet):
            raise MathValidationError(f"暂不支持非有限解集：{solution}")
        reference = self._reference_set(reference_answer)
        if solution != reference:
            raise MathValidationError(
                f"参考答案与独立求解冲突：求解得到 {solution}，参考答案为 {reference}"
            )
        return ProblemValidation(
            solution_strings=sorted([str(value) for value in solution])
        )

    def validate_step(self, step: MathStep):
        allowed_operations = {
            "simplify",
            "add_both_sides",
            "subtract_both_sides",
            "multiply_both_sides",
            "divide_both_sides",
            "expand",
            "factor",
            "combine_like_terms",
            "take_square_root_both_sides",
            "split_plus_minus",
            "complete_the_square",
            "quadratic_formula",
        }
        if step.operation not in allowed_operations:
            raise MathValidationError(f"不支持的数学操作：{step.operation}")
        before = self.solution_set(step.state_before)
        after = self.solution_set(step.state_after)
        if before != after:
            raise MathValidationError(
                f"步骤“{step.purpose}”前后解集不一致：{before} != {after}"
            )

    def expressions_equivalent(self, actual: str, expected: str) -> bool:
        return simplify(
            self.parse_expression(actual) - self.parse_expression(expected)
        ) == 0

    def answers_equivalent(self, actual: str, expected: str) -> bool:
        return self._reference_set(actual) == self._reference_set(expected)
```

- [ ] **Step 4: Run math tests**

Run: `pytest -q tests/test_math_engine.py` after activating `general`.

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/math_engine.py tests/test_math_engine.py
git commit -m "feat: validate equation solutions with SymPy"
```

## Task 4: Implement the OpenAI-compatible JSON client

**Files:**
- Create: `app/llm_client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing client tests**

```python
# tests/test_llm_client.py
import asyncio

import httpx

from app.config import Settings
from app.llm_client import OpenAICompatibleClient


def test_client_posts_chat_completions_and_parses_json():
    async def run():
        def handler(request):
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer secret"
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"status":"approved"}'}}
                    ]
                },
            )

        transport = httpx.MockTransport(handler)
        settings = Settings("https://example.test/v1", "secret", "demo-model")
        client = OpenAICompatibleClient(settings, transport=transport)
        result = await client.complete_json("system", "user")
        await client.close()
        return result

    assert asyncio.run(run()) == {"status": "approved"}


def test_client_strips_markdown_json_fence():
    assert OpenAICompatibleClient.parse_json_content(
        '```json\\n{"ok":true}\\n```'
    ) == {"ok": True}
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_llm_client.py` after activating `general`.

Expected: import fails because `app.llm_client` does not exist.

- [ ] **Step 3: Implement the client**

```python
# app/llm_client.py
import json
from typing import Any, Dict, Optional

import httpx

from app.config import Settings


class ModelResponseError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.settings = settings
        self.http = httpx.AsyncClient(
            timeout=settings.openai_timeout_seconds,
            transport=transport,
        )

    @staticmethod
    def parse_json_content(content: str) -> Dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\\n", 1)[1].rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"模型没有返回有效 JSON：{exc}") from exc

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        request_body = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
            "response_format": {"type": "json_object"},
        }
        response = await self.http.post(
            self.settings.chat_completions_url,
            headers=headers,
            json=request_body,
        )
        if response.status_code in {400, 422}:
            request_body.pop("response_format")
            response = await self.http.post(
                self.settings.chat_completions_url,
                headers=headers,
                json=request_body,
            )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError("模型响应缺少 choices[0].message.content") from exc
        return self.parse_json_content(content)

    async def close(self):
        await self.http.aclose()
```

- [ ] **Step 4: Run client tests**

Run: `pytest -q tests/test_llm_client.py` after activating `general`.

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/llm_client.py tests/test_llm_client.py
git commit -m "feat: add OpenAI-compatible model client"
```

## Task 5: Orchestrate Director, whole-lesson review and revision

**Files:**
- Create: `app/prompts.py`
- Create: `app/compiler.py`
- Create: `app/generation.py`
- Create: `tests/test_generation.py`
- Create: `tests/test_compiler.py`

- [ ] **Step 1: Write failing generation-loop tests**

```python
# tests/test_generation.py
import asyncio

from app.generation import LessonGenerationService
from app.math_engine import MathEngine
from app.schemas import ProblemInput


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.responses.pop(0)


def valid_draft():
    return {
        "title": "把二次式补成一个平方",
        "learning_goal": "理解配方项来自哪里，并完成求解。",
        "opening": "先别急着算，我们先找可以变成平方的结构。",
        "method_rationale": "二次项系数是 1，适合构造完全平方。",
        "math_steps": [
            {
                "purpose": "移出常数项",
                "operation": "subtract_both_sides",
                "state_before": ["x^2-6x+5=0"],
                "state_after": ["x^2-6x=-5"],
                "reason": "等式两边同时减 5",
            },
            {
                "purpose": "构造完全平方",
                "operation": "complete_the_square",
                "state_before": ["x^2-6x=-5"],
                "state_after": ["x^2-6x+9=4"],
                "reason": "等式两边同时加 9",
            },
            {
                "purpose": "开平方",
                "operation": "split_plus_minus",
                "state_before": ["(x-3)^2=4"],
                "state_after": ["x-3=2", "x-3=-2"],
                "reason": "平方根有正负两个分支",
            },
        ],
        "moments": [
            {
                "purpose": "预测配方项",
                "narration": "一次项系数是 -6，它的一半再平方是多少？",
                "board_actions": [
                    {"type": "focus", "target": "linear_coefficient"},
                    {"type": "mask", "target": "term_to_add"},
                ],
                "layer": "interaction",
                "interaction": {
                    "interaction_id": "predict-term",
                    "kind": "expression",
                    "prompt": "两边应该同时加多少？",
                    "expected_answer": "9",
                    "hints": ["先求 -6 的一半，再平方。"],
                    "explanation_after_correct": "对，(-6÷2)^2=9。",
                },
            }
        ],
        "summary": "看到 x²+bx，考虑加上 (b/2)²。",
        "transfer_item": {
            "problem_text": "用配方法解 x^2-8x+7=0",
            "expected_answer": "x=1 或 x=7",
            "method_signal": "二次项系数为 1，可以构造完全平方。",
        },
    }


def test_approved_draft_is_not_rewritten():
    client = FakeClient([
        valid_draft(),
        {
            "status": "approved",
            "overall_assessment": "主线清楚",
            "must_fix": [],
            "evidence": ["互动位于配方关键点"],
        },
    ])
    service = LessonGenerationService(client, MathEngine())
    lesson = asyncio.run(service.generate(ProblemInput(
        problem_text="用配方法解方程：x^2-6x+5=0",
        reference_answer="x=1 或 x=5",
        required_method="complete_the_square",
    )))
    assert len(client.calls) == 2
    assert lesson.validation_report["review_status"] == "approved"


def test_revision_required_returns_to_director():
    revised = valid_draft()
    revised["opening"] = "我们先观察一次项系数，再决定补多少。"
    client = FakeClient([
        valid_draft(),
        {
            "status": "revision_required",
            "overall_assessment": "开场过快",
            "must_fix": ["在给出方法前引导观察一次项系数"],
            "evidence": ["开场直接说配方"],
        },
        revised,
        {
            "status": "approved",
            "overall_assessment": "已经修正",
            "must_fix": [],
            "evidence": ["观察先于方法"],
        },
    ])
    service = LessonGenerationService(client, MathEngine())
    lesson = asyncio.run(service.generate(ProblemInput(
        problem_text="用配方法解方程：x^2-6x+5=0",
        reference_answer="x=1 或 x=5",
        required_method="complete_the_square",
    )))
    assert len(client.calls) == 4
    assert lesson.validation_report["revision_count"] == 1
```

- [ ] **Step 2: Write compiler tests**

```python
# tests/test_compiler.py
from app.compiler import LessonCompiler
from app.schemas import LessonDraft, ProblemInput
from tests.test_generation import valid_draft


def test_compiler_assigns_navigation_ids():
    lesson = LessonCompiler().compile(
        ProblemInput(
            problem_text="x^2-6x+5=0",
            reference_answer="x=1 或 x=5",
        ),
        LessonDraft.model_validate(valid_draft()),
        {"review_status": "approved"},
    )
    assert lesson.beats[0].beat_id == "beat-001"
    assert lesson.beats[0].next_beat_id == "beat-002"
    assert lesson.lesson_id
```

- [ ] **Step 3: Run and verify failure**

Run: `pytest -q tests/test_generation.py tests/test_compiler.py` after activating `general`.

Expected: imports fail because generation and compiler modules do not exist.

- [ ] **Step 4: Implement prompts and orchestration**

Use these role prompts in `app/prompts.py`:

```python
import json

from app.schemas import LessonDraft, ProblemInput, ReviewDecision


DIRECTOR_SYSTEM = """
你是无图初中数学课堂的 Lesson Director。你的任务是为初学者创作一节完整、
连贯、能实际播放的单题讲解。你拥有渐进板书、结构聚焦、变换演示、对照比较、
互动留白、总结压缩、圈注、高亮、遮罩、标记和临时讲解图层。

教学表达可以自由设计，不要机械填充固定七段模板。你必须：
1. 围绕一个清楚的教学主线创作整节课；
2. 只在真实认知转折点安排 1-3 个互动；
3. 使用语义 target，不输出坐标、字号或动画参数；
4. 在 math_steps 中列出所有结论关键的数学状态；
5. 互动发生前不得在 narration 或 board_actions 中泄露 expected_answer；
6. 生成一道同结构、不同表面的近迁移题；
7. 只返回符合给定 JSON Schema 的 JSON，不返回 Markdown。
"""

REVIEWER_SYSTEM = """
你是独立教研 Reviewer。阅读原题、目标学生和整节课后，从整体上判断学生能否
跟上主线、看见重点、理解关键理由，并通过互动与近迁移产生真实思考。检查临时
图层是否帮助理解且能返回主线。不要逐段代写讲稿；只返回 approved 或
revision_required、整体判断、必须修改的问题及其原文证据。只返回 JSON。
"""

REVISION_SYSTEM = """
你仍是这节课唯一的 Lesson Director。根据 Reviewer 的整篇意见重新审视并改写
完整课程，保持统一的教学叙事。修复问题，但不要把意见机械追加为新段落。
数学事实、互动答案和近迁移题仍须可验证。只返回完整 LessonDraft JSON。
"""


def director_prompt(problem: ProblemInput, solution_strings):
    return json.dumps(
        {
            "problem": problem.model_dump(),
            "independent_solutions": solution_strings,
            "lesson_schema": LessonDraft.model_json_schema(),
        },
        ensure_ascii=False,
    )


def reviewer_prompt(problem: ProblemInput, draft: LessonDraft):
    return json.dumps(
        {
            "problem": problem.model_dump(),
            "whole_lesson": draft.model_dump(),
            "review_schema": ReviewDecision.model_json_schema(),
        },
        ensure_ascii=False,
    )


def revision_prompt(problem, draft, review):
    return json.dumps(
        {
            "problem": problem.model_dump(),
            "current_whole_lesson": draft.model_dump(),
            "review": review.model_dump(),
            "lesson_schema": LessonDraft.model_json_schema(),
        },
        ensure_ascii=False,
    )
```

Implement `LessonGenerationService` with this control flow:

```python
class LessonGenerationService:
    MAX_REVISIONS = 2

    def __init__(self, client, math_engine, compiler=None):
        self.client = client
        self.math_engine = math_engine
        self.compiler = compiler or LessonCompiler()

    async def generate(self, problem, on_stage=None):
        await self._emit(on_stage, "正在验证数学路线")
        problem_report = self.math_engine.validate_problem(
            problem.problem_text,
            problem.reference_answer,
        )
        await self._emit(on_stage, "正在设计完整讲解")
        draft = await self._create_draft(problem, problem_report)
        revision_count = 0

        while True:
            self._validate_draft(problem, draft)
            await self._emit(on_stage, "正在进行整篇审稿")
            review = await self._review(problem, draft)
            if review.status == "approved":
                break
            if revision_count >= self.MAX_REVISIONS:
                raise LessonQualityError("整篇讲稿在两轮修订后仍未通过")
            await self._emit(on_stage, "正在修订完整讲解")
            draft = await self._revise(problem, draft, review)
            revision_count += 1

        await self._emit(on_stage, "正在编译课堂")
        validation_report = {
            "math_status": "verified",
            "review_status": review.status,
            "revision_count": revision_count,
            "independent_solutions": problem_report.solution_strings,
            "review_assessment": review.overall_assessment,
        }
        return self.compiler.compile(problem, draft, validation_report)

    async def _create_draft(self, problem, problem_report):
        payload = await self.client.complete_json(
            DIRECTOR_SYSTEM,
            director_prompt(problem, problem_report.solution_strings),
        )
        return LessonDraft.model_validate(payload)

    async def _review(self, problem, draft):
        payload = await self.client.complete_json(
            REVIEWER_SYSTEM,
            reviewer_prompt(problem, draft),
        )
        return ReviewDecision.model_validate(payload)

    async def _revise(self, problem, draft, review):
        payload = await self.client.complete_json(
            REVISION_SYSTEM,
            revision_prompt(problem, draft, review),
        )
        return LessonDraft.model_validate(payload)
```

`_emit` accepts a synchronous or asynchronous callback by checking
`inspect.isawaitable(result)`. `_validate_draft` uses the exact rules below:

```python
    def _validate_draft(self, problem, draft):
        for step in draft.math_steps:
            self.math_engine.validate_step(step)
        self.math_engine.validate_problem(
            draft.transfer_item.problem_text,
            draft.transfer_item.expected_answer,
        )
        interactions = [
            moment.interaction
            for moment in draft.moments
            if moment.interaction is not None
        ]
        if not interactions:
            raise LessonQualityError("讲解没有设置学生互动")
        required_operations = {
            "factor": "factor",
            "quadratic_formula": "quadratic_formula",
            "complete_the_square": "complete_the_square",
        }
        expected_operation = required_operations.get(problem.required_method)
        if expected_operation and expected_operation not in {
            step.operation for step in draft.math_steps
        }:
            raise LessonQualityError("讲解没有真正使用指定方法")
```

- [ ] **Step 5: Implement deterministic compilation**

`LessonCompiler.compile` creates a UUID lesson ID, prepends an opening beat,
maps manuscript moments, and appends summary and transfer beats:

```python
from uuid import uuid4

from app.schemas import (
    BoardAction,
    Interaction,
    LessonMoment,
    RuntimeBeat,
    RuntimeLesson,
)


class LessonCompiler:
    def compile(self, problem, draft, validation_report):
        moments = [
            LessonMoment(
                purpose="进入问题",
                narration=draft.opening,
                board_actions=[
                    BoardAction(
                        type="write",
                        target="original_problem",
                        content=problem.problem_text,
                    )
                ],
            ),
            *draft.moments,
            LessonMoment(
                purpose="压缩方法",
                narration=draft.summary,
                board_actions=[
                    BoardAction(
                        type="write",
                        target="method_summary",
                        content=draft.summary,
                    )
                ],
            ),
            LessonMoment(
                purpose="完成近迁移",
                narration="现在换一道表面不同、结构相同的题。",
                layer="interaction",
                interaction=Interaction(
                    interaction_id="near-transfer",
                    kind="transfer",
                    prompt=draft.transfer_item.problem_text,
                    expected_answer=draft.transfer_item.expected_answer,
                    hints=[draft.transfer_item.method_signal],
                    explanation_after_correct="你已经识别并使用了同一方法结构。",
                ),
            ),
        ]
        beats = []
        for index, moment in enumerate(moments, start=1):
            next_id = (
                f"beat-{index + 1:03d}"
                if index < len(moments)
                else None
            )
            beats.append(
                RuntimeBeat(
                    beat_id=f"beat-{index:03d}",
                    purpose=moment.purpose,
                    narration=moment.narration,
                    board_actions=moment.board_actions,
                    layer=moment.layer,
                    interaction=moment.interaction,
                    next_beat_id=next_id,
                )
            )
        return RuntimeLesson(
            lesson_id=str(uuid4()),
            problem=problem,
            title=draft.title,
            learning_goal=draft.learning_goal,
            beats=beats,
            summary=draft.summary,
            transfer_item=draft.transfer_item,
            validation_report=validation_report,
        )
```

- [ ] **Step 6: Run generation and compiler tests**

Run: `pytest -q tests/test_generation.py tests/test_compiler.py` after activating `general`.

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/prompts.py app/compiler.py app/generation.py tests/test_generation.py tests/test_compiler.py
git commit -m "feat: generate and review whole lessons"
```

## Task 6: Add in-memory jobs, lesson APIs and answer evaluation

**Files:**
- Create: `app/store.py`
- Create: `app/api.py`
- Create: `tests/test_api.py`
- Modify: `app/main.py`

- [ ] **Step 1: Write failing API tests**

```python
# tests/test_api.py
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import RuntimeLesson


class FakeGenerator:
    async def generate(self, problem, on_stage=None):
        if on_stage:
            on_stage("正在编译课堂")
        return RuntimeLesson(
            lesson_id="lesson-1",
            problem=problem,
            title="测试课程",
            learning_goal="学会解方程",
            beats=[],
            summary="完成",
            transfer_item={
                "problem_text": "2x=4",
                "expected_answer": "x=2",
                "method_signal": "保持等式平衡",
            },
            validation_report={"math_status": "verified"},
        )


def test_health_does_not_expose_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    client = TestClient(create_app(generator=FakeGenerator()))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "must-not-leak" not in response.text


def test_generation_job_completes_and_returns_lesson():
    client = TestClient(create_app(generator=FakeGenerator()))
    response = client.post(
        "/api/lessons/generate",
        json={
            "problem_text": "2x+3=7",
            "reference_answer": "x=2",
            "lesson_length": "standard",
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "completed"
    lesson = client.get(f"/api/lessons/{job['lesson_id']}")
    assert lesson.status_code == 200


def test_expression_interaction_uses_math_equivalence():
    client = TestClient(create_app(generator=FakeGenerator()))
    response = client.post(
        "/api/interactions/evaluate",
        json={"kind": "expression", "answer": "x^2-6x+9", "expected": "(x-3)^2"},
    )
    assert response.json()["classification"] == "correct"
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_api.py` after activating `general`.

Expected: `create_app` or API modules are missing.

- [ ] **Step 3: Implement in-memory stores and routes**

`MemoryStore` owns two dictionaries under a lock:

```python
from threading import RLock
from uuid import uuid4


class MemoryStore:
    def __init__(self):
        self.jobs = {}
        self.lessons = {}
        self.lock = RLock()

    def create_job(self):
        job = GenerationJob(
            job_id=str(uuid4()),
            status="queued",
            stage="等待生成",
        )
        with self.lock:
            self.jobs[job.job_id] = job
        return job

    def update_job(self, job_id, **changes):
        with self.lock:
            job = self.jobs[job_id].model_copy(update=changes)
            self.jobs[job_id] = job
            return job

    def get_job(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def save_lesson(self, lesson):
        with self.lock:
            self.lessons[lesson.lesson_id] = lesson

    def get_lesson(self, lesson_id):
        with self.lock:
            return self.lessons.get(lesson_id)
```

`POST /api/lessons/generate` returns `202` and schedules an async FastAPI background task. The task updates stages in this order:

```text
正在理解题目
正在验证数学路线
正在设计完整讲解
正在进行整篇审稿
正在修订并编译课堂
已完成
```

Implement the job route and background task with:

```python
async def run_generation(job_id, problem, store, generator):
    store.update_job(job_id, status="running", stage="正在理解题目")

    def report_stage(stage):
        store.update_job(job_id, stage=stage)

    try:
        lesson = await generator.generate(problem, on_stage=report_stage)
        store.save_lesson(lesson)
        store.update_job(
            job_id,
            status="completed",
            stage="已完成",
            lesson_id=lesson.lesson_id,
        )
    except Exception as exc:
        store.update_job(
            job_id,
            status="failed",
            stage="生成失败",
            error=safe_generation_error(exc),
        )


@router.post("/lessons/generate", status_code=202)
async def generate_lesson(problem: ProblemInput, background_tasks: BackgroundTasks):
    job = store.create_job()
    background_tasks.add_task(
        run_generation,
        job.job_id,
        problem,
        store,
        generator,
    )
    return {"job_id": job.job_id}
```

Expression, choice and point-select evaluation use deterministic comparison:

```python
class InteractionSubmission(BaseModel):
    kind: Literal["point_select", "choice", "expression", "free_text", "transfer"]
    answer: str
    expected: str


@router.post("/interactions/evaluate")
async def evaluate_interaction(submission: InteractionSubmission):
    if submission.kind in {"choice", "point_select"}:
        correct = submission.answer.strip() == submission.expected.strip()
    elif submission.kind == "expression":
        correct = math_engine.expressions_equivalent(
            submission.answer,
            submission.expected,
        )
    elif submission.kind == "transfer":
        correct = math_engine.answers_equivalent(
            submission.answer,
            submission.expected,
        )
    else:
        return {
            "classification": "needs_review",
            "message": "首版暂不自动判错这类文字回答。",
        }
    return {
        "classification": "correct" if correct else "incorrect",
    }
```

Do not send free-text answers to the model in Task 6. Add that only after a
separate privacy and prompt-injection test; the v0.1 fallback is always
`needs_review`, never `incorrect`.

- [ ] **Step 4: Build the app factory**

`create_app(settings=None, generator=None)` mounts `/static` with
`StaticFiles(..., check_dir=False)`, includes the API router, and serves
`index.html` at `/` and `lesson.html` at `/lesson/{lesson_id}`. Production
construction creates `Settings`, `OpenAICompatibleClient`, `MathEngine`,
`LessonGenerationService`, and one shared `MemoryStore`.

- [ ] **Step 5: Run API tests**

Run: `pytest -q tests/test_api.py` after activating `general`.

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add app/store.py app/api.py app/main.py tests/test_api.py
git commit -m "feat: expose lesson generation APIs"
```

## Task 7: Build the generation page

**Files:**
- Create: `app/static/index.html`
- Create: `app/static/styles.css`
- Create: `app/static/generate.js`
- Create: `tests/test_static_pages.py`

- [ ] **Step 1: Write failing static-page tests**

```python
# tests/test_static_pages.py
from fastapi.testclient import TestClient

from app.main import create_app


def test_generation_page_has_only_authoring_controls():
    html = TestClient(create_app()).get("/").text
    assert 'name="problem_text"' in html
    assert 'name="reference_answer"' in html
    assert 'name="required_method"' in html
    assert "OPENAI_API_KEY" not in html
    assert "validation_report" not in html


def test_lesson_page_has_fullscreen_classroom_regions():
    html = TestClient(create_app()).get("/lesson/example").text
    assert 'id="board-stage"' in html
    assert 'id="interaction-stage"' in html
    assert 'id="layer-stage"' in html
    assert 'id="lesson-controls"' in html
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_static_pages.py` after activating `general`.

Expected: static files are missing.

- [ ] **Step 3: Implement the generation page**

The page contains one compact centered form, not a permanent left/right workspace. It shows:

- product title and one-sentence promise;
- question textarea;
- reference-answer input;
- optional method select;
- concise/standard selector;
- generate button;
- model connectivity badge;
- one progress area that replaces the form after submission.

`generate.js` posts the input, polls `/api/jobs/{job_id}` every 700 ms, updates the stage label, and redirects to `/lesson/{lesson_id}` on completion. On failure, it restores the form and shows the exact safe error message.

- [ ] **Step 4: Establish the visual system**

Use CSS variables and a distinctive classroom direction:

```css
:root {
  --ink: #132a24;
  --chalk: #eff7e8;
  --board: #173e34;
  --board-deep: #0d2b25;
  --focus: #f2c84b;
  --reason: #91c788;
  --danger: #e66a5c;
  --paper: #f2ecdf;
}
```

Use a restrained editorial Chinese type system, subtle board grain, large touch targets, and no purple gradients, dashboard cards, or generic chat bubbles.

- [ ] **Step 5: Run static-page tests**

Run: `pytest -q tests/test_static_pages.py` after activating `general`.

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/static/index.html app/static/styles.css app/static/generate.js tests/test_static_pages.py
git commit -m "feat: add focused lesson generation page"
```

## Task 8: Build the full-screen classroom runtime

**Files:**
- Create: `app/static/lesson.html`
- Create: `app/static/runtime-core.mjs`
- Create: `app/static/lesson.js`
- Create: `tests/runtime-core.test.mjs`

- [ ] **Step 1: Write failing runtime state tests**

```javascript
// tests/runtime-core.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { LessonRuntime } from "../app/static/runtime-core.mjs";

const beats = [
  { beat_id: "beat-001", layer: "base", board_actions: [], narration: "开始" },
  {
    beat_id: "beat-002",
    layer: "micro_explanation",
    board_actions: [{ type: "write", target: "square", content: "(x-3)²" }],
    narration: "展开解释",
  },
  { beat_id: "beat-003", layer: "base", board_actions: [], narration: "继续" },
];

test("temporary layer returns to the exact base-board snapshot", () => {
  const runtime = new LessonRuntime(beats);
  runtime.baseBoard.set("equation", "x²-6x=-5");
  runtime.next();
  runtime.pushLayer("micro_explanation");
  runtime.activeBoard.set("square", "(x-3)²");
  runtime.popLayer();
  assert.equal(runtime.baseBoard.get("equation"), "x²-6x=-5");
  assert.equal(runtime.baseBoard.has("square"), false);
});

test("incorrect answers reveal one hint and require retry", () => {
  const runtime = new LessonRuntime(beats);
  const result = runtime.recordAnswer({
    classification: "incorrect",
    hints: ["先求一次项系数的一半。"],
  });
  assert.equal(result.canContinue, false);
  assert.equal(result.hint, "先求一次项系数的一半。");
});
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
node --test tests/runtime-core.test.mjs
```

Expected: import fails because `runtime-core.mjs` does not exist.

- [ ] **Step 3: Implement the pure runtime state machine**

`LessonRuntime` owns:

- `beats`;
- `currentIndex`;
- `baseBoard` as a `Map`;
- `layerStack` containing cloned temporary Maps;
- `answers`;
- `hintLevel`.

It exposes:

```javascript
current()
next()
previous()
pushLayer(layerName)
popLayer()
recordAnswer(result)
get activeBoard()
```

`next()` refuses to advance while the current interaction has not been answered correctly. `popLayer()` discards the temporary board and restores the unchanged base board.

- [ ] **Step 4: Implement the classroom DOM**

`lesson.html` contains:

- fixed 16:10 classroom shell centered in the viewport;
- compact top rail with lesson title, progress and exit;
- dominant board stage;
- narration strip;
- hidden interaction stage that appears inside the teaching flow;
- layer stage that covers the board without destroying it;
- previous, replay, pause/continue controls.

`lesson.js` loads the lesson ID from the path, creates `LessonRuntime`, and interprets board actions:

```text
write       → create a semantic board object
transform   → replace content while retaining object identity
focus       → focus target and fade non-targets
annotate    → add circle, box, underline, arrow, bracket or label
compare     → create a temporary two-column comparison
mask        → cover target content
reveal      → remove target mask
fade        → reduce target emphasis
pause       → wait for explicit continue
clear       → clear the current temporary region
```

For `micro_explanation`, `comparison`, or `interaction` layers, call `pushLayer` before rendering. When the beat ends, call `popLayer`, restore the base board, and optionally write the Director-provided conclusion into the base board.

- [ ] **Step 5: Implement interaction presentation**

- `point_select`: selectable semantic board objects;
- `choice`: large touch options;
- `expression`: math text input;
- `free_text`: short explanation input;
- `transfer`: new problem shown only after the summary.

All submissions call `/api/interactions/evaluate`. Correct responses reveal `explanation_after_correct` and enable continue. Incorrect responses show one hint and keep continue disabled.

- [ ] **Step 6: Run runtime and page tests**

Run:

```bash
node --test tests/runtime-core.test.mjs
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_static_pages.py
```

Expected: Node tests and pytest tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/static/lesson.html app/static/runtime-core.mjs app/static/lesson.js tests/runtime-core.test.mjs
git commit -m "feat: add full-screen interactive classroom"
```

## Task 9: Add regression cases, live smoke script and operating guide

**Files:**
- Create: `tests/fixtures/demo_cases.json`
- Create: `scripts/smoke_live.py`
- Create: `README.md`
- Modify: `docs/superpowers/specs/2026-08-05-ai-math-explanation-demo-design.md`

- [ ] **Step 1: Add six regression cases**

`tests/fixtures/demo_cases.json` contains:

```json
[
  {"problem_text":"2x+3=7","reference_answer":"x=2","required_method":null},
  {"problem_text":"3(x-2)+4=10","reference_answer":"x=4","required_method":null},
  {"problem_text":"x^2-5x+6=0","reference_answer":"x=2 或 x=3","required_method":"factor"},
  {"problem_text":"2x^2-3x-2=0","reference_answer":"x=2 或 x=-1/2","required_method":"quadratic_formula"},
  {"problem_text":"x^2-6x+5=0","reference_answer":"x=1 或 x=5","required_method":"complete_the_square"},
  {"problem_text":"x^2+x-1=0","reference_answer":"x=(-1+sqrt(5))/2 或 x=(-1-sqrt(5))/2","required_method":"complete_the_square"}
]
```

Add a parametrized math-engine test that loads all six cases and verifies each reference answer.

- [ ] **Step 2: Add a real-endpoint smoke script**

`scripts/smoke_live.py`:

```python
import asyncio
import json

from app.config import Settings
from app.generation import LessonGenerationService
from app.llm_client import OpenAICompatibleClient
from app.math_engine import MathEngine
from app.schemas import ProblemInput


async def main():
    settings = Settings.from_env()
    if not settings.model_configured:
        raise SystemExit(
            "缺少环境变量：" + ", ".join(settings.missing_model_settings)
        )
    client = OpenAICompatibleClient(settings)
    try:
        lesson = await LessonGenerationService(
            client,
            MathEngine(),
        ).generate(
            ProblemInput(
                problem_text="用配方法解方程：x^2-6x+5=0",
                reference_answer="x=1 或 x=5",
                required_method="complete_the_square",
            )
        )
        print(json.dumps(
            {
                "lesson_id": lesson.lesson_id,
                "title": lesson.title,
                "beats": len(lesson.beats),
                "validation_report": lesson.validation_report,
            },
            ensure_ascii=False,
            indent=2,
        ))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

The script deliberately prints no prompt content, response body, or credentials.

- [ ] **Step 3: Write operating instructions**

`README.md` documents exact commands:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
python -m pip install -r requirements.txt
cp .env.example .env
set -a
source .env
set +a
python -m uvicorn app.main:app --reload
```

It includes:

- environment-variable descriptions;
- `pytest -q` and `node --test tests/runtime-core.test.mjs`;
- optional `python scripts/smoke_live.py`;
- supported question types;
- the full-screen classroom walkthrough;
- known v0.1 limitations;
- the evidence boundary between schema checks, symbolic math checks, visual verification and actual learning effectiveness.

Confirm the design specification records Python 3.9, matching the `general`
environment; this correction is already present in the working tree and is
committed with the implementation plan before Task 1 begins.

- [ ] **Step 4: Run the complete automated suite**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q
node --test tests/runtime-core.test.mjs
```

Expected: all pytest and Node tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/smoke_live.py tests/fixtures/demo_cases.json tests/test_math_engine.py docs/superpowers/specs/2026-08-05-ai-math-explanation-demo-design.md
git commit -m "docs: add demo operation and regression cases"
```

## Task 10: Run live and browser verification

**Files:**
- Modify only files implicated by failures.

- [ ] **Step 1: Start the service in the required environment**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
set -a
source .env
set +a
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Expected: Uvicorn serves `http://127.0.0.1:8765`.

- [ ] **Step 2: Verify the configured model live**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
set -a
source .env
set +a
python scripts/smoke_live.py
```

Expected: a lesson ID, title, beat count and `math_status: verified`. If credentials have not been provided, record live generation as unverified rather than substituting mock output.

- [ ] **Step 3: Verify the complete browser story**

At a 1280×800 viewport:

1. Open the generation page.
2. Enter a previously unseen supported equation and reference answer.
3. Confirm generation progress is visible.
4. Confirm the result replaces the authoring page with a full-screen classroom.
5. Verify board text is legible without scrolling.
6. Verify focus, annotation, masking and reveal.
7. Submit one wrong answer, receive a hint and remain blocked.
8. Submit the correct answer and continue.
9. Enter a temporary micro-explanation layer and return to the identical base board.
10. Complete the near-transfer item.
11. Check browser console and server logs for errors.

- [ ] **Step 4: Run final verification**

Run:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q
node --test tests/runtime-core.test.mjs
git diff --check
git status -sb
```

Expected: all tests pass, no whitespace errors, and only intentional changes remain.

- [ ] **Step 5: Commit verification fixes**

```bash
git add app tests scripts README.md requirements.txt .env.example .gitignore
git commit -m "fix: complete demo verification"
```

Only create this commit when browser or live verification required code changes. Otherwise leave the previous tested commits as the final history.
