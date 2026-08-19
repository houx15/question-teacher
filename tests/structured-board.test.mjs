import test from "node:test";
import assert from "node:assert/strict";

import {
  applyStructuredBoardAction,
  cloneStructuredBoard,
  emptyStructuredBoard,
  stepForScroll,
} from "../app/static/structured-board.mjs";


function action(type, overrides = {}) {
  return {
    surface: "board",
    type,
    target: "teaching-step-001",
    teaching_step_id: "teaching-step-001",
    ...overrides,
  };
}


function completedMainLine() {
  let state = emptyStructuredBoard();
  state = applyStructuredBoardAction(state, action("reveal_step_header", {
    step_label: "第一步：理解方程的根",
  }));
  state = applyStructuredBoardAction(state, action("write", {
    target: "board-root-meaning",
    content: "方程的根 → 代入后等式成立",
    board_role: "knowledge_anchor",
  }));
  return applyStructuredBoardAction(state, action("complete_step"));
}


test("step grows from questioning to active to completed", () => {
  const initial = emptyStructuredBoard();
  const active = applyStructuredBoardAction(
    initial,
    action("reveal_step_header", {
      step_label: "第一步：理解方程的根",
    }),
  );
  const written = applyStructuredBoardAction(active, action("write", {
    target: "board-root-meaning",
    content: "方程的根 → 代入后等式成立",
    board_role: "knowledge_anchor",
  }));
  const completed = applyStructuredBoardAction(
    written,
    action("complete_step"),
  );

  assert.equal(active.steps.get("teaching-step-001").status, "active");
  assert.equal(completed.steps.get("teaching-step-001").status, "completed");
  assert.equal(completed.steps.get("teaching-step-001").lines.length, 1);
  assert.equal(initial.steps.size, 0);
  assert.equal(active.steps.get("teaching-step-001").lines.length, 0);
});


test("method condition and result roles survive the structured reducer", () => {
  let state = applyStructuredBoardAction(
    emptyStructuredBoard(),
    action("reveal_step_header", { step_label: "第一步" }),
  );
  for (const [index, role] of ["method", "condition", "result"].entries()) {
    state = applyStructuredBoardAction(state, action("write", {
      target: `board-role-${index}`,
      content: `受控板书 ${role}`,
      board_role: role,
    }));
  }

  assert.deepEqual(
    state.steps.get("teaching-step-001").lines.map((line) => line.role),
    ["method", "condition", "result"],
  );
});


test("revealing the next step completes the previously active step", () => {
  let state = applyStructuredBoardAction(
    emptyStructuredBoard(),
    action("reveal_step_header", {step_label: "第一步"}),
  );
  state = applyStructuredBoardAction(
    state,
    action("reveal_step_header", {
      target: "teaching-step-002",
      teaching_step_id: "teaching-step-002",
      step_label: "第二步",
    }),
  );

  assert.equal(state.steps.get("teaching-step-001").status, "completed");
  assert.equal(state.steps.get("teaching-step-002").status, "active");
  assert.equal(state.activeStepId, "teaching-step-002");
});


test("a step created before its header remains in questioning state", () => {
  const state = applyStructuredBoardAction(
    emptyStructuredBoard(),
    action("scroll_to_step"),
  );

  assert.equal(
    state.steps.get("teaching-step-001").status,
    "questioning",
  );
  assert.equal(state.activeStepId, null);
  assert.equal(state.requestedScrollStepId, "teaching-step-001");
});


test("semantic targets update in place without reordering existing lines", () => {
  let state = completedMainLine();
  state = applyStructuredBoardAction(state, action("write", {
    target: "board-second",
    content: "第二行",
    board_role: "working",
  }));
  state = applyStructuredBoardAction(state, action("write", {
    target: "board-root-meaning",
    content: "把根代入原方程，等式仍然成立",
    board_role: "summary",
  }));
  state = applyStructuredBoardAction(state, action("transform", {
    target: "board-second",
    content: "更新后的第二行",
  }));

  const lines = state.steps.get("teaching-step-001").lines;
  assert.deepEqual(lines.map((line) => line.target), [
    "board-root-meaning",
    "board-second",
  ]);
  assert.equal(lines[0].content, "把根代入原方程，等式仍然成立");
  assert.equal(lines[0].role, "summary");
  assert.equal(lines[1].content, "更新后的第二行");
  assert.equal(lines[1].previousContent, "第二行");
});


test("support closes without deleting or rewriting the main step", () => {
  const state = completedMainLine();
  const opened = applyStructuredBoardAction(
    state,
    action("open_supporting_explanation", { target: "support-square" }),
  );
  const supported = applyStructuredBoardAction(opened, action("write", {
    target: "support-square-line",
    content: "先算括号整体的平方",
    board_role: "support",
  }));
  const repeated = applyStructuredBoardAction(supported, action("write", {
    target: "support-square-line",
    content: "平方要乘两次",
    board_role: "support",
  }));
  const closed = applyStructuredBoardAction(
    repeated,
    action("close_supporting_explanation", { target: "support-square" }),
  );

  assert.equal(opened.steps.get("teaching-step-001").status, "supporting");
  assert.equal(repeated.steps.get("teaching-step-001").support.lines.length, 1);
  assert.equal(
    repeated.steps.get("teaching-step-001").support.lines[0].content,
    "平方要乘两次",
  );
  assert.equal(closed.steps.get("teaching-step-001").support, null);
  assert.equal(closed.steps.get("teaching-step-001").status, "active");
  assert.equal(closed.steps.get("teaching-step-001").lines.length, 1);
  assert.deepEqual(state.steps.get("teaching-step-001").lines, [
    {
      target: "board-root-meaning",
      content: "方程的根 → 代入后等式成立",
      role: "knowledge_anchor",
      source: null,
    },
  ]);
});


test("support write without an open support region is a no-op clone", () => {
  const state = completedMainLine();
  const result = applyStructuredBoardAction(state, action("write", {
    target: "support-orphan",
    content: "不应进入主线",
    board_role: "support",
  }));

  assert.deepEqual(result, state);
  assert.notEqual(result, state);
  assert.notEqual(result.steps, state.steps);
  assert.notEqual(
    result.steps.get("teaching-step-001"),
    state.steps.get("teaching-step-001"),
  );
});


test("multiple steps preserve insertion order and expose the requested scroll step", () => {
  let state = completedMainLine();
  state = applyStructuredBoardAction(state, action("reveal_step_header", {
    target: "teaching-step-002",
    teaching_step_id: "teaching-step-002",
    step_label: "第二步：代入方程",
  }));
  state = applyStructuredBoardAction(state, action("scroll_to_step", {
    target: "teaching-step-002",
    teaching_step_id: "teaching-step-002",
  }));

  assert.deepEqual([...state.steps.keys()], [
    "teaching-step-001",
    "teaching-step-002",
  ]);
  assert.equal(state.activeStepId, "teaching-step-002");
  assert.equal(state.requestedScrollStepId, "teaching-step-002");
  assert.equal(stepForScroll(state).stepId, "teaching-step-002");
});


test("line emphasis uses data only and rejects model-supplied class names", () => {
  let state = completedMainLine();
  state = applyStructuredBoardAction(state, action("emphasize", {
    target: "board-root-meaning",
    emphasis_style: "highlight",
    persistence: "trace",
    className: "hostile-model-class",
  }));
  const faded = applyStructuredBoardAction(state, action("fade", {
    target: "board-root-meaning",
  }));
  const line = faded.steps.get("teaching-step-001").lines[0];

  assert.deepEqual(line.emphasis, {
    style: "highlight",
    strength: "trace",
    persistence: "trace",
  });
  assert.equal("className" in line, false);
});


test("unknown and malformed actions return unchanged deep clones", () => {
  const state = completedMainLine();
  const malformed = [
    null,
    {},
    action("unknown"),
    action("write", { content: "缺少角色", target: "line-missing-role" }),
    action("reveal_step_header", { step_label: "" }),
    action("complete_step", { target: "another-step" }),
    action("write", {
      target: "../hostile",
      content: "x",
      board_role: "working",
    }),
    action("write", {
      surface: "problem",
      target: "line-on-problem",
      content: "x",
      board_role: "working",
    }),
  ];

  for (const item of malformed) {
    const result = applyStructuredBoardAction(state, item);
    assert.deepEqual(result, state);
    assert.notEqual(result, state);
    assert.notEqual(result.steps, state.steps);
    assert.notEqual(
      result.steps.get("teaching-step-001"),
      state.steps.get("teaching-step-001"),
    );
    assert.notEqual(
      result.steps.get("teaching-step-001").lines,
      state.steps.get("teaching-step-001").lines,
    );
  }
});


test("reference actions cannot materialize an unknown semantic target", () => {
  const initial = emptyStructuredBoard();
  const missingStep = applyStructuredBoardAction(initial, action("transform", {
    target: "missing-line",
    content: "不能凭空出现",
  }));
  const state = completedMainLine();
  const missingLine = applyStructuredBoardAction(state, action("focus", {
    target: "missing-line",
  }));

  assert.deepEqual(missingStep, initial);
  assert.equal(missingStep.steps.size, 0);
  assert.deepEqual(missingLine, state);
  assert.notEqual(missingLine.steps, state.steps);
});


test("cloneStructuredBoard clones nested support and annotation data", () => {
  let state = completedMainLine();
  state = applyStructuredBoardAction(
    state,
    action("open_supporting_explanation", { target: "support-square" }),
  );
  state = applyStructuredBoardAction(state, action("write", {
    target: "support-square-line",
    content: "先算平方",
    board_role: "support",
  }));
  state = applyStructuredBoardAction(state, action("annotate", {
    target: "support-square-line",
    annotation: "underline",
  }));
  const cloned = cloneStructuredBoard(state);

  assert.deepEqual(cloned, state);
  assert.notEqual(cloned.steps, state.steps);
  assert.notEqual(
    cloned.steps.get("teaching-step-001").support,
    state.steps.get("teaching-step-001").support,
  );
  assert.notEqual(
    cloned.steps.get("teaching-step-001").support.lines[0].annotations,
    state.steps.get("teaching-step-001").support.lines[0].annotations,
  );
});
