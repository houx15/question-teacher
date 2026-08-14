import test from "node:test";
import assert from "node:assert/strict";

import * as runtimeCore from "../app/static/runtime-core.mjs";


const {
  LessonRuntime,
  applyBoardAction,
  applySyncVisualAction,
  boardActionAnnouncement,
  classifyInteractionControl,
  cloneBoard,
  createBoundedSettlement,
  emphasisClassName,
  emptyVisualState,
  isCurrentInteractionSubmission,
  isNativeInteractiveTarget,
  resolveInteractionPresentation,
  runSupportCueSequence,
  scheduleBoardActions,
} = runtimeCore;


const beats = [
  {
    beat_id: "beat-001",
    next_beat_id: "beat-002",
    layer: "base",
    board_actions: [
      { type: "write", target: "equation", content: "x²-6x=-5" },
      { type: "focus", target: "linear_coefficient" },
    ],
    narration: "先看一次项系数负六。",
  },
  {
    beat_id: "beat-002",
    next_beat_id: "beat-003",
    layer: "micro_explanation",
    board_actions: [
      { type: "write", target: "square", content: "(x-3)²" },
    ],
    narration: "展开解释。",
  },
  {
    beat_id: "beat-003",
    next_beat_id: null,
    layer: "interaction",
    board_actions: [],
    narration: "轮到你判断。",
    interaction: {
      interaction_id: "check",
      kind: "choice",
      hints: ["先看乘积。", "再看和。"],
    },
  },
];


test("problem emphasis reducer exports the synchronized visual contract", () => {
  assert.equal(typeof emptyVisualState, "function");
  assert.equal(typeof applySyncVisualAction, "function");
  assert.equal(typeof emphasisClassName, "function");
});


test("problem emphasis stores active trace metadata only for known targets", () => {
  const initial = emptyVisualState(["problem-math-001"]);
  const result = applySyncVisualAction(initial, {
    surface: "problem",
    type: "emphasize",
    target: "problem-math-001",
    emphasis_style: "highlight",
    persistence: "trace",
  });

  assert.deepEqual(result.problem.get("problem-math-001"), {
    style: "highlight",
    strength: "active",
    persistence: "trace",
  });
  assert.equal(initial.problem.get("problem-math-001"), null);
  assert.notEqual(result.problem, initial.problem);
  assert.notEqual(result.board, initial.board);
});


test("problem emphasis treats nullable persistence as transient", () => {
  const initial = emptyVisualState(["problem-math-001"]);
  const result = applySyncVisualAction(initial, {
    surface: "problem",
    type: "emphasize",
    target: "problem-math-001",
    emphasis_style: "highlight",
    persistence: null,
  });

  assert.deepEqual(result.problem.get("problem-math-001"), {
    style: "highlight",
    strength: "active",
    persistence: "transient",
  });
});


test("selective trace lifecycle downgrades trace and removes transient emphasis", () => {
  const seeded = emptyVisualState([
    "problem-math-001",
    "problem-math-002",
  ]);
  const traced = applySyncVisualAction(seeded, {
    surface: "problem",
    type: "emphasize",
    target: "problem-math-001",
    emphasis_style: "underline",
    persistence: "trace",
  });
  const transient = applySyncVisualAction(traced, {
    surface: "problem",
    type: "emphasize",
    target: "problem-math-002",
    emphasis_style: "red",
  });
  const clearedTrace = applySyncVisualAction(transient, {
    surface: "problem",
    type: "clear_focus",
    target: "problem-math-001",
  });
  const fadedTransient = applySyncVisualAction(clearedTrace, {
    surface: "problem",
    type: "fade",
    target: "problem-math-002",
  });

  assert.deepEqual(fadedTransient.problem.get("problem-math-001"), {
    style: "underline",
    strength: "trace",
    persistence: "trace",
  });
  assert.equal(fadedTransient.problem.get("problem-math-002"), null);
});


test("problem emphasis ignores unknown targets and hostile styles", () => {
  const initial = emptyVisualState(["problem-math-001"]);
  const unknownTarget = applySyncVisualAction(initial, {
    surface: "problem",
    type: "emphasize",
    target: "problem-math-999",
    emphasis_style: "highlight",
  });
  const hostileStyle = applySyncVisualAction(unknownTarget, {
    surface: "problem",
    type: "emphasize",
    target: "problem-math-001",
    emphasis_style: "x; background:url(javascript:alert(1))",
  });

  assert.equal(hostileStyle.problem.has("problem-math-999"), false);
  assert.equal(hostileStyle.problem.get("problem-math-001"), null);
  assert.deepEqual(initial, hostileStyle);
});


test("problem emphasis rejects a mutation type even with an allowed style", () => {
  const initial = emptyVisualState(["problem-math-001"]);

  assert.throws(
    () => applySyncVisualAction(initial, {
      surface: "problem",
      type: "write",
      target: "problem-math-001",
      content: "x = 1",
      emphasis_style: "highlight",
    }),
    /Unsupported problem visual action: write/,
  );
  assert.equal(initial.problem.get("problem-math-001"), null);
});


test("selective trace board emphasis stays separate from unchanged content", () => {
  let state = emptyVisualState();
  state = applySyncVisualAction(state, {
    surface: "board",
    type: "write",
    target: "equation",
    content: String.raw`\(x^2-5x+6=0\)`,
  });
  state = applySyncVisualAction(state, {
    surface: "board",
    type: "write",
    target: "reason",
    content: "观察乘积与和",
  });
  const emphasized = applySyncVisualAction(state, {
    surface: "board",
    type: "emphasize",
    target: "equation",
    emphasis_style: "highlight",
    persistence: "trace",
  });
  const traced = applySyncVisualAction(emphasized, {
    surface: "board",
    type: "fade",
    target: "equation",
  });

  assert.equal(
    traced.board.get("equation").content,
    String.raw`\(x^2-5x+6=0\)`,
  );
  assert.deepEqual(traced.board.get("equation").emphasis, {
    style: "highlight",
    strength: "trace",
    persistence: "trace",
  });
  assert.equal("emphasis" in traced.board.get("reason"), false);
  assert.deepEqual(state.board.get("equation").emphasis, undefined);
});


test("selective trace board write and transform accept explicit safe emphasis", () => {
  let state = emptyVisualState();
  state = applySyncVisualAction(state, {
    surface: "board",
    type: "write",
    target: "equation",
    content: "x²-5x+6=0",
    emphasis_style: "highlight",
    persistence: "trace",
  });
  const transformed = applySyncVisualAction(state, {
    surface: "board",
    type: "transform",
    target: "equation",
    content: "(x-2)(x-3)=0",
    emphasis_style: "underline",
  });

  assert.deepEqual(state.board.get("equation").emphasis, {
    style: "highlight",
    strength: "active",
    persistence: "trace",
  });
  assert.deepEqual(transformed.board.get("equation").emphasis, {
    style: "underline",
    strength: "active",
    persistence: "transient",
  });
});


test("board write and emphasize treat nullable persistence as transient", () => {
  const written = applySyncVisualAction(emptyVisualState(), {
    surface: "board",
    type: "write",
    target: "equation",
    content: "x²-5x+6=0",
    emphasis_style: "highlight",
    persistence: null,
  });
  const emphasized = applySyncVisualAction(written, {
    surface: "board",
    type: "emphasize",
    target: "equation",
    emphasis_style: "underline",
    persistence: null,
  });

  assert.deepEqual(written.board.get("equation").emphasis, {
    style: "highlight",
    strength: "active",
    persistence: "transient",
  });
  assert.deepEqual(emphasized.board.get("equation").emphasis, {
    style: "underline",
    strength: "active",
    persistence: "transient",
  });
});


test("emphasis rejects non-null unsupported persistence strings", () => {
  const problem = applySyncVisualAction(
    emptyVisualState(["problem-math-001"]),
    {
      surface: "problem",
      type: "emphasize",
      target: "problem-math-001",
      emphasis_style: "highlight",
      persistence: "sticky",
    },
  );
  const written = applySyncVisualAction(emptyVisualState(), {
    surface: "board",
    type: "write",
    target: "written-equation",
    content: "x²-5x+6=0",
    emphasis_style: "highlight",
    persistence: "sticky",
  });
  let board = applySyncVisualAction(emptyVisualState(), {
    surface: "board",
    type: "write",
    target: "equation",
    content: "x²-5x+6=0",
  });
  board = applySyncVisualAction(board, {
    surface: "board",
    type: "emphasize",
    target: "equation",
    emphasis_style: "underline",
    persistence: "sticky",
  });

  assert.equal(problem.problem.get("problem-math-001"), null);
  assert.equal("emphasis" in written.board.get("written-equation"), false);
  assert.equal("emphasis" in board.board.get("equation"), false);
});


test("selective trace board focus clears without deleting content", () => {
  let state = emptyVisualState();
  const ignored = applySyncVisualAction(state, {
    surface: "board",
    type: "focus",
    target: "missing",
  });
  assert.deepEqual(ignored, state);

  state = applySyncVisualAction(state, {
    surface: "board",
    type: "write",
    target: "left",
    content: "左边",
  });
  state = applySyncVisualAction(state, {
    surface: "board",
    type: "write",
    target: "right",
    content: "右边",
  });
  const focused = applySyncVisualAction(state, {
    surface: "board",
    type: "focus",
    target: "left",
  });
  const cleared = applySyncVisualAction(focused, {
    surface: "board",
    type: "clear_focus",
    target: "left",
  });

  assert.equal(focused.board.get("left").focused, true);
  assert.equal(focused.board.get("right").focusFaded, true);
  assert.equal(cleared.board.get("left").focused, false);
  assert.equal(cleared.board.get("right").focusFaded, false);
  assert.equal(cleared.board.get("left").content, "左边");
  assert.equal(cleared.board.get("right").content, "右边");
});


test("selective trace fade weakens only its named board target", () => {
  let state = emptyVisualState();
  for (const target of ["left", "right"]) {
    state = applySyncVisualAction(state, {
      surface: "board",
      type: "write",
      target,
      content: target,
    });
    state = applySyncVisualAction(state, {
      surface: "board",
      type: "emphasize",
      target,
      emphasis_style: "highlight",
      persistence: "trace",
    });
  }
  state = applySyncVisualAction(state, {
    surface: "board",
    type: "focus",
    target: "left",
  });
  const rightBefore = state.board.get("right");
  const faded = applySyncVisualAction(state, {
    surface: "board",
    type: "fade",
    target: "left",
  });

  assert.equal(faded.board.get("left").emphasis.strength, "trace");
  assert.deepEqual(faded.board.get("right"), rightBefore);
});


test("selective trace fade leaves an ordinary board object unchanged", () => {
  let state = emptyVisualState();
  state = applySyncVisualAction(state, {
    surface: "board",
    type: "write",
    target: "reason",
    content: "观察乘积与和",
  });
  const before = state.board.get("reason");
  const faded = applySyncVisualAction(state, {
    surface: "board",
    type: "fade",
    target: "reason",
  });

  assert.deepEqual(faded.board.get("reason"), before);
  assert.deepEqual(state.board.get("reason"), before);
});


test("selective trace transform preserves emphasis and input immutability", () => {
  let state = emptyVisualState();
  state = applySyncVisualAction(state, {
    surface: "board",
    type: "write",
    target: "equation",
    content: "x²-5x+6=0",
  });
  state = applySyncVisualAction(state, {
    surface: "board",
    type: "emphasize",
    target: "equation",
    emphasis_style: "red",
  });
  const before = state.board.get("equation");
  const transformed = applySyncVisualAction(state, {
    surface: "board",
    type: "transform",
    target: "equation",
    content: "(x-2)(x-3)=0",
  });

  assert.equal(before.content, "x²-5x+6=0");
  assert.equal(transformed.board.get("equation").content, "(x-2)(x-3)=0");
  assert.deepEqual(
    transformed.board.get("equation").emphasis,
    before.emphasis,
  );
  assert.notEqual(transformed.board.get("equation"), before);
  assert.notEqual(transformed.board, state.board);
  assert.notEqual(transformed.problem, state.problem);
});


test("problem emphasis class mapping is a closed allowlist", () => {
  assert.equal(emphasisClassName("highlight"), "is-highlighted");
  assert.equal(emphasisClassName("underline"), "is-underlined");
  assert.equal(emphasisClassName("red"), "is-red-emphasis");
  assert.equal(emphasisClassName("green"), "");
  assert.equal(emphasisClassName("x; color:red"), "");
  assert.equal(emphasisClassName(null), "");
});


test("scheduling distributes actions across actual audio duration", () => {
  const schedule = scheduleBoardActions(
    [{ type: "write" }, { type: "focus" }, { type: "annotate" }],
    9000,
  );

  assert.deepEqual(
    schedule.map((item) => item.atMs),
    [0, 3000, 6000],
  );
  assert.equal(schedule[1].action.type, "focus");
});


test("scheduling uses a bounded narration fallback without audio", () => {
  const schedule = scheduleBoardActions(
    [{ type: "write" }, { type: "focus" }],
    Number.NaN,
    4200,
  );

  assert.deepEqual(
    schedule.map((item) => item.atMs),
    [0, 2100],
  );
});


test("bounded settlement resolves once and cleans up event completion", () => {
  let timeoutCallback = null;
  let clearCount = 0;
  let cleanupCount = 0;
  let resolveCount = 0;
  const settle = createBoundedSettlement({
    resolve: () => { resolveCount += 1; },
    timeoutMs: 12000,
    setTimeoutImpl: (callback, timeoutMs) => {
      assert.equal(timeoutMs, 12000);
      timeoutCallback = callback;
      return 41;
    },
    clearTimeoutImpl: (timer) => {
      assert.equal(timer, 41);
      clearCount += 1;
    },
    cleanup: () => { cleanupCount += 1; },
  });

  assert.equal(settle(), true);
  assert.equal(settle(), false);
  timeoutCallback();
  assert.equal(resolveCount, 1);
  assert.equal(clearCount, 1);
  assert.equal(cleanupCount, 1);
});


test("bounded settlement timeout follows the same single finalizer", () => {
  let timeoutCallback = null;
  let cleaned = false;
  let resolved = false;
  const settle = createBoundedSettlement({
    resolve: () => { resolved = true; },
    timeoutMs: 7,
    setTimeoutImpl: (callback) => {
      timeoutCallback = callback;
      return 9;
    },
    clearTimeoutImpl: () => {},
    cleanup: () => { cleaned = true; },
  });

  timeoutCallback();
  assert.equal(resolved, true);
  assert.equal(cleaned, true);
  assert.equal(settle(), false);
});


test("bounded settlement still resolves if cleanup itself fails", () => {
  let resolved = false;
  const settle = createBoundedSettlement({
    resolve: () => { resolved = true; },
    timeoutMs: 12,
    setTimeoutImpl: () => 3,
    clearTimeoutImpl: () => {},
    cleanup: () => { throw new Error("cleanup failed"); },
  });

  assert.doesNotThrow(() => settle());
  assert.equal(resolved, true);
});


test("board reducer executes write transform focus and visibility actions", () => {
  let board = new Map();
  board = applyBoardAction(board, {
    type: "write",
    target: "equation",
    content: "x+1=2",
  });
  board = applyBoardAction(board, {
    type: "write",
    target: "reason",
    content: "等式两边相同操作",
  });
  board = applyBoardAction(board, {
    type: "transform",
    target: "equation",
    content: "x=1",
  });
  board = applyBoardAction(board, { type: "focus", target: "equation" });
  board = applyBoardAction(board, { type: "mask", target: "equation" });
  board = applyBoardAction(board, { type: "reveal", target: "equation" });
  board = applyBoardAction(board, { type: "fade", target: "reason" });

  assert.equal(board.get("equation").content, "x=1");
  assert.equal(board.get("equation").focused, true);
  assert.equal(board.get("equation").masked, false);
  assert.equal(board.get("reason").faded, true);
});


test("board reducer executes annotations comparisons pause and clear", () => {
  let board = new Map([
    ["left", { target: "left", content: "x²-6x", annotations: [] }],
    ["right", { target: "right", content: "(x-3)²-9", annotations: [] }],
  ]);
  board = applyBoardAction(board, {
    type: "annotate",
    target: "left",
    annotation: "underline",
  });
  board = applyBoardAction(board, {
    type: "compare",
    target: "left",
    relation_target: "right",
  });
  board = applyBoardAction(board, { type: "pause" });

  assert.equal(board.get("left").annotations[0].type, "underline");
  assert.equal(board.get("__comparison__").kind, "comparison");
  assert.equal(board.get("__runtime__").paused, true);

  board = applyBoardAction(board, { type: "clear" });
  assert.equal(board.size, 0);
});


test("board reducer ignores reference actions whose targets do not exist", () => {
  const initial = new Map([
    ["equation", {
      kind: "object",
      target: "equation",
      content: "x² - 6x + 5 = 0",
      annotations: [],
    }],
  ]);
  const invalidActions = [
    { type: "focus", target: "step1_eq_coeff_-6" },
    {
      type: "annotate",
      target: "step1_eq_coeff_-6",
      annotation: "underline",
    },
    { type: "mask", target: "step1_eq_coeff_-6" },
    { type: "reveal", target: "step1_eq_coeff_-6" },
    { type: "fade", target: "step1_eq_coeff_-6" },
    {
      type: "compare",
      target: "equation",
      relation_target: "step1_eq_coeff_-6",
    },
  ];

  for (const action of invalidActions) {
    const result = applyBoardAction(initial, action);
    assert.deepEqual(
      result,
      initial,
      `${action.type} must not create a phantom board object`,
    );
  }
});


test("arrow annotation requires both board objects to exist", () => {
  const initial = new Map([
    ["equation", {
      kind: "object",
      target: "equation",
      content: "x² - 6x + 5 = 0",
      annotations: [],
    }],
  ]);

  const result = applyBoardAction(initial, {
    type: "annotate",
    target: "equation",
    annotation: "arrow",
    relation_target: "step1_eq_coeff_-6",
  });

  assert.deepEqual(result, initial);
});


test("board action announcements use visible content and never target ids", () => {
  const board = new Map([
    ["equation", {
      kind: "object",
      target: "equation",
      content: "x² - 6x + 5 = 0",
      annotations: [],
    }],
  ]);

  assert.equal(
    boardActionAnnouncement(
      board,
      { type: "focus", target: "equation" },
    ),
    "x² - 6x + 5 = 0",
  );
  assert.equal(
    boardActionAnnouncement(
      board,
      { type: "focus", target: "step1_eq_coeff_-6" },
    ),
    "",
  );
  assert.equal(
    boardActionAnnouncement(
      board,
      {
        type: "annotate",
        target: "step1_eq_coeff_-6",
        annotation: "label",
        content: "一次项系数",
      },
    ),
    "",
  );
});


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
    [
      "left",
      {
        kind: "object",
        target: "left",
        content: "x+1",
        annotations: [],
      },
    ],
    [
      "right",
      {
        kind: "object",
        target: "right",
        content: "2",
        annotations: [],
      },
    ],
  ]);

  const result = applyBoardAction(initial, {
    type: "annotate",
    target: "left",
    annotation: "circle",
  });

  assert.equal(result.get("left").annotations[0].type, "circle");
});


test("temporary layer returns to the exact base-board snapshot", () => {
  const runtime = new LessonRuntime(beats);
  runtime.baseBoard.set("equation", {
    target: "equation",
    content: "x²-6x=-5",
    annotations: [],
  });
  const snapshot = cloneBoard(runtime.baseBoard);

  runtime.pushLayer("micro_explanation");
  runtime.apply({ type: "write", target: "square", content: "(x-3)²" });
  runtime.popLayer();

  assert.deepEqual(runtime.baseBoard, snapshot);
  assert.equal(runtime.baseBoard.has("square"), false);
});


test("navigation follows beat ids and is gated by audio and interaction", () => {
  const runtime = new LessonRuntime(beats);

  runtime.markAudioStarted();
  assert.equal(runtime.next(), false);
  runtime.markAudioEnded();
  assert.equal(runtime.next(), true);
  assert.equal(runtime.current().beat_id, "beat-002");
  assert.equal(runtime.next(), true);
  assert.equal(runtime.current().beat_id, "beat-003");
  assert.equal(runtime.next(), false);

  runtime.recordAnswer({ classification: "correct", hints: [] });
  assert.equal(runtime.next(), false);
  assert.equal(runtime.previous(), true);
  assert.equal(runtime.current().beat_id, "beat-002");
});


test("incorrect answers reveal progressive hints and require retry", () => {
  const runtime = new LessonRuntime([beats[2]]);

  const first = runtime.recordAnswer({
    classification: "incorrect",
    hints: beats[2].interaction.hints,
  });
  const second = runtime.recordAnswer({
    classification: "incorrect",
    hints: beats[2].interaction.hints,
  });

  assert.deepEqual(first, {
    classification: "incorrect",
    canContinue: false,
    hint: "先看乘积。",
    hintIndex: 0,
  });
  assert.equal(second.hint, "再看和。");
  assert.equal(second.hintIndex, 1);
});


test("adaptive wrong answers unlock only interactions that advance after response", () => {
  const adaptiveBeat = {
    ...beats[2],
    interaction: {
      ...beats[2].interaction,
      advance_after_response: true,
    },
  };
  const adaptive = new LessonRuntime([adaptiveBeat]);
  const legacy = new LessonRuntime([beats[2]]);

  assert.deepEqual(
    adaptive.recordAnswer({ classification: "incorrect", hints: ["提示"] }),
    {
      classification: "incorrect",
      canContinue: true,
      hint: null,
      hintIndex: null,
    },
  );
  assert.deepEqual(adaptive.answers.get(adaptiveBeat.interaction.interaction_id), {
    classification: "incorrect",
    canContinue: true,
  });
  assert.equal(legacy.recordAnswer({ classification: "incorrect" }).canContinue, false);
});


test("correct and needs-review answers unlock the interaction", () => {
  const runtime = new LessonRuntime([beats[2]]);

  assert.equal(
    runtime.recordAnswer({ classification: "needs_review", hints: [] })
      .canContinue,
    true,
  );
  assert.equal(runtime.interactionComplete(), true);
});


test("wrong transfer answers unlock navigation to the next beat", () => {
  const runtime = new LessonRuntime([
    {
      beat_id: "transfer-beat",
      next_beat_id: "after-transfer",
      interaction: {
        interaction_id: "transfer-check",
        kind: "transfer",
      },
    },
    { beat_id: "after-transfer", next_beat_id: null },
  ]);

  const outcome = runtime.recordAnswer({ classification: "incorrect" });

  assert.equal(outcome.canContinue, true);
  assert.equal(runtime.interactionComplete(), true);
  assert.equal(runtime.next(), true);
  assert.equal(runtime.current().beat_id, "after-transfer");
});


test("interaction kinds route to deterministic controls", () => {
  assert.equal(classifyInteractionControl({ kind: "choice" }), "options");
  assert.equal(classifyInteractionControl({ kind: "point_select" }), "board");
  assert.equal(classifyInteractionControl({ kind: "expression" }), "math-input");
  assert.equal(classifyInteractionControl({ kind: "transfer" }), "math-input");
  assert.equal(classifyInteractionControl({ kind: "free_text" }), "text-input");
  assert.equal(classifyInteractionControl(null), "none");
});


test("interaction submission identity rejects token id sequence and state drift", () => {
  const submission = {
    beatToken: 8,
    interactionId: "method-choice",
    sequence: 3,
  };
  const current = {
    beatToken: 8,
    interactionId: "method-choice",
    sequence: 3,
    interactionVisible: true,
    interactionSubmitting: true,
  };

  assert.equal(isCurrentInteractionSubmission(submission, current), true);
  assert.equal(
    isCurrentInteractionSubmission(submission, { ...current, beatToken: 9 }),
    false,
  );
  assert.equal(
    isCurrentInteractionSubmission(
      submission,
      { ...current, interactionId: "next-check" },
    ),
    false,
  );
  assert.equal(
    isCurrentInteractionSubmission(submission, { ...current, sequence: 4 }),
    false,
  );
  assert.equal(
    isCurrentInteractionSubmission(
      submission,
      { ...current, interactionVisible: false },
    ),
    false,
  );
  assert.equal(
    isCurrentInteractionSubmission(
      submission,
      { ...current, interactionSubmitting: false },
    ),
    false,
  );
});


test("native interactive target recognizes controls and contenteditable ancestry", () => {
  for (const tagName of [
    "BUTTON",
    "A",
    "INPUT",
    "TEXTAREA",
    "SELECT",
    "SUMMARY",
  ]) {
    assert.equal(isNativeInteractiveTarget({ tagName }), true);
  }
  const editableParent = { tagName: "DIV", isContentEditable: true };
  const editableChild = {
    tagName: "SPAN",
    isContentEditable: false,
    parentElement: editableParent,
  };

  assert.equal(isNativeInteractiveTarget(editableChild), true);
  assert.equal(isNativeInteractiveTarget({ tagName: "DIV" }), false);
  assert.equal(isNativeInteractiveTarget(null), false);
});


test("wrong diagnostic option presents its own feedback and audio", () => {
  const presentation = resolveInteractionPresentation({
    result: {
      classification: "incorrect",
      feedback: "这里应满足 \\(ab=5\\)，这组数的乘积不对。",
      feedback_audio_url: "/audio/option-b.mp3",
    },
    interaction: {
      hints: ["先看常数项。"],
      hint_audio_urls: ["/audio/hint-1.mp3"],
    },
    selectedOption: {
      option_id: "option-b",
      label: "错误选项",
    },
    outcome: { hint: "先看常数项。", hintIndex: 0 },
  });

  assert.deepEqual(presentation, {
    message: "这里应满足 \\(ab=5\\)，这组数的乘积不对。",
    audioUrl: "/audio/option-b.mp3",
    advanceMode: "retry",
  });
});


test("wrong adaptive option presents authored support and advances automatically", () => {
  const supportCues = [
    {
      cue_id: "support-1",
      display_text: "先回到根条件。",
      spoken_text: "先回到根条件。",
      lead_actions: [],
      start_actions: [],
      end_actions: [],
      audio_url: null,
    },
  ];
  const presentation = resolveInteractionPresentation({
    result: { classification: "incorrect", support_cues: supportCues },
    interaction: { advance_after_response: true },
    selectedOption: {
      option_id: "option-b",
      support_cues: [
        { cue_id: "untrusted", spoken_text: "不应采信。" },
      ],
    },
    outcome: { classification: "incorrect", canContinue: true },
  });

  assert.deepEqual(presentation, {
    message: "先回到根条件。",
    audioUrl: null,
    supportCues,
    advanceMode: "automatic",
  });
});


test("wrong legacy option still retries even if support-shaped data is present", () => {
  const presentation = resolveInteractionPresentation({
    result: { classification: "incorrect" },
    interaction: {
      advance_after_response: false,
      hints: ["返回原条件。"],
    },
    selectedOption: {
      option_id: "legacy-option",
      support_cues: [
        { cue_id: "ignored-support", spoken_text: "不应自动推进。" },
      ],
    },
    outcome: {
      classification: "incorrect",
      canContinue: false,
      hint: "返回原条件。",
      hintIndex: 0,
    },
  });

  assert.deepEqual(presentation, {
    message: "提示：返回原条件。",
    audioUrl: null,
    advanceMode: "retry",
  });
});


test("support cue sequence applies phases in order and completes once", async () => {
  const events = [];
  await runSupportCueSequence(
    [
      {
        cue_id: "support-1",
        spoken_text: "先解释原因。",
        lead_actions: [{ type: "focus", target: "condition" }],
        start_actions: [{ type: "write", target: "reason" }],
        end_actions: [{ type: "clear_focus", target: "condition" }],
        audio_url: "/audio/support-1.mp3",
      },
      {
        cue_id: "support-2",
        spoken_text: "再给出修正。",
        lead_actions: [],
        start_actions: [],
        end_actions: [],
        audio_url: null,
      },
    ],
    {
      applyActions: (phase, actions) => events.push(`${phase}:${actions.length}`),
      presentCue: (cue) => events.push(`present:${cue.cue_id}`),
      playAudio: async (url, spokenText) => events.push(`audio:${url}:${spokenText}`),
      complete: () => events.push("complete"),
    },
  );

  assert.deepEqual(events, [
    "lead:1",
    "present:support-1",
    "start:1",
    "audio:/audio/support-1.mp3:先解释原因。",
    "end:1",
    "lead:0",
    "present:support-2",
    "start:0",
    "audio:null:再给出修正。",
    "end:0",
    "complete",
  ]);
  assert.equal(events.filter((event) => event === "complete").length, 1);
});


test("wrong legacy response presents its staged hint and matching audio", () => {
  const presentation = resolveInteractionPresentation({
    result: { classification: "incorrect" },
    interaction: {
      hints: ["先看乘积。", "再看和。"],
      hint_audio_urls: ["/audio/hint-1.mp3", "/audio/hint-2.mp3"],
    },
    selectedOption: { option_id: "legacy-option", label: "旧选项" },
    outcome: { hint: "再看和。", hintIndex: 1 },
  });

  assert.deepEqual(presentation, {
    message: "提示：再看和。",
    audioUrl: "/audio/hint-2.mp3",
    advanceMode: "retry",
  });
});


test("correct diagnostic option presents its own feedback and audio", () => {
  const presentation = resolveInteractionPresentation({
    result: {
      classification: "correct",
      feedback: "对，\\(2+3=5\\) 且 \\(2\\times3=6\\)。",
      feedback_audio_url: "/audio/option-a.mp3",
    },
    interaction: {
      explanation_after_correct: "旧的正确解释。",
      correct_audio_url: "/audio/correct.mp3",
    },
    selectedOption: {
      option_id: "option-a",
      label: "正确选项",
    },
    outcome: { canContinue: true },
  });

  assert.deepEqual(presentation, {
    message: "对，\\(2+3=5\\) 且 \\(2\\times3=6\\)。",
    audioUrl: "/audio/option-a.mp3",
    advanceMode: "automatic",
  });
});


test("server feedback overrides any answer-revealing local option data", () => {
  const presentation = resolveInteractionPresentation({
    result: {
      classification: "incorrect",
      feedback: "只显示本次提交的服务端反馈。",
      feedback_audio_url: "/audio/server-selected.mp3",
    },
    selectedOption: {
      feedback: "不应信任的本地反馈。",
      feedback_audio_url: "/audio/client-leak.mp3",
    },
    outcome: { hint: "继续观察。", hintIndex: 0 },
  });

  assert.deepEqual(presentation, {
    message: "只显示本次提交的服务端反馈。",
    audioUrl: "/audio/server-selected.mp3",
    advanceMode: "retry",
  });
});


test("correct legacy response presents the shared explanation and audio", () => {
  const presentation = resolveInteractionPresentation({
    result: { classification: "correct" },
    interaction: {
      explanation_after_correct: "两边同时加 \\(3\\)，等式仍成立。",
      correct_audio_url: "/audio/correct.mp3",
    },
    selectedOption: null,
    outcome: { canContinue: true },
  });

  assert.deepEqual(presentation, {
    message: "两边同时加 \\(3\\)，等式仍成立。",
    audioUrl: "/audio/correct.mp3",
    advanceMode: "automatic",
  });
});


test("needs-review response preserves the server message and fallback", () => {
  const withMessage = resolveInteractionPresentation({
    result: { classification: "needs_review", message: "已记录你的思路。" },
    interaction: { correct_audio_url: "/audio/correct.mp3" },
    selectedOption: null,
    outcome: { canContinue: true },
  });
  const fallback = resolveInteractionPresentation({
    result: { classification: "needs_review" },
    interaction: {},
    selectedOption: null,
    outcome: { canContinue: true },
  });

  assert.deepEqual(withMessage, {
    message: "已记录你的思路。",
    audioUrl: null,
    advanceMode: "manual",
  });
  assert.equal(fallback.message, "思路已经记录，我们继续沿主线往下走。");
  assert.equal(fallback.audioUrl, null);
});


test("ended pause action makes primary control advance without replay", () => {
  const runtime = new LessonRuntime([
    {
      beat_id: "pause-beat",
      next_beat_id: "next-beat",
      board_actions: [{ type: "pause" }],
      narration: "先停在这里想一想。",
    },
    {
      beat_id: "next-beat",
      next_beat_id: null,
      board_actions: [],
      narration: "继续。",
    },
  ]);

  runtime.markAudioStarted();
  assert.equal(runtime.primaryControlIntent(false), "pause");
  runtime.markAudioEnded();

  assert.equal(runtime.requiresManualAdvance(), true);
  assert.equal(runtime.canAutoAdvance(), false);
  assert.equal(runtime.primaryControlIntent(false), "advance");
  assert.equal(runtime.next(), true);
  assert.equal(runtime.current().beat_id, "next-beat");
});


test("interaction takes priority over pause when beat audio ends", () => {
  const runtime = new LessonRuntime([
    {
      beat_id: "interactive-pause",
      next_beat_id: null,
      board_actions: [
        { type: "transform", target: "equation", content: "x=1" },
        { type: "pause" },
      ],
      narration: "现在轮到你回答。",
      interaction: {
        interaction_id: "answer-now",
        kind: "expression",
      },
    },
  ]);

  runtime.markAudioStarted();
  assert.equal(runtime.completionDisposition(), "wait");
  runtime.markAudioEnded();

  assert.equal(runtime.completionDisposition(), "interaction");
  assert.equal(runtime.canAutoAdvance(), false);
});
