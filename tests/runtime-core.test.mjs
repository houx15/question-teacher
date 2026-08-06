import test from "node:test";
import assert from "node:assert/strict";

import {
  LessonRuntime,
  applyBoardAction,
  classifyInteractionControl,
  cloneBoard,
  scheduleBoardActions,
} from "../app/static/runtime-core.mjs";


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


test("correct and needs-review answers unlock the interaction", () => {
  const runtime = new LessonRuntime([beats[2]]);

  assert.equal(
    runtime.recordAnswer({ classification: "needs_review", hints: [] })
      .canContinue,
    true,
  );
  assert.equal(runtime.interactionComplete(), true);
});


test("interaction kinds route to deterministic controls", () => {
  assert.equal(classifyInteractionControl({ kind: "choice" }), "options");
  assert.equal(classifyInteractionControl({ kind: "point_select" }), "board");
  assert.equal(classifyInteractionControl({ kind: "expression" }), "math-input");
  assert.equal(classifyInteractionControl({ kind: "transfer" }), "math-input");
  assert.equal(classifyInteractionControl({ kind: "free_text" }), "text-input");
  assert.equal(classifyInteractionControl(null), "none");
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
