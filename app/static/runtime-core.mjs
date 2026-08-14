const MIN_FALLBACK_MS = 2200;
const MAX_FALLBACK_MS = 9000;
const SAFE_VISUAL_TARGET = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const EMPHASIS_CLASSES = Object.freeze({
  highlight: "is-highlighted",
  underline: "is-underlined",
  red: "is-red-emphasis",
});


function cloneValue(value) {
  if (value === undefined || value === null) return value;
  return JSON.parse(JSON.stringify(value));
}


export function cloneBoard(board) {
  return new Map(
    [...board.entries()].map(([key, value]) => [key, cloneValue(value)]),
  );
}


function cloneProblem(problem) {
  return new Map(
    [...problem.entries()].map(([key, value]) => [key, cloneValue(value)]),
  );
}


export function emphasisClassName(style) {
  return EMPHASIS_CLASSES[style] || "";
}


export function emptyVisualState(problemTargetIds = []) {
  const problem = new Map();
  if (Array.isArray(problemTargetIds)) {
    for (const target of problemTargetIds) {
      if (
        typeof target === "string"
        && SAFE_VISUAL_TARGET.test(target)
        && !problem.has(target)
      ) {
        problem.set(target, null);
      }
    }
  }
  return { board: new Map(), problem };
}


function cloneVisualState(state) {
  const board = state?.board instanceof Map ? state.board : new Map();
  const problem = state?.problem instanceof Map ? state.problem : new Map();
  return {
    board: cloneBoard(board),
    problem: cloneProblem(problem),
  };
}


function activeEmphasis(action) {
  const style = action?.emphasis_style;
  if (!emphasisClassName(style)) return null;
  if (
    action.persistence !== undefined
    && action.persistence !== null
    && action.persistence !== "trace"
    && action.persistence !== "transient"
  ) {
    return null;
  }
  return {
    style,
    strength: "active",
    persistence: action.persistence || "transient",
  };
}


function settledEmphasis(emphasis) {
  if (emphasis?.persistence !== "trace") return null;
  return { ...emphasis, strength: "trace" };
}


function rejectUnsupportedProblemAction(type) {
  const runtimeProcess = globalThis.process;
  if (runtimeProcess && runtimeProcess.env?.NODE_ENV !== "production") {
    throw new Error(`Unsupported problem visual action: ${type}`);
  }
}


function clearBoardFocus(board) {
  for (const [key, value] of board.entries()) {
    if (value?.kind !== "object") continue;
    board.set(key, {
      ...value,
      focused: false,
      focusFaded: false,
    });
  }
}


function boardHasFocus(board) {
  return [...board.values()].some((value) => (
    value?.kind === "object"
    && (value.focused === true || value.focusFaded === true)
  ));
}


function settleBoardTarget(board, target) {
  const current = board.get(target);
  if (!current || current.kind !== "object" || !current.emphasis) return;
  const next = { ...current };
  const emphasis = settledEmphasis(current.emphasis);
  if (emphasis) next.emphasis = emphasis;
  else delete next.emphasis;
  board.set(target, next);
}


export function applySyncVisualAction(currentState, action) {
  const state = cloneVisualState(currentState);
  if (
    !action
    || typeof action.type !== "string"
    || typeof action.surface !== "string"
    || typeof action.target !== "string"
    || !SAFE_VISUAL_TARGET.test(action.target)
  ) {
    return state;
  }

  const { target } = action;
  if (action.surface === "problem") {
    if (!state.problem.has(target)) return state;
    if (
      !["emphasize", "focus", "fade", "clear_focus"].includes(action.type)
    ) {
      rejectUnsupportedProblemAction(action.type);
      return state;
    }
    if (action.type === "focus") {
      state.problem.set(target, {
        style: "highlight",
        strength: "active",
        persistence: "transient",
      });
    } else if (action.type === "emphasize") {
      const emphasis = activeEmphasis(action);
      if (emphasis) state.problem.set(target, emphasis);
    } else if (
      action.type === "fade"
      || action.type === "clear_focus"
    ) {
      state.problem.set(
        target,
        settledEmphasis(state.problem.get(target)),
      );
    }
    return state;
  }

  if (action.surface !== "board") return state;
  if (action.type === "write") {
    if (typeof action.content !== "string") return state;
    state.board = applyBoardAction(state.board, action);
    const emphasis = activeEmphasis(action);
    if (emphasis) {
      state.board.set(target, {
        ...state.board.get(target),
        emphasis,
      });
    }
    return state;
  }
  if (
    !state.board.has(target)
    || state.board.get(target)?.kind !== "object"
  ) {
    return state;
  }

  if (action.type === "transform") {
    if (typeof action.content !== "string") return state;
    state.board = applyBoardAction(state.board, action);
    const emphasis = activeEmphasis(action);
    if (emphasis) {
      state.board.set(target, {
        ...state.board.get(target),
        emphasis,
      });
    }
  } else if (action.type === "annotate") {
    if (!["underline", "arrow", "bracket", "label"].includes(action.annotation)) {
      return state;
    }
    state.board = applyBoardAction(state.board, action);
  } else if (action.type === "reveal") {
    state.board = applyBoardAction(state.board, action);
  } else if (action.type === "focus") {
    for (const [key, value] of state.board.entries()) {
      if (value?.kind !== "object") continue;
      state.board.set(key, {
        ...value,
        focused: key === target,
        focusFaded: key !== target,
      });
    }
  } else if (action.type === "emphasize") {
    const emphasis = activeEmphasis(action);
    if (!emphasis) return state;
    state.board.set(target, {
      ...state.board.get(target),
      emphasis,
    });
  } else if (action.type === "fade") {
    settleBoardTarget(state.board, target);
  } else if (action.type === "clear_focus") {
    const hadFocus = boardHasFocus(state.board);
    settleBoardTarget(state.board, target);
    if (hadFocus) clearBoardFocus(state.board);
  }
  return state;
}


export function fallbackDurationForNarration(narration = "") {
  const readableCharacters = String(narration).replace(/\s/g, "").length;
  return Math.min(
    MAX_FALLBACK_MS,
    Math.max(MIN_FALLBACK_MS, readableCharacters * 115),
  );
}


export function scheduleBoardActions(actions, durationMs, fallbackMs = 3600) {
  if (!Array.isArray(actions) || actions.length === 0) return [];
  const resolvedDuration = Number.isFinite(durationMs) && durationMs > 0
    ? durationMs
    : fallbackMs;
  const interval = resolvedDuration / actions.length;
  return actions.map((action, index) => ({
    action,
    atMs: Math.round(interval * index),
  }));
}


export function createBoundedSettlement({
  resolve,
  timeoutMs,
  setTimeoutImpl = globalThis.setTimeout,
  clearTimeoutImpl = globalThis.clearTimeout,
  cleanup = () => {},
}) {
  let settled = false;
  let timer = null;
  const settle = () => {
    if (settled) return false;
    settled = true;
    try {
      if (timer !== null) clearTimeoutImpl(timer);
    } catch {
      // Settlement must continue even if a host timer shim rejects cleanup.
    }
    try {
      cleanup();
    } catch {
      // Media cleanup is best-effort; callers must still be released.
    }
    resolve();
    return true;
  };
  timer = setTimeoutImpl(settle, timeoutMs);
  return settle;
}


function boardObject(board, target) {
  return board.get(target) || {
    kind: "object",
    target,
    content: "",
    source: null,
    focused: false,
    faded: false,
    masked: false,
    annotations: [],
  };
}


function visibleObjectCount(board) {
  return [...board.values()].filter((value) => (
    value?.kind === "object" && value.masked !== true
  )).length;
}


export function boardActionAnnouncement(board, action) {
  const writesContent = (
    action?.type === "write" || action?.type === "transform"
  );
  if (
    writesContent
    && typeof action?.content === "string"
    && action.content
  ) {
    return action.content;
  }
  if (!(board instanceof Map) || !action?.target) return "";
  const value = board.get(action.target);
  if (!value) return "";
  if (typeof action?.content === "string" && action.content) {
    return action.content;
  }
  return typeof value?.content === "string" ? value.content : "";
}


export function applyBoardAction(currentBoard, action) {
  const board = cloneBoard(currentBoard);
  if (!action || typeof action.type !== "string") return board;

  const target = action.target || "";
  switch (action.type) {
    case "write": {
      const current = boardObject(board, target);
      board.set(target, {
        ...current,
        kind: "object",
        target,
        content: action.content || "",
        source: action.source || current.source || null,
      });
      break;
    }
    case "transform": {
      const current = boardObject(board, target);
      board.set(target, {
        ...current,
        kind: "object",
        target,
        previousContent: current.content,
        content: action.content || "",
        source: action.source || current.source || null,
        transforming: true,
      });
      break;
    }
    case "focus": {
      if (!board.has(target)) break;
      for (const [key, value] of board.entries()) {
        if (value.kind !== "object") continue;
        board.set(key, {
          ...value,
          focused: key === target,
          faded: key !== target,
        });
      }
      break;
    }
    case "annotate": {
      const annotation = action.annotation || "highlight";
      if (
        !board.has(target)
        || (
          annotation === "arrow"
          && !board.has(action.relation_target || "")
        )
      ) break;
      const isUselessEnclosure = (
        (annotation === "circle" || annotation === "box")
        && visibleObjectCount(board) <= 1
      );
      if (isUselessEnclosure) break;
      const current = boardObject(board, target);
      board.set(target, {
        ...current,
        annotations: [
          ...(current.annotations || []),
          {
            type: annotation,
            content: action.content || "",
            relationTarget: action.relation_target || null,
          },
        ],
      });
      break;
    }
    case "compare": {
      const rightTarget = action.relation_target || "";
      if (!board.has(target) || !board.has(rightTarget)) break;
      const left = boardObject(board, target);
      const right = boardObject(board, rightTarget);
      board.set("__comparison__", {
        kind: "comparison",
        target,
        relationTarget: rightTarget,
        leftContent: left.content,
        rightContent: right.content,
      });
      break;
    }
    case "mask": {
      if (!board.has(target)) break;
      const current = boardObject(board, target);
      board.set(target, { ...current, masked: true });
      break;
    }
    case "reveal": {
      if (!board.has(target)) break;
      const current = boardObject(board, target);
      board.set(target, { ...current, masked: false, revealed: true });
      break;
    }
    case "fade": {
      if (!board.has(target)) break;
      const current = boardObject(board, target);
      board.set(target, { ...current, faded: true, focused: false });
      break;
    }
    case "pause": {
      board.set("__runtime__", { kind: "runtime", paused: true });
      break;
    }
    case "clear": {
      if (target) board.delete(target);
      else board.clear();
      break;
    }
    default:
      break;
  }
  return board;
}


export function classifyInteractionControl(interaction) {
  if (!interaction) return "none";
  if (interaction.kind === "choice") return "options";
  if (interaction.kind === "point_select") return "board";
  if (interaction.kind === "expression" || interaction.kind === "transfer") {
    return "math-input";
  }
  if (interaction.kind === "free_text") return "text-input";
  return "none";
}


export function isCurrentInteractionSubmission(submission, current) {
  return Boolean(
    submission
    && current
    && current.interactionSubmitting === true
    && current.interactionVisible === true
    && submission.beatToken === current.beatToken
    && submission.interactionId === current.interactionId
    && submission.sequence === current.sequence
  );
}


export function isNativeInteractiveTarget(target) {
  const interactiveTags = new Set([
    "BUTTON",
    "A",
    "INPUT",
    "TEXTAREA",
    "SELECT",
    "SUMMARY",
  ]);
  let node = target;
  while (node) {
    if (node.isContentEditable === true) return true;
    if (interactiveTags.has(String(node.tagName || "").toUpperCase())) {
      return true;
    }
    node = node.parentElement || null;
  }
  return false;
}


export function resolveInteractionPresentation({
  result = {},
  interaction = {},
  selectedOption = null,
  outcome = {},
} = {}) {
  const classification = result?.classification || outcome?.classification;
  if (classification === "needs_review") {
    return {
      message: result?.message
        || "思路已经记录，我们继续沿主线往下走。",
      audioUrl: null,
      advanceMode: "manual",
    };
  }

  const supportCues = Array.isArray(selectedOption?.support_cues)
    ? selectedOption.support_cues
    : [];
  if (supportCues.length > 0 && outcome?.canContinue === true) {
    const firstCue = supportCues[0];
    return {
      message: firstCue?.display_text || firstCue?.spoken_text || "继续看这一步。",
      audioUrl: null,
      supportCues,
      advanceMode: "automatic",
    };
  }

  if (result?.feedback) {
    return {
      message: result.feedback,
      audioUrl: result.feedback_audio_url || null,
      advanceMode: classification === "correct" ? "automatic" : "retry",
    };
  }

  if (classification !== "correct") {
    const hintIndex = Number.isInteger(outcome?.hintIndex)
      ? outcome.hintIndex
      : null;
    return {
      message: outcome?.hint
        ? `提示：${outcome.hint}`
        : "回到题目中的已知关系再试一次。",
      audioUrl: hintIndex === null
        ? null
        : (interaction?.hint_audio_urls?.[hintIndex] || null),
      advanceMode: "retry",
    };
  }

  return {
    message: interaction?.explanation_after_correct || "判断正确。",
    audioUrl: interaction?.correct_audio_url || null,
    advanceMode: "automatic",
  };
}


export async function runSupportCueSequence(
  supportCues = [],
  {
    applyActions = () => {},
    presentCue = () => {},
    playAudio = async () => {},
    complete = () => {},
  } = {},
) {
  const cues = Array.isArray(supportCues) ? supportCues : [];
  for (const cue of cues) {
    const leadActions = Array.isArray(cue?.lead_actions)
      ? cue.lead_actions
      : [];
    const startActions = Array.isArray(cue?.start_actions)
      ? cue.start_actions
      : [];
    const endActions = Array.isArray(cue?.end_actions)
      ? cue.end_actions
      : [];
    applyActions("lead", leadActions);
    presentCue(cue);
    applyActions("start", startActions);
    await playAudio(cue?.audio_url || null, cue?.spoken_text || "");
    applyActions("end", endActions);
  }
  complete();
}


export class LessonRuntime {
  constructor(beats = []) {
    this.beats = Array.isArray(beats) ? beats.slice() : [];
    this.currentIndex = 0;
    this.baseBoard = new Map();
    this.layerStack = [];
    this.answers = new Map();
    this.hintLevels = new Map();
    this.audioState = "idle";
  }

  current() {
    return this.beats[this.currentIndex] || null;
  }

  get activeBoard() {
    const activeLayer = this.layerStack[this.layerStack.length - 1];
    return activeLayer ? activeLayer.board : this.baseBoard;
  }

  apply(action) {
    const nextBoard = applyBoardAction(this.activeBoard, action);
    if (this.layerStack.length > 0) {
      this.layerStack[this.layerStack.length - 1].board = nextBoard;
    } else {
      this.baseBoard = nextBoard;
    }
    return nextBoard;
  }

  next() {
    const beat = this.current();
    if (!beat || this.audioState === "playing") return false;
    if (beat.interaction && !this.interactionComplete()) return false;

    let nextIndex;
    if (beat.next_beat_id) {
      nextIndex = this.beats.findIndex(
        (candidate) => candidate.beat_id === beat.next_beat_id,
      );
    } else if (beat.next_beat_id === null) {
      return false;
    } else {
      nextIndex = this.currentIndex + 1;
    }
    if (nextIndex < 0 || nextIndex >= this.beats.length) return false;

    this.layerStack = [];
    this.currentIndex = nextIndex;
    this.audioState = "idle";
    return true;
  }

  previous() {
    if (this.currentIndex <= 0) return false;
    this.layerStack = [];
    this.currentIndex -= 1;
    this.audioState = "idle";
    return true;
  }

  pushLayer(layerName) {
    this.layerStack.push({
      name: layerName,
      board: cloneBoard(this.activeBoard),
    });
    return this.activeBoard;
  }

  popLayer() {
    if (this.layerStack.length === 0) return false;
    this.layerStack.pop();
    return true;
  }

  recordAnswer(result = {}) {
    const beat = this.current();
    const key = beat?.interaction?.interaction_id || beat?.beat_id || "unknown";
    const classification = result.classification || "incorrect";
    const canContinue = (
      classification === "correct"
      || classification === "needs_review"
      || beat?.interaction?.advance_after_response === true
    );
    if (canContinue) {
      this.answers.set(key, { classification, canContinue: true });
      return {
        classification,
        canContinue: true,
        hint: null,
        hintIndex: null,
      };
    }

    const hints = Array.isArray(result.hints) ? result.hints : [];
    const currentLevel = this.hintLevels.get(key) || 0;
    const hintIndex = hints.length
      ? Math.min(currentLevel, hints.length - 1)
      : null;
    const hint = hintIndex === null ? null : hints[hintIndex];
    this.hintLevels.set(key, currentLevel + 1);
    this.answers.set(key, { classification, canContinue: false });
    return {
      classification,
      canContinue: false,
      hint,
      hintIndex,
    };
  }

  interactionComplete() {
    const beat = this.current();
    if (!beat?.interaction) return true;
    const key = beat.interaction.interaction_id || beat.beat_id;
    return this.answers.get(key)?.canContinue === true;
  }

  requiresManualAdvance() {
    const actions = this.current()?.board_actions;
    return (
      Array.isArray(actions)
      && actions.some((action) => action?.type === "pause")
    );
  }

  completionDisposition() {
    if (this.audioState !== "ended") return "wait";
    if (this.current()?.interaction) return "interaction";
    if (this.requiresManualAdvance()) return "manual_advance";
    return "auto_advance";
  }

  canAutoAdvance() {
    return this.completionDisposition() === "auto_advance";
  }

  primaryControlIntent(isPaused) {
    if (this.audioState === "ended") {
      return this.requiresManualAdvance() ? "advance" : "idle";
    }
    return isPaused ? "resume" : "pause";
  }

  markAudioStarted() {
    this.audioState = "playing";
  }

  markAudioEnded() {
    this.audioState = "ended";
  }
}
