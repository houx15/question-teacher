const MIN_FALLBACK_MS = 2200;
const MAX_FALLBACK_MS = 9000;


function cloneValue(value) {
  if (value === undefined || value === null) return value;
  return JSON.parse(JSON.stringify(value));
}


export function cloneBoard(board) {
  return new Map(
    [...board.entries()].map(([key, value]) => [key, cloneValue(value)]),
  );
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
      for (const [key, value] of board.entries()) {
        if (value.kind !== "object") continue;
        board.set(key, {
          ...value,
          focused: key === target,
          faded: key !== target,
        });
      }
      if (!board.has(target)) {
        board.set(target, {
          ...boardObject(board, target),
          focused: true,
        });
      }
      break;
    }
    case "annotate": {
      const annotation = action.annotation || "highlight";
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
      const left = boardObject(board, target);
      const rightTarget = action.relation_target || "";
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
      const current = boardObject(board, target);
      board.set(target, { ...current, masked: true });
      break;
    }
    case "reveal": {
      const current = boardObject(board, target);
      board.set(target, { ...current, masked: false, revealed: true });
      break;
    }
    case "fade": {
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
      classification === "correct" || classification === "needs_review"
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
