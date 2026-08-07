import {
  LessonRuntime,
  applySyncVisualAction,
  boardActionAnnouncement,
  classifyInteractionControl,
  cloneBoard,
  createBoundedSettlement,
  emphasisClassName,
  emptyVisualState,
  fallbackDurationForNarration,
  isCurrentInteractionSubmission,
  isNativeInteractiveTarget,
  resolveInteractionPresentation,
  scheduleBoardActions,
} from "./runtime-core.mjs?v=20260807-2";
import { CuePlayer } from "./cue-player.mjs?v=20260807-2";
import {
  mathTextToPlainText,
  renderMathText,
  renderProblemMathText,
} from "./math-text.mjs?v=20260807-2";


const dom = {
  shell: document.querySelector("#classroom-shell"),
  title: document.querySelector("#lesson-title"),
  startTitle: document.querySelector("#start-title"),
  goal: document.querySelector("#learning-goal"),
  purpose: document.querySelector("#current-purpose"),
  problem: document.querySelector("#problem-text"),
  progressCurrent: document.querySelector("#progress-current"),
  progressTotal: document.querySelector("#progress-total"),
  baseBoard: document.querySelector("#base-board"),
  layerStage: document.querySelector("#layer-stage"),
  interactionStage: document.querySelector("#interaction-stage"),
  narration: document.querySelector("#narration-line"),
  loading: document.querySelector("#loading-state"),
  error: document.querySelector("#error-state"),
  errorMessage: document.querySelector("#error-message"),
  empty: document.querySelector("#empty-state"),
  startOverlay: document.querySelector("#start-overlay"),
  startButton: document.querySelector("#start-button"),
  previous: document.querySelector("#previous-button"),
  replay: document.querySelector("#replay-button"),
  pause: document.querySelector("#pause-button"),
  pauseLabel: document.querySelector("#pause-label"),
  next: document.querySelector("#next-button"),
  fullscreen: document.querySelector("#fullscreen-button"),
  announcer: document.querySelector("#runtime-announcer"),
};

const TEMPORARY_LAYERS = new Set([
  "micro_explanation",
  "comparison",
  "interaction",
]);
const AUTO_ADVANCE_DELAY_MS = 760;
const EVALUATION_TIMEOUT_MS = 14000;
const FEEDBACK_AUDIO_TIMEOUT_MS = 12000;
const EMPHASIS_CLASSES = [
  "is-highlighted",
  "is-underlined",
  "is-red-emphasis",
  "is-active",
  "is-trace",
];

let lesson = null;
let runtime = null;
let primaryAudio = null;
let feedbackAudio = null;
let feedbackAudioFinalizer = null;
let activeEvaluationController = null;
let timeline = null;
let cuePlayer = null;
let beatToken = 0;
let submissionSequence = 0;
let started = false;
let paused = false;
let interactionVisible = false;
let interactionSubmitting = false;
let appliedActionIndexes = new Set();
let beatSnapshots = new Map();
let visualState = emptyVisualState();

const boardRegistries = new WeakMap();


class PausableTimeline {
  constructor() {
    this.items = [];
    this.isPaused = false;
  }

  schedule(callback, delay) {
    const item = {
      callback,
      remaining: Math.max(0, delay),
      startedAt: performance.now(),
      timer: null,
      complete: false,
    };
    item.timer = window.setTimeout(() => this.run(item), item.remaining);
    this.items.push(item);
  }

  run(item) {
    if (item.complete || this.isPaused) return;
    item.complete = true;
    item.callback();
  }

  pause() {
    if (this.isPaused) return;
    this.isPaused = true;
    const now = performance.now();
    for (const item of this.items) {
      if (item.complete) continue;
      clearTimeout(item.timer);
      item.remaining = Math.max(0, item.remaining - (now - item.startedAt));
    }
  }

  resume() {
    if (!this.isPaused) return;
    this.isPaused = false;
    for (const item of this.items) {
      if (item.complete) continue;
      item.startedAt = performance.now();
      item.timer = window.setTimeout(() => this.run(item), item.remaining);
    }
  }

  clear() {
    for (const item of this.items) clearTimeout(item.timer);
    this.items = [];
    this.isPaused = false;
  }
}


function safeLessonId() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  if (parts.length !== 2 || parts[0] !== "lesson") return null;
  try {
    const value = decodeURIComponent(parts[1]);
    return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value)
      ? value
      : null;
  } catch {
    return null;
  }
}


function showClassroomState(state, message = "") {
  dom.loading.hidden = state !== "loading";
  dom.error.hidden = state !== "error";
  dom.empty.hidden = state !== "empty";
  if (message) dom.errorMessage.textContent = message;
  dom.shell.dataset.state = state;
}


async function fetchLesson() {
  const lessonId = safeLessonId();
  if (!lessonId) {
    showClassroomState("error", "课堂地址不完整，请返回后重新进入。");
    return;
  }
  try {
    const response = await fetch(
      `/api/lessons/${encodeURIComponent(lessonId)}`,
      { headers: { Accept: "application/json" }, cache: "no-store" },
    );
    if (!response.ok) {
      throw new Error(response.status === 404 ? "missing" : "request");
    }
    const payload = await response.json();
    if (!Array.isArray(payload.beats) || payload.beats.length === 0) {
      showClassroomState("empty");
      return;
    }
    lesson = payload;
    runtime = new LessonRuntime(payload.beats);
    hydrateLesson();
    showClassroomState("ready");
    dom.startOverlay.hidden = false;
    dom.startButton.focus();
  } catch (error) {
    showClassroomState(
      "error",
      error instanceof Error && error.message === "missing"
        ? "这节课不存在或已经失效。"
        : "网络暂时不可用，请稍后重新进入。",
    );
  }
}


function hydrateLesson() {
  const problemTargetIds = (lesson.problem_focus_targets || [])
    .map((target) => target.target_id);
  visualState = emptyVisualState(problemTargetIds);
  renderMathText(dom.title, lesson.title);
  renderMathText(dom.startTitle, lesson.title);
  renderMathText(dom.goal, lesson.learning_goal);
  renderProblemFocus();
  dom.progressTotal.textContent = String(lesson.beats.length);
  document.title = `${lesson.title} · 拾光讲题`;
  updateControls();
}


function renderProblemFocus() {
  renderProblemMathText(dom.problem, lesson.problem.problem_text, {
    focusTargets: lesson.problem_focus_targets || [],
    visualState: visualState.problem,
  });
}


function humanizeTarget(target) {
  return String(target || "")
    .replaceAll("_", " · ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}


function createBoardNode(target) {
  const node = document.createElement("article");
  node.className = "board-object is-new";
  node.tabIndex = -1;
  node.dataset.boardTarget = target;

  const selectablePrefix = document.createElement("span");
  selectablePrefix.className = "sr-only board-selectable-prefix";
  selectablePrefix.textContent = "选择板书：";
  selectablePrefix.setAttribute("aria-hidden", "true");
  node.append(selectablePrefix);

  const content = document.createElement("span");
  content.className = "board-content";
  node.append(content);

  const annotations = document.createElement("span");
  annotations.className = "board-annotations";
  annotations.setAttribute("aria-hidden", "true");
  node.append(annotations);
  return node;
}


function renderAnnotations(container, annotations = []) {
  container.replaceChildren();
  for (const annotation of annotations) {
    const mark = document.createElement("span");
    const type = annotation.type || "highlight";
    mark.className = `annotation annotation-${type}`;
    if (type === "label") renderMathText(mark, annotation.content || "");
    if (type === "arrow") {
      renderMathText(
        mark,
        annotation.content || humanizeTarget(annotation.relationTarget),
      );
    }
    container.append(mark);
  }
}


function renderComparison(region, value, registry) {
  let node = registry.get("__comparison__");
  if (!node) {
    node = document.createElement("article");
    node.className = "comparison-sheet";
    for (const label of ["观察 A", "观察 B"]) {
      const column = document.createElement("div");
      const heading = document.createElement("small");
      const content = document.createElement("span");
      heading.textContent = label;
      column.append(heading, content);
      node.append(column);
    }
    registry.set("__comparison__", node);
    region.append(node);
  }
  const values = node.querySelectorAll("span");
  const sources = [
    value.leftContent || humanizeTarget(value.target),
    value.rightContent || humanizeTarget(value.relationTarget),
  ];
  values.forEach((content, index) => {
    const source = sources[index];
    if (content.dataset.source === source) return;
    content.dataset.source = source;
    renderMathText(content, source);
  });
}


function renderBoard(board, region) {
  let registry = boardRegistries.get(region);
  if (!registry) {
    registry = new Map();
    boardRegistries.set(region, registry);
  }
  const visibleKeys = new Set();

  for (const [target, value] of board.entries()) {
    if (value.kind === "runtime") continue;
    visibleKeys.add(target);
    if (value.kind === "comparison") {
      renderComparison(region, value, registry);
      continue;
    }

    let node = registry.get(target);
    if (!node) {
      node = createBoardNode(target);
      registry.set(target, node);
      region.append(node);
      window.setTimeout(() => node.classList.remove("is-new"), 500);
    }
    const content = node.querySelector(".board-content");
    const source = value.content || humanizeTarget(target);
    if (content.dataset.source !== source) {
      content.dataset.source = source;
      renderMathText(content, source);
      if (value.transforming) {
        node.classList.add("is-transforming");
        window.setTimeout(
          () => node.classList.remove("is-transforming"),
          620,
        );
      }
    }
    node.classList.toggle("is-focused", value.focused === true);
    node.classList.toggle(
      "is-faded",
      value.faded === true || value.focusFaded === true,
    );
    node.classList.toggle("is-masked", value.masked === true);
    node.classList.toggle("is-revealed", value.revealed === true);
    node.classList.remove(...EMPHASIS_CLASSES);
    const emphasisClass = emphasisClassName(value.emphasis?.style);
    if (emphasisClass) {
      node.classList.add(emphasisClass);
      if (
        value.emphasis?.strength === "active"
        || value.emphasis?.strength === "trace"
      ) {
        node.classList.add(`is-${value.emphasis.strength}`);
      }
    }
    renderAnnotations(
      node.querySelector(".board-annotations"),
      value.annotations,
    );
  }

  for (const [target, node] of registry.entries()) {
    if (visibleKeys.has(target)) continue;
    node.remove();
    registry.delete(target);
  }
}


function clearBoardRegion(region) {
  region.replaceChildren();
  boardRegistries.set(region, new Map());
}


function renderActiveBoards() {
  renderBoard(runtime.baseBoard, dom.baseBoard);
  if (runtime.layerStack.length > 0) {
    dom.layerStage.hidden = false;
    renderBoard(runtime.activeBoard, dom.layerStage);
  } else {
    dom.layerStage.hidden = true;
    clearBoardRegion(dom.layerStage);
  }
}


function isCueBeat(beat) {
  return Array.isArray(beat?.sync_cues) && beat.sync_cues.length > 0;
}


function captureBeatSnapshot() {
  return {
    baseBoard: cloneBoard(runtime.baseBoard),
    problem: cloneBoard(visualState.problem),
  };
}


function restoreBeatSnapshot(snapshot) {
  if (!snapshot) return;
  runtime.baseBoard = cloneBoard(snapshot.baseBoard);
  visualState = {
    ...visualState,
    board: cloneBoard(snapshot.baseBoard),
    problem: cloneBoard(snapshot.problem),
  };
  runtime.layerStack = [];
  renderProblemFocus();
  renderActiveBoards();
}


function applyCueActions(actions) {
  let nextState = {
    board: runtime.activeBoard,
    problem: visualState.problem,
  };
  let announcement = "";
  for (const action of actions) {
    const reduced = applySyncVisualAction(nextState, action);
    if (action?.surface === "problem") {
      nextState = {
        board: nextState.board,
        problem: reduced.problem,
      };
    } else if (action?.surface === "board") {
      nextState = {
        board: reduced.board,
        problem: nextState.problem,
      };
      announcement = boardActionAnnouncement(nextState.board, action);
    }
  }
  if (runtime.layerStack.length > 0) {
    runtime.layerStack[runtime.layerStack.length - 1].board = nextState.board;
  } else {
    runtime.baseBoard = nextState.board;
  }
  visualState = {
    ...visualState,
    board: nextState.board,
    problem: nextState.problem,
  };
  renderProblemFocus();
  renderActiveBoards();
  if (announcement) dom.announcer.textContent = announcement;
}


cuePlayer = new CuePlayer({
  applyActions: applyCueActions,
  fallbackDuration: (cue) => fallbackDurationForNarration(
    cue?.spoken_text || "",
  ),
  onCueText: (spokenText) => {
    runtime.markAudioStarted();
    dom.shell.classList.add("is-speaking");
    renderMathText(dom.narration, spokenText);
    updateControls();
  },
  onBeatComplete: () => finishBeat(beatToken),
  onAudioUnavailable: () => {
    dom.announcer.textContent = "语音暂不可用，将按讲解节奏继续。";
  },
  restoreSnapshot: (snapshot) => {
    restoreBeatSnapshot(snapshot);
    prepareBeatLayer(runtime.current());
  },
});


function stopMedia() {
  submissionSequence += 1;
  interactionSubmitting = false;
  activeEvaluationController?.abort();
  activeEvaluationController = null;
  timeline?.clear();
  timeline = null;
  cuePlayer?.stop();
  if (primaryAudio) {
    primaryAudio.pause();
    primaryAudio.removeAttribute("src");
    primaryAudio.load();
    primaryAudio = null;
  }
  const activeFeedbackAudio = feedbackAudio;
  feedbackAudioFinalizer?.();
  activeFeedbackAudio?.pause();
  feedbackAudio = null;
  dom.shell.classList.remove("is-speaking");
}


function updateControls() {
  if (!runtime) {
    dom.previous.disabled = true;
    dom.replay.disabled = true;
    dom.pause.disabled = true;
    dom.next.disabled = true;
    return;
  }
  const beat = runtime.current();
  const primaryIntent = runtime.primaryControlIntent(paused);
  dom.previous.disabled = (
    runtime.currentIndex === 0 || !started || interactionSubmitting
  );
  dom.replay.disabled = !started || interactionSubmitting;
  dom.pause.disabled = (
    !started
    || interactionVisible
    || primaryIntent === "idle"
  );
  dom.pause.classList.toggle(
    "is-paused",
    primaryIntent === "resume" || primaryIntent === "advance",
  );
  dom.pauseLabel.textContent = primaryIntent === "advance"
    ? "继续"
    : (paused ? "继续" : "暂停");
  dom.next.disabled = (
    !started
    || interactionSubmitting
    || runtime.audioState === "playing"
    || (beat?.interaction && !runtime.interactionComplete())
    || beat?.next_beat_id === null
  );
  dom.progressCurrent.textContent = String(runtime.currentIndex + 1);
}


function setPaused(nextPaused) {
  if (
    !started
    || interactionVisible
    || runtime.audioState === "ended"
  ) return;
  paused = nextPaused;
  dom.pause.classList.toggle("is-paused", paused);
  dom.pauseLabel.textContent = paused ? "继续" : "暂停";
  dom.shell.classList.toggle("is-speaking", !paused);
  const beat = runtime.current();
  if (isCueBeat(beat) && paused) {
    cuePlayer.pause();
  } else if (isCueBeat(beat)) {
    cuePlayer.resume();
  } else if (paused) {
    primaryAudio?.pause();
    timeline?.pause();
  } else {
    if (primaryAudio?.src) {
      primaryAudio.play().catch(() => {});
    }
    timeline?.resume();
  }
}


function executeBoardAction(action, actionIndex, token) {
  if (token !== beatToken || appliedActionIndexes.has(actionIndex)) return;
  appliedActionIndexes.add(actionIndex);
  runtime.apply(action);
  renderActiveBoards();
  const announcement = boardActionAnnouncement(runtime.activeBoard, action);
  if (announcement) dom.announcer.textContent = announcement;
}


function scheduleLegacyActions(beat, durationMs, token) {
  const fallback = fallbackDurationForNarration(beat.narration);
  const schedule = scheduleBoardActions(
    beat.board_actions || [],
    durationMs,
    fallback,
  );
  schedule.forEach((item, index) => {
    timeline.schedule(
      () => executeBoardAction(item.action, index, token),
      item.atMs,
    );
  });
}


function beginLegacyFallbackPlayback(beat, token) {
  const duration = fallbackDurationForNarration(beat.narration);
  timeline?.clear();
  timeline = new PausableTimeline();
  runtime.markAudioStarted();
  dom.shell.classList.add("is-speaking");
  scheduleLegacyActions(beat, duration, token);
  timeline.schedule(() => finishBeat(token), duration);
  updateControls();
}


function waitForMetadata(audio) {
  if (Number.isFinite(audio.duration) && audio.duration > 0) {
    return Promise.resolve(audio.duration);
  }
  return new Promise((resolve) => {
    const timeout = window.setTimeout(() => done(Number.NaN), 1400);
    const done = (duration) => {
      clearTimeout(timeout);
      audio.removeEventListener("loadedmetadata", loaded);
      audio.removeEventListener("error", failed);
      resolve(duration);
    };
    const loaded = () => done(audio.duration);
    const failed = () => done(Number.NaN);
    audio.addEventListener("loadedmetadata", loaded, { once: true });
    audio.addEventListener("error", failed, { once: true });
    audio.load();
  });
}


async function beginLegacyBeatPlayback(beat, token) {
  if (!beat.audio_url) {
    beginLegacyFallbackPlayback(beat, token);
    return;
  }

  const audio = new Audio();
  primaryAudio = audio;
  audio.preload = "metadata";
  audio.src = beat.audio_url;
  const durationSeconds = await waitForMetadata(audio);
  if (token !== beatToken) return;

  try {
    await audio.play();
  } catch {
    if (token === beatToken) beginLegacyFallbackPlayback(beat, token);
    return;
  }

  runtime.markAudioStarted();
  dom.shell.classList.add("is-speaking");
  timeline = new PausableTimeline();
  scheduleLegacyActions(
    beat,
    Number.isFinite(durationSeconds) ? durationSeconds * 1000 : Number.NaN,
    token,
  );
  audio.addEventListener("ended", () => finishBeat(token), { once: true });
  audio.addEventListener(
    "error",
    () => {
      if (token !== beatToken) return;
      beginLegacyFallbackPlayback(beat, token);
    },
    { once: true },
  );
  updateControls();
}


function beginCueBeatPlayback(beat, snapshot) {
  cuePlayer.playBeat(beat, { snapshot });
}


function prepareBeatLayer(beat) {
  runtime.layerStack = [];
  clearBoardRegion(dom.layerStage);
  if (TEMPORARY_LAYERS.has(beat.layer)) {
    runtime.pushLayer(beat.layer);
    dom.layerStage.dataset.layer = beat.layer;
  }
  renderActiveBoards();
}


function playCurrentBeat() {
  const beat = runtime.current();
  if (!beat) {
    showClassroomState("empty");
    return;
  }
  stopMedia();
  beatToken += 1;
  const token = beatToken;
  appliedActionIndexes = new Set();
  paused = false;
  interactionVisible = false;
  dom.pause.classList.remove("is-paused");
  dom.pauseLabel.textContent = "暂停";
  dom.interactionStage.hidden = true;
  dom.interactionStage.classList.remove("is-point-select");
  clearPointSelection();

  if (!beatSnapshots.has(beat.beat_id)) {
    beatSnapshots.set(beat.beat_id, captureBeatSnapshot());
  }
  const snapshot = beatSnapshots.get(beat.beat_id);
  prepareBeatLayer(beat);
  renderMathText(dom.purpose, beat.purpose);
  dom.announcer.textContent = `第 ${runtime.currentIndex + 1} 段：${beat.purpose}`;
  updateControls();
  if (Array.isArray(beat.sync_cues) && beat.sync_cues.length > 0) {
    beginCueBeatPlayback(beat, snapshot);
  } else {
    renderMathText(dom.narration, beat.narration);
    beginLegacyBeatPlayback(beat, token);
  }
}


function finishBeat(token) {
  if (token !== beatToken) return;
  runtime.markAudioEnded();
  dom.shell.classList.remove("is-speaking");
  paused = false;
  updateControls();
  const beat = runtime.current();
  const disposition = runtime.completionDisposition();
  if (disposition === "interaction") {
    showInteraction(beat.interaction);
    return;
  }
  if (disposition === "manual_advance") {
    dom.next.disabled = false;
    updateControls();
    return;
  }
  if (disposition === "auto_advance") {
    timeline?.clear();
    timeline = new PausableTimeline();
    timeline.schedule(() => advanceBeat(), AUTO_ADVANCE_DELAY_MS);
  }
}


function leaveTemporaryLayer() {
  if (runtime.layerStack.length > 0) runtime.popLayer();
  clearBoardRegion(dom.layerStage);
  dom.layerStage.hidden = true;
}


function advanceBeat() {
  stopMedia();
  clearPointSelection();
  interactionVisible = false;
  dom.interactionStage.hidden = true;
  dom.interactionStage.classList.remove("is-point-select");
  leaveTemporaryLayer();
  runtime.markAudioEnded();
  if (runtime.next()) {
    playCurrentBeat();
    return;
  }
  renderMathText(dom.purpose, "本课完成");
  renderMathText(
    dom.narration,
    lesson.summary || "这条数学路线已经走完，可以重播任意一步。",
  );
  dom.next.disabled = true;
  dom.pause.disabled = true;
  dom.shell.classList.remove("is-speaking");
}


function restoreSnapshotForCurrentBeat() {
  const beat = runtime.current();
  const snapshot = beatSnapshots.get(beat.beat_id);
  if (snapshot) restoreBeatSnapshot(snapshot);
  runtime.layerStack = [];
  renderActiveBoards();
}


function replayCurrentBeat() {
  if (!runtime || !started) return;
  const beat = runtime.current();
  if (isCueBeat(beat)) {
    stopMedia();
    runtime.audioState = "idle";
    paused = false;
    interactionVisible = false;
    dom.pause.classList.remove("is-paused");
    dom.pauseLabel.textContent = "暂停";
    dom.interactionStage.hidden = true;
    dom.interactionStage.classList.remove("is-point-select");
    clearPointSelection();
    cuePlayer.replay();
    updateControls();
    return;
  }
  restoreSnapshotForCurrentBeat();
  runtime.audioState = "idle";
  playCurrentBeat();
}


function previousBeat() {
  if (!runtime || !started) return;
  stopMedia();
  leaveTemporaryLayer();
  if (!runtime.previous()) return;
  restoreSnapshotForCurrentBeat();
  playCurrentBeat();
}


function activatePrimaryControl() {
  if (!runtime || !started) return;
  const intent = runtime.primaryControlIntent(paused);
  if (intent === "advance") {
    advanceBeat();
  } else if (intent === "resume") {
    setPaused(false);
  } else if (intent === "pause") {
    setPaused(true);
  }
}


function clearPointSelection() {
  for (const node of document.querySelectorAll(".board-object.is-selectable")) {
    const prefix = node.querySelector(".board-selectable-prefix");
    prefix?.setAttribute("aria-hidden", "true");
    node.classList.remove("is-selectable");
    node.removeAttribute("role");
    node.tabIndex = -1;
    node.onclick = null;
    node.onkeydown = null;
  }
}


function enablePointSelection(onSelect) {
  const activeRegion = runtime.layerStack.length > 0
    ? dom.layerStage
    : dom.baseBoard;
  const nodes = [...activeRegion.querySelectorAll(".board-object")]
    .filter((node) => node.getClientRects().length > 0);
  nodes.forEach((node) => {
    const prefix = node.querySelector(".board-selectable-prefix");
    prefix?.removeAttribute("aria-hidden");
    node.classList.add("is-selectable");
    node.setAttribute("role", "button");
    node.tabIndex = 0;
    const select = () => onSelect(node.dataset.boardTarget);
    node.onclick = select;
    node.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    };
  });
  nodes[0]?.focus();
}


function element(tag, className, text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}


function showInteraction(interaction) {
  stopMedia();
  interactionVisible = true;
  dom.interactionStage.hidden = false;
  dom.interactionStage.replaceChildren();
  updateControls();
  const controlType = classifyInteractionControl(interaction);
  dom.interactionStage.classList.toggle("is-point-select", controlType === "board");

  const card = element("div", "interaction-card");
  const kicker = element(
    "p",
    "interaction-kicker",
    interaction.kind === "transfer" ? "NEAR TRANSFER" : "PAUSE & THINK",
  );
  const heading = element("h2", "");
  renderMathText(heading, interaction.prompt);
  const controls = element("div", "interaction-controls");
  const feedback = element("p", "interaction-feedback");
  feedback.setAttribute("aria-live", "polite");
  const hint = element("p", "interaction-hint");
  hint.setAttribute("aria-live", "polite");
  const continueButton = element("button", "interaction-continue", "继续讲解");
  continueButton.type = "button";
  continueButton.hidden = true;
  continueButton.addEventListener("click", advanceBeat);

  card.append(kicker, heading, controls, feedback, hint, continueButton);
  dom.interactionStage.append(card);

  if (controlType === "options") {
    const optionGrid = element("div", "interaction-options");
    for (const [optionIndex, option] of (interaction.options || []).entries()) {
      const button = element("button", "interaction-option");
      renderMathText(button, option.label);
      const accessibleLabel = mathTextToPlainText(option.label);
      button.setAttribute(
        "aria-label",
        accessibleLabel || `选项 ${optionIndex + 1}`,
      );
      button.type = "button";
      button.addEventListener(
        "click",
        () => submitInteraction(interaction, option.option_id, option, {
          controls, feedback, hint, continueButton,
        }),
      );
      optionGrid.append(button);
    }
    controls.append(optionGrid);
    optionGrid.querySelector("button")?.focus();
  } else if (controlType === "board") {
    controls.append(
      element("p", "point-instruction", "请直接点选黑板上的对应对象。"),
    );
    enablePointSelection((answer) => {
      clearPointSelection();
      submitInteraction(interaction, answer, null, {
        controls,
        feedback,
        hint,
        continueButton,
      });
    });
  } else {
    const form = element("form", "interaction-form");
    const input = element("input", "");
    input.type = "text";
    input.required = true;
    input.autocomplete = "off";
    input.inputMode = controlType === "math-input" ? "text" : "text";
    input.placeholder = controlType === "math-input"
      ? "在这里输入数学答案"
      : "用一句话写下你的想法";
    input.setAttribute("aria-label", input.placeholder);
    const submit = element("button", "interaction-submit", "提交");
    submit.type = "submit";
    form.append(input, submit);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const answer = input.value.trim();
      if (!answer) {
        input.focus();
        return;
      }
      submitInteraction(interaction, answer, null, {
        controls,
        feedback,
        hint,
        continueButton,
      });
    });
    controls.append(form);
    input.focus();
  }
}


async function evaluateInteraction(interaction, answer) {
  const controller = new AbortController();
  activeEvaluationController?.abort();
  activeEvaluationController = controller;
  const timeout = window.setTimeout(
    () => controller.abort(),
    EVALUATION_TIMEOUT_MS,
  );
  try {
    const response = await fetch("/api/interactions/evaluate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify({
        lesson_id: lesson.lesson_id,
        interaction_id: interaction.interaction_id,
        answer,
      }),
    });
    if (!response.ok) throw new Error("evaluation unavailable");
    return await response.json();
  } finally {
    clearTimeout(timeout);
    if (activeEvaluationController === controller) {
      activeEvaluationController = null;
    }
  }
}


function playFeedbackAudio(url) {
  const previousAudio = feedbackAudio;
  feedbackAudioFinalizer?.();
  previousAudio?.pause();
  feedbackAudio = null;
  if (!url) return Promise.resolve();

  return new Promise((resolve) => {
    const audio = new Audio(url);
    feedbackAudio = audio;
    let settle = null;
    const onEnded = () => settle();
    const onError = () => settle();
    settle = createBoundedSettlement({
      resolve,
      timeoutMs: FEEDBACK_AUDIO_TIMEOUT_MS,
      cleanup: () => {
        if (feedbackAudio === audio) feedbackAudio = null;
        if (feedbackAudioFinalizer === settle) {
          feedbackAudioFinalizer = null;
        }
        audio.removeEventListener("ended", onEnded);
        audio.removeEventListener("error", onError);
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
      },
    });
    feedbackAudioFinalizer = settle;
    audio.addEventListener("ended", onEnded, { once: true });
    audio.addEventListener("error", onError, { once: true });
    try {
      const playAttempt = audio.play();
      playAttempt?.catch(settle);
    } catch {
      settle();
    }
  });
}


async function submitInteraction(interaction, answer, selectedOption, ui) {
  submissionSequence += 1;
  const originatingBeatToken = beatToken;
  const originatingInteractionId = interaction.interaction_id;
  const originatingSequence = submissionSequence;
  interactionSubmitting = true;
  const isCurrentSubmission = () => isCurrentInteractionSubmission(
    {
      beatToken: originatingBeatToken,
      interactionId: originatingInteractionId,
      sequence: originatingSequence,
    },
    {
      beatToken,
      interactionId: runtime?.current()?.interaction?.interaction_id,
      sequence: submissionSequence,
      interactionVisible,
      interactionSubmitting,
    },
  );
  const interactiveNodes = ui.controls.querySelectorAll("button, input");
  interactiveNodes.forEach((node) => { node.disabled = true; });
  ui.feedback.classList.remove("is-wrong");
  ui.feedback.textContent = "正在核对…";
  updateControls();

  try {
    const result = await evaluateInteraction(interaction, answer);
    if (!isCurrentSubmission()) return;
    const outcome = runtime.recordAnswer({
      classification: result.classification,
      hints: interaction.hints || [],
    });
    const presentation = resolveInteractionPresentation({
      result,
      interaction,
      selectedOption,
      outcome,
    });
    if (!outcome.canContinue) {
      ui.feedback.classList.add("is-wrong");
      if (result?.feedback) {
        renderMathText(ui.feedback, presentation.message);
        renderMathText(ui.hint, "");
      } else {
        renderMathText(ui.feedback, "这一步还可以再想一想。");
        renderMathText(ui.hint, presentation.message);
      }
      await playFeedbackAudio(presentation.audioUrl);
      if (!isCurrentSubmission()) return;
      interactiveNodes.forEach((node) => { node.disabled = false; });
      if (interaction.kind === "point_select") {
        enablePointSelection((retryAnswer) => {
          clearPointSelection();
          submitInteraction(interaction, retryAnswer, null, ui);
        });
      }
      ui.controls.querySelector("input, button")?.focus();
      return;
    }

    renderMathText(ui.feedback, presentation.message);
    renderMathText(ui.hint, "");
    ui.continueButton.hidden = false;
    clearPointSelection();
    updateControls();
    if (presentation.advanceMode === "manual") {
      ui.continueButton.focus();
      return;
    }
    const answeredBeatToken = beatToken;
    await playFeedbackAudio(presentation.audioUrl);
    if (!isCurrentSubmission()) return;
    if (
      answeredBeatToken === beatToken
      && interactionVisible
      && runtime.interactionComplete()
    ) {
      window.setTimeout(() => {
        if (answeredBeatToken === beatToken && interactionVisible) {
          advanceBeat();
        }
      }, 520);
    }
  } catch {
    if (!isCurrentSubmission()) return;
    ui.feedback.textContent = "暂时无法核对答案，请再提交一次。";
    ui.feedback.classList.add("is-wrong");
    renderMathText(ui.hint, "");
    interactiveNodes.forEach((node) => { node.disabled = false; });
    if (interaction.kind === "point_select") {
      enablePointSelection((retryAnswer) => {
        clearPointSelection();
        submitInteraction(interaction, retryAnswer, null, ui);
      });
    }
    ui.controls.querySelector("input, button")?.focus();
  } finally {
    if (isCurrentSubmission()) {
      interactionSubmitting = false;
      updateControls();
    }
  }
}


async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await dom.shell.requestFullscreen();
    }
  } catch {
    dom.announcer.textContent = "当前浏览器未能进入全屏。";
  }
}


dom.startButton.addEventListener("click", () => {
  if (!runtime) return;
  started = true;
  dom.startOverlay.hidden = true;
  showClassroomState("ready");
  playCurrentBeat();
});
dom.previous.addEventListener("click", previousBeat);
dom.replay.addEventListener("click", replayCurrentBeat);
dom.pause.addEventListener("click", activatePrimaryControl);
dom.next.addEventListener("click", advanceBeat);
dom.fullscreen.addEventListener("click", toggleFullscreen);

document.addEventListener("fullscreenchange", () => {
  dom.fullscreen.setAttribute(
    "aria-label",
    document.fullscreenElement ? "退出全屏" : "进入全屏",
  );
});

document.addEventListener("keydown", (event) => {
  if (
    isNativeInteractiveTarget(event.target)
    || isNativeInteractiveTarget(document.activeElement)
  ) return;
  if (event.key === " ") {
    event.preventDefault();
    if (!dom.pause.disabled) activatePrimaryControl();
  } else if (event.key === "ArrowLeft") {
    if (!dom.previous.disabled) previousBeat();
  } else if (event.key === "ArrowRight") {
    if (!dom.next.disabled) advanceBeat();
  } else if (event.key.toLowerCase() === "r") {
    if (!dom.replay.disabled) replayCurrentBeat();
  } else if (event.key.toLowerCase() === "f") {
    toggleFullscreen();
  }
});

window.addEventListener("beforeunload", stopMedia);
showClassroomState("loading");
fetchLesson();
