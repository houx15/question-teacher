import {
  LessonRuntime,
  classifyInteractionControl,
  cloneBoard,
  fallbackDurationForNarration,
  scheduleBoardActions,
} from "./runtime-core.mjs";


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

let lesson = null;
let runtime = null;
let primaryAudio = null;
let feedbackAudio = null;
let timeline = null;
let beatToken = 0;
let started = false;
let paused = false;
let interactionVisible = false;
let appliedActionIndexes = new Set();
let beatSnapshots = new Map();

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
  dom.title.textContent = lesson.title;
  dom.startTitle.textContent = lesson.title;
  dom.goal.textContent = lesson.learning_goal;
  dom.problem.textContent = lesson.problem.problem_text;
  dom.progressTotal.textContent = String(lesson.beats.length);
  document.title = `${lesson.title} · 拾光讲题`;
  updateControls();
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
    if (type === "label") mark.textContent = annotation.content || "";
    if (type === "arrow") {
      mark.textContent = annotation.content
        || humanizeTarget(annotation.relationTarget);
    }
    container.append(mark);
  }
}


function renderComparison(region, value, registry) {
  let node = registry.get("__comparison__");
  if (!node) {
    node = document.createElement("article");
    node.className = "comparison-sheet";
    node.innerHTML = (
      "<div><small>观察 A</small><span></span></div>"
      + "<div><small>观察 B</small><span></span></div>"
    );
    registry.set("__comparison__", node);
    region.append(node);
  }
  const values = node.querySelectorAll("span");
  values[0].textContent = value.leftContent || humanizeTarget(value.target);
  values[1].textContent = (
    value.rightContent || humanizeTarget(value.relationTarget)
  );
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
    if (content.textContent !== value.content) {
      content.textContent = value.content || humanizeTarget(target);
      if (value.transforming) {
        node.classList.add("is-transforming");
        window.setTimeout(
          () => node.classList.remove("is-transforming"),
          620,
        );
      }
    }
    node.classList.toggle("is-focused", value.focused === true);
    node.classList.toggle("is-faded", value.faded === true);
    node.classList.toggle("is-masked", value.masked === true);
    node.classList.toggle("is-revealed", value.revealed === true);
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


function stopMedia() {
  timeline?.clear();
  timeline = null;
  if (primaryAudio) {
    primaryAudio.pause();
    primaryAudio.removeAttribute("src");
    primaryAudio.load();
    primaryAudio = null;
  }
  if (feedbackAudio) {
    feedbackAudio.pause();
    feedbackAudio = null;
  }
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
  dom.previous.disabled = runtime.currentIndex === 0 || !started;
  dom.replay.disabled = !started;
  dom.pause.disabled = !started || interactionVisible;
  dom.next.disabled = (
    !started
    || runtime.audioState === "playing"
    || (beat?.interaction && !runtime.interactionComplete())
    || beat?.next_beat_id === null
  );
  dom.progressCurrent.textContent = String(runtime.currentIndex + 1);
}


function setPaused(nextPaused) {
  if (!started || interactionVisible) return;
  paused = nextPaused;
  dom.pause.classList.toggle("is-paused", paused);
  dom.pauseLabel.textContent = paused ? "继续" : "暂停";
  dom.shell.classList.toggle("is-speaking", !paused);
  if (paused) {
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
  dom.announcer.textContent = action.content
    || `${action.type} ${humanizeTarget(action.target)}`;
  if (action.type === "pause") setPaused(true);
}


function scheduleActions(beat, durationMs, token) {
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


function beginFallbackPlayback(beat, token) {
  const duration = fallbackDurationForNarration(beat.narration);
  timeline?.clear();
  timeline = new PausableTimeline();
  runtime.markAudioStarted();
  dom.shell.classList.add("is-speaking");
  scheduleActions(beat, duration, token);
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


async function beginBeatPlayback(beat, token) {
  if (!beat.audio_url) {
    beginFallbackPlayback(beat, token);
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
    if (token === beatToken) beginFallbackPlayback(beat, token);
    return;
  }

  runtime.markAudioStarted();
  dom.shell.classList.add("is-speaking");
  timeline = new PausableTimeline();
  scheduleActions(
    beat,
    Number.isFinite(durationSeconds) ? durationSeconds * 1000 : Number.NaN,
    token,
  );
  audio.addEventListener("ended", () => finishBeat(token), { once: true });
  audio.addEventListener(
    "error",
    () => {
      if (token !== beatToken) return;
      beginFallbackPlayback(beat, token);
    },
    { once: true },
  );
  updateControls();
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
  clearPointSelection();

  if (!beatSnapshots.has(beat.beat_id)) {
    beatSnapshots.set(beat.beat_id, cloneBoard(runtime.baseBoard));
  }
  prepareBeatLayer(beat);
  dom.purpose.textContent = beat.purpose;
  dom.narration.textContent = beat.narration;
  dom.announcer.textContent = `第 ${runtime.currentIndex + 1} 段：${beat.purpose}`;
  updateControls();
  beginBeatPlayback(beat, token);
}


function finishBeat(token) {
  if (token !== beatToken) return;
  runtime.markAudioEnded();
  dom.shell.classList.remove("is-speaking");
  paused = false;
  updateControls();
  const beat = runtime.current();
  if (beat?.interaction) {
    showInteraction(beat.interaction);
    return;
  }
  if (beat?.board_actions?.some((action) => action.type === "pause")) {
    setPaused(true);
    dom.next.disabled = false;
    return;
  }
  timeline?.schedule(() => advanceBeat(), AUTO_ADVANCE_DELAY_MS);
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
  leaveTemporaryLayer();
  runtime.markAudioEnded();
  if (runtime.next()) {
    playCurrentBeat();
    return;
  }
  dom.purpose.textContent = "本课完成";
  dom.narration.textContent = "这条数学路线已经走完，可以重播任意一步。";
  dom.next.disabled = true;
  dom.pause.disabled = true;
  dom.shell.classList.remove("is-speaking");
}


function restoreSnapshotForCurrentBeat() {
  const beat = runtime.current();
  const snapshot = beatSnapshots.get(beat.beat_id);
  if (snapshot) runtime.baseBoard = cloneBoard(snapshot);
  runtime.layerStack = [];
  renderActiveBoards();
}


function replayCurrentBeat() {
  if (!runtime || !started) return;
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


function clearPointSelection() {
  for (const node of document.querySelectorAll(".board-object.is-selectable")) {
    node.classList.remove("is-selectable");
    node.removeAttribute("role");
    node.removeAttribute("aria-label");
    node.tabIndex = -1;
    node.onclick = null;
    node.onkeydown = null;
  }
}


function enablePointSelection(onSelect) {
  const nodes = [
    ...dom.baseBoard.querySelectorAll(".board-object"),
    ...dom.layerStage.querySelectorAll(".board-object"),
  ];
  nodes.forEach((node) => {
    node.classList.add("is-selectable");
    node.setAttribute("role", "button");
    node.setAttribute("aria-label", `选择 ${node.textContent.trim()}`);
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

  const card = element("div", "interaction-card");
  const kicker = element(
    "p",
    "interaction-kicker",
    interaction.kind === "transfer" ? "NEAR TRANSFER" : "PAUSE & THINK",
  );
  const heading = element("h2", "", interaction.prompt);
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

  const controlType = classifyInteractionControl(interaction);
  if (controlType === "options") {
    const optionGrid = element("div", "interaction-options");
    for (const option of interaction.options || []) {
      const button = element("button", "interaction-option", option.label);
      button.type = "button";
      button.addEventListener(
        "click",
        () => submitInteraction(interaction, option.option_id, {
          controls,
          feedback,
          hint,
          continueButton,
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
      submitInteraction(interaction, answer, {
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
      submitInteraction(interaction, answer, {
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
  const response = await fetch("/api/interactions/evaluate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      lesson_id: lesson.lesson_id,
      interaction_id: interaction.interaction_id,
      answer,
    }),
  });
  if (!response.ok) throw new Error("evaluation unavailable");
  return response.json();
}


function playFeedbackAudio(url) {
  return new Promise((resolve) => {
    if (!url) {
      window.setTimeout(resolve, 650);
      return;
    }
    feedbackAudio?.pause();
    feedbackAudio = new Audio(url);
    feedbackAudio.addEventListener("ended", resolve, { once: true });
    feedbackAudio.addEventListener("error", resolve, { once: true });
    feedbackAudio.play().catch(resolve);
  });
}


async function submitInteraction(interaction, answer, ui) {
  const interactiveNodes = ui.controls.querySelectorAll("button, input");
  interactiveNodes.forEach((node) => { node.disabled = true; });
  ui.feedback.classList.remove("is-wrong");
  ui.feedback.textContent = "正在核对…";

  try {
    const result = await evaluateInteraction(interaction, answer);
    const outcome = runtime.recordAnswer({
      classification: result.classification,
      hints: interaction.hints || [],
    });
    if (!outcome.canContinue) {
      ui.feedback.textContent = "这一步还可以再想一想。";
      ui.feedback.classList.add("is-wrong");
      ui.hint.textContent = outcome.hint
        ? `提示：${outcome.hint}`
        : "回到题目中的已知关系再试一次。";
      const hintAudio = outcome.hintIndex === null
        ? null
        : interaction.hint_audio_urls?.[outcome.hintIndex];
      await playFeedbackAudio(hintAudio);
      interactiveNodes.forEach((node) => { node.disabled = false; });
      ui.controls.querySelector("input, button")?.focus();
      return;
    }

    ui.feedback.textContent = result.classification === "needs_review"
      ? (result.message || "思路已经记录，我们继续沿主线往下走。")
      : (interaction.explanation_after_correct || "判断正确。");
    ui.hint.textContent = "";
    ui.continueButton.hidden = false;
    clearPointSelection();
    updateControls();
    const answeredBeatToken = beatToken;
    await playFeedbackAudio(interaction.correct_audio_url);
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
    ui.feedback.textContent = "暂时无法核对答案，请再提交一次。";
    ui.feedback.classList.add("is-wrong");
    interactiveNodes.forEach((node) => { node.disabled = false; });
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
dom.pause.addEventListener("click", () => setPaused(!paused));
dom.next.addEventListener("click", advanceBeat);
dom.fullscreen.addEventListener("click", toggleFullscreen);

document.addEventListener("fullscreenchange", () => {
  dom.fullscreen.setAttribute(
    "aria-label",
    document.fullscreenElement ? "退出全屏" : "进入全屏",
  );
});

document.addEventListener("keydown", (event) => {
  const activeTag = document.activeElement?.tagName;
  if (activeTag === "INPUT" || activeTag === "TEXTAREA") return;
  if (event.key === " ") {
    event.preventDefault();
    setPaused(!paused);
  } else if (event.key === "ArrowLeft") {
    previousBeat();
  } else if (event.key === "ArrowRight") {
    if (!dom.next.disabled) advanceBeat();
  } else if (event.key.toLowerCase() === "r") {
    replayCurrentBeat();
  } else if (event.key.toLowerCase() === "f") {
    toggleFullscreen();
  }
});

window.addEventListener("beforeunload", stopMedia);
showClassroomState("loading");
fetchLesson();
