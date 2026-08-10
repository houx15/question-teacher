import test from "node:test";
import assert from "node:assert/strict";


const loadGenerationUi = () => import("../app/static/generation-flow.mjs");


function createHarness({ responses = [], copyError = null } = {}) {
  const fetchCalls = [];
  const navigations = [];
  const views = [];
  const copies = [];
  const fetchImpl = async (url, options) => {
    fetchCalls.push({ url, options });
    const next = responses.shift();
    if (next instanceof Error) throw next;
    return next;
  };
  const view = {
    showCompletion(lessonId, lessonPath) {
      views.push({ type: "completion", lessonId, lessonPath });
    },
    setLookupPending(pending) {
      views.push({ type: "lookup-pending", pending });
    },
    setAuthoringLocked(locked) {
      views.push({ type: "authoring-locked", locked });
    },
    showLookupError(message) {
      views.push({ type: "lookup-error", message });
    },
    showCopyStatus(message, success) {
      views.push({ type: "copy-status", message, success });
    },
    restoreForm() {
      views.push({ type: "restore-form" });
    },
    selectCompletedLessonId() {
      views.push({ type: "select-id" });
    },
  };
  const clipboard = {
    async writeText(value) {
      copies.push(value);
      if (copyError) throw copyError;
    },
  };
  return {
    fetchCalls,
    navigations,
    views,
    copies,
    fetchImpl,
    view,
    clipboard,
    navigate(path) {
      navigations.push(path);
    },
  };
}


test("completed generation shows its saved ID without navigating", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  const harness = createHarness();
  const actions = createSavedLessonActions(harness);

  actions.showCompletion("lesson.safe:1");

  assert.deepEqual(harness.navigations, []);
  assert.deepEqual(harness.views, [{
    type: "completion",
    lessonId: "lesson.safe:1",
    lessonPath: "/lesson/lesson.safe%3A1",
  }]);
});


test("valid saved lesson is checked without cache before navigation", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  const harness = createHarness({ responses: [{ ok: true, status: 200 }] });
  const actions = createSavedLessonActions(harness);

  await actions.openExisting("  lesson.safe:1  ");

  assert.equal(harness.fetchCalls.length, 1);
  assert.equal(
    harness.fetchCalls[0].url,
    "/api/lessons/lesson.safe%3A1",
  );
  assert.equal(harness.fetchCalls[0].options.cache, "no-store");
  assert.equal(harness.fetchCalls[0].options.headers.Accept, "application/json");
  assert.deepEqual(harness.navigations, ["/lesson/lesson.safe%3A1"]);
  assert.deepEqual(
    harness.views.filter((event) => event.type === "lookup-pending"),
    [
      { type: "lookup-pending", pending: true },
      { type: "lookup-pending", pending: false },
    ],
  );
});


test("invalid lesson ID is rejected before a request", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  const harness = createHarness();
  const actions = createSavedLessonActions(harness);

  await actions.openExisting("../../secret");

  assert.deepEqual(harness.fetchCalls, []);
  assert.deepEqual(harness.navigations, []);
  assert.equal(
    harness.views.at(-1).message,
    "课程 ID 格式不正确，请检查后重试。",
  );
});


test("lesson ID validation mirrors the 128-character backend contract", async () => {
  const { isValidLessonId } = await loadGenerationUi();

  assert.equal(isValidLessonId(`a${"._:-".repeat(31)}xyz`), true);
  assert.equal(isValidLessonId(`a${"b".repeat(127)}`), true);
  assert.equal(isValidLessonId(`a${"b".repeat(128)}`), false);
  assert.equal(isValidLessonId(" leading-space"), false);
  assert.equal(isValidLessonId("contains/slash"), false);
});


test("missing and unavailable saved lessons have distinct messages", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  const harness = createHarness({
    responses: [
      { ok: false, status: 404 },
      { ok: false, status: 503 },
      new TypeError("offline"),
    ],
  });
  const actions = createSavedLessonActions(harness);

  await actions.openExisting("missing-1");
  await actions.openExisting("unavailable-1");
  await actions.openExisting("offline-1");

  assert.deepEqual(
    harness.views
      .filter((event) => event.type === "lookup-error")
      .map((event) => event.message),
    [
      "",
      "没有找到这个课程 ID。",
      "",
      "暂时无法读取课程，请稍后重试。",
      "",
      "暂时无法读取课程，请稍后重试。",
    ],
  );
  assert.deepEqual(harness.navigations, []);
});


test("only an exact 200 response opens a saved lesson", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  const harness = createHarness({ responses: [{ ok: true, status: 204 }] });
  const actions = createSavedLessonActions(harness);

  await actions.openExisting("lesson-204");

  assert.deepEqual(harness.navigations, []);
  assert.equal(
    harness.views.filter((event) => event.type === "lookup-error").at(-1).message,
    "暂时无法读取课程，请稍后重试。",
  );
});


test("stale lookup cannot navigate after a newer lookup", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  let resolveFirst;
  const first = new Promise((resolve) => {
    resolveFirst = resolve;
  });
  const harness = createHarness({
    responses: [first, { ok: true, status: 200 }],
  });
  const actions = createSavedLessonActions(harness);

  const staleLookup = actions.openExisting("old-lesson");
  await actions.openExisting("new-lesson");
  resolveFirst({ ok: true, status: 200 });
  await staleLookup;

  assert.deepEqual(harness.navigations, ["/lesson/new-lesson"]);
});


test("starting a new generation cancels a pending saved lesson lookup", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  let resolveLookup;
  const pendingLookup = new Promise((resolve) => {
    resolveLookup = resolve;
  });
  const harness = createHarness({ responses: [pendingLookup] });
  const actions = createSavedLessonActions(harness);

  const lookup = actions.openExisting("old-lesson");
  actions.cancelLookup();
  resolveLookup({ ok: true, status: 200 });
  await lookup;

  assert.deepEqual(harness.navigations, []);
  assert.deepEqual(
    harness.views.filter((event) => event.type === "lookup-pending"),
    [
      { type: "lookup-pending", pending: true },
      { type: "lookup-pending", pending: false },
    ],
  );
});


test("generation submission lock rejects lookups until authoring is restored", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  let acceptGeneration;
  const pendingGeneration = new Promise((resolve) => {
    acceptGeneration = resolve;
  });
  const harness = createHarness({ responses: [{ ok: true, status: 200 }] });
  const actions = createSavedLessonActions(harness);

  actions.lockForGeneration();
  const blockedLookup = actions.openExisting("old-lesson");
  acceptGeneration({ job_id: "new-job" });
  await pendingGeneration;
  actions.showCompletion("new-lesson");
  await blockedLookup;

  assert.deepEqual(harness.fetchCalls, []);
  assert.deepEqual(harness.navigations, []);
  assert.deepEqual(
    harness.views.filter((event) => event.type === "lookup-pending"),
    [{ type: "lookup-pending", pending: false }],
  );
  assert.deepEqual(
    harness.views.filter((event) => event.type === "authoring-locked"),
    [{ type: "authoring-locked", locked: true }],
  );

  actions.createAnother();
  await actions.openExisting("old-lesson");
  assert.deepEqual(harness.navigations, ["/lesson/old-lesson"]);
  assert.deepEqual(
    harness.views.filter((event) => event.type === "authoring-locked"),
    [
      { type: "authoring-locked", locked: true },
      { type: "authoring-locked", locked: false },
    ],
  );
});


test("copy gives confirmation and keeps a manual-copy fallback", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  const successHarness = createHarness();
  const successActions = createSavedLessonActions(successHarness);
  await successActions.copyLessonId("lesson-42");

  assert.deepEqual(successHarness.copies, ["lesson-42"]);
  assert.deepEqual(successHarness.views.at(-1), {
    type: "copy-status",
    message: "课程 ID 已复制。",
    success: true,
  });

  const failureHarness = createHarness({ copyError: new Error("blocked") });
  const failureActions = createSavedLessonActions(failureHarness);
  await failureActions.copyLessonId("lesson-42");

  assert.deepEqual(failureHarness.views.slice(-2), [
    { type: "select-id" },
    {
      type: "copy-status",
      message: "未能自动复制，请手动复制上方课程 ID。",
      success: false,
    },
  ]);
});


test("create another returns to the mutually exclusive authoring state", async () => {
  const { createSavedLessonActions } = await loadGenerationUi();
  const harness = createHarness();
  const actions = createSavedLessonActions(harness);

  actions.createAnother();

  assert.deepEqual(harness.views, [
    { type: "lookup-pending", pending: false },
    { type: "restore-form" },
  ]);
});
