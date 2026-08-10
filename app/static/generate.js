import { createSavedLessonActions } from "./generation-flow.mjs?v=20260810-1";

const form = document.querySelector("#lesson-form");
const savedLessonEntry = document.querySelector("#saved-lesson-entry");
const existingLessonForm = document.querySelector("#existing-lesson-form");
const existingLessonId = document.querySelector("#existing-lesson-id");
const existingLessonError = document.querySelector("#existing-lesson-error");
const progress = document.querySelector("#generation-progress");
const completion = document.querySelector("#generation-complete");
const completedLessonId = document.querySelector("#completed-lesson-id");
const copyLessonButton = document.querySelector("#copy-lesson-id");
const copyLessonStatus = document.querySelector("#copy-lesson-status");
const enterCompletedLesson = document.querySelector("#enter-completed-lesson");
const createAnotherButton = document.querySelector("#create-another-lesson");
const progressTitle = document.querySelector("#progress-title");
const progressDetail = document.querySelector("#progress-detail");
const progressSteps = [...document.querySelectorAll("#progress-steps li")];
const formError = document.querySelector("#form-error");
const returnButton = document.querySelector("#return-to-form");
const modelStatus = document.querySelector("#model-status");
const voiceStatus = document.querySelector("#voice-status");

const POLL_INTERVAL_MS = 700;
const STAGE_DETAILS = {
  "正在理解题目": "先确认题意与参考答案，找到这节课真正要解决的问题。",
  "正在核对题目材料": "正在整理题目、参考答案与解析，确认讲解依据和关键推理。",
  "正在设计完整讲解": "课堂导演正在组织观察、推导、互动与近迁移。",
  "正在进行整篇审稿": "教研审稿正在检查整节课能否让学生跟上并思考。",
  "正在修订并编译课堂": "把审稿意见落实到完整讲解，并编排板书节奏。",
  "正在生成讲解语音": "最后为每个短讲解片段生成同步语音。",
  "已完成": "课堂已经准备好。",
};

let activeJob = null;
let pollTimer = null;
let lookupPending = false;


function setServiceStatus(element, available, readyLabel, unavailableLabel) {
  element.dataset.state = available ? "ready" : "unavailable";
  element.lastChild.textContent = available ? readyLabel : unavailableLabel;
}


async function loadHealth() {
  try {
    const response = await fetch("/api/health", {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error("health unavailable");
    const health = await response.json();
    setServiceStatus(
      modelStatus,
      health.model_configured === true,
      "讲解模型已连接",
      "讲解模型未配置",
    );
    setServiceStatus(
      voiceStatus,
      health.voice_configured === true,
      "语音服务已连接",
      "语音服务未配置",
    );
  } catch {
    setServiceStatus(modelStatus, false, "", "讲解模型状态未知");
    setServiceStatus(voiceStatus, false, "", "语音服务状态未知");
  }
}


function setProgressStage(stage) {
  progressTitle.textContent = stage || "正在生成课堂";
  progressDetail.textContent = STAGE_DETAILS[stage]
    || "正在把题目整理成一节可播放的课堂。";
  const activeIndex = Math.max(
    0,
    progressSteps.findIndex((step) => step.dataset.stage === stage),
  );
  progressSteps.forEach((step, index) => {
    step.classList.toggle("is-complete", index < activeIndex);
    step.classList.toggle("is-active", index === activeIndex);
  });
}


function clearPolling() {
  window.clearTimeout(pollTimer);
  pollTimer = null;
}


function showProgress() {
  clearPolling();
  savedLessonEntry.hidden = true;
  form.hidden = true;
  completion.hidden = true;
  progress.hidden = false;
  returnButton.hidden = true;
  formError.hidden = true;
  setProgressStage("正在理解题目");
}


function restoreForm(message = "") {
  clearPolling();
  activeJob = null;
  progress.hidden = true;
  completion.hidden = true;
  savedLessonEntry.hidden = false;
  form.hidden = false;
  copyLessonStatus.textContent = "";
  formError.textContent = message;
  formError.hidden = !message;
  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = false;
  submitButton.querySelector("span").textContent = "开始生成讲解";
  if (message) formError.focus?.();
}


const savedLessonActions = createSavedLessonActions({
  fetchImpl: (...args) => fetch(...args),
  navigate: (path) => window.location.assign(path),
  clipboard: window.navigator.clipboard || {
    writeText: async () => {
      throw new Error("clipboard unavailable");
    },
  },
  view: {
    showCompletion(lessonId, path) {
      clearPolling();
      activeJob = null;
      progress.hidden = true;
      form.hidden = true;
      savedLessonEntry.hidden = true;
      completion.hidden = false;
      completedLessonId.value = lessonId;
      enterCompletedLesson.href = path;
      copyLessonStatus.textContent = "";
      completion.focus?.();
    },
    setLookupPending(pending) {
      lookupPending = pending;
      const button = existingLessonForm.querySelector("button[type='submit']");
      button.disabled = pending;
      button.textContent = pending ? "正在查找" : "打开课程";
      existingLessonForm.setAttribute("aria-busy", String(pending));
    },
    showLookupError(message) {
      existingLessonError.textContent = message;
      existingLessonError.hidden = !message;
    },
    showCopyStatus(message, success) {
      copyLessonStatus.textContent = message;
      copyLessonStatus.dataset.state = success ? "success" : "error";
    },
    restoreForm() {
      restoreForm();
      form.querySelector("textarea, input, select")?.focus();
    },
    selectCompletedLessonId() {
      completedLessonId.focus();
      completedLessonId.select();
    },
  },
});


async function safeJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}


async function pollJob(jobId) {
  if (activeJob !== jobId) return;
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) throw new Error("job unavailable");
    const job = await response.json();
    setProgressStage(job.stage);
    if (job.status === "completed" && job.lesson_id) {
      savedLessonActions.showCompletion(job.lesson_id);
      return;
    }
    if (job.status === "failed") {
      restoreForm(job.error || "课程生成失败，请稍后重试。");
      return;
    }
    pollTimer = window.setTimeout(
      () => pollJob(jobId),
      POLL_INTERVAL_MS,
    );
  } catch {
    restoreForm("连接中断，未能确认生成状态。请稍后重试。");
  }
}


form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "正在提交";
  formError.hidden = true;
  completion.hidden = true;

  const data = new FormData(form);
  const method = String(data.get("required_method") || "").trim();
  const referenceSolution = String(
    data.get("reference_solution_text") || "",
  ).trim();
  const payload = {
    problem_text: String(data.get("problem_text") || "").trim(),
    reference_answer: String(data.get("reference_answer") || "").trim(),
    reference_solution_text: referenceSolution || null,
    required_method: method || null,
    lesson_length: String(data.get("lesson_length") || "standard"),
  };

  try {
    const response = await fetch("/api/lessons/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });
    const result = await safeJson(response);
    if (!response.ok || !result?.job_id) {
      const message = response.status === 422
        ? "题目或参考答案格式不完整，请检查后再试。"
        : "暂时无法创建课堂，请稍后重试。";
      throw new Error(message);
    }
    activeJob = result.job_id;
    showProgress();
    pollJob(activeJob);
  } catch (error) {
    restoreForm(
      error instanceof Error && error.message
        ? error.message
        : "暂时无法创建课堂，请稍后重试。",
    );
  }
});


existingLessonForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (lookupPending) return;
  await savedLessonActions.openExisting(existingLessonId.value);
});

copyLessonButton.addEventListener("click", () => {
  savedLessonActions.copyLessonId(completedLessonId.value);
});
createAnotherButton.addEventListener("click", () => {
  savedLessonActions.createAnother();
});
returnButton.addEventListener("click", () => restoreForm());
window.addEventListener("beforeunload", clearPolling);
loadHealth();
