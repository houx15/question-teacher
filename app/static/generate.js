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
  "正在核对题目材料": "核对题目、答案和参考解析是否一致。",
  "正在整理参考解析": "提取参考解析的数学步骤、条件与结论。",
  "正在设计解题思维轨迹": "设计学生如何观察条件、做出决定并理解为什么。",
  "正在设计课堂推进": "把思路组织成学生能够一步步跟上的课堂结构。",
  "正在设计互动": "在关键决策点设计选择题，检查学生是否真正理解。",
  "正在编写讲稿": "把课堂推进和每个互动结果写成可朗读的讲稿。",
  "正在编排板书与高亮": "让关键公式、步骤与重点随讲解同步出现。",
  "正在审核和优化课程": "模拟学生听课并审核整节课，必要时定向修订。",
  "正在编译课程": "将通过审核的台本、板书和互动组装成可播放课堂。",
  "正在生成语音": "最后为每个短讲解片段生成同步语音。",
  "正在保存课程": "正在保存语音和课堂内容，成功后会显示课程 ID。",
  "课程已生成": "课堂已经准备好。",
};

let activeJob = null;
let pollTimer = null;
let lookupPending = false;
let authoringLocked = false;


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
  savedLessonActions.unlockAuthoring();
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
      button.disabled = pending || authoringLocked;
      button.textContent = pending ? "正在查找" : "打开课程";
      existingLessonForm.setAttribute("aria-busy", String(pending));
    },
    setAuthoringLocked(locked) {
      authoringLocked = locked;
      savedLessonEntry.hidden = locked;
      existingLessonId.disabled = locked;
      const button = existingLessonForm.querySelector("button[type='submit']");
      button.disabled = locked || lookupPending;
      existingLessonForm.setAttribute("aria-disabled", String(locked));
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
  savedLessonActions.lockForGeneration();

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
  if (lookupPending || savedLessonActions.isAuthoringLocked()) return;
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
