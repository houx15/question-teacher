const LESSON_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

export function isValidLessonId(value) {
  return typeof value === "string" && LESSON_ID_PATTERN.test(value);
}

export function lessonPath(lessonId) {
  return `/lesson/${encodeURIComponent(lessonId)}`;
}

export function createSavedLessonActions({
  fetchImpl,
  navigate,
  clipboard,
  view,
}) {
  let lookupSequence = 0;

  function showCompletion(lessonId) {
    view.showCompletion(lessonId, lessonPath(lessonId));
  }

  async function openExisting(rawLessonId) {
    const requestSequence = ++lookupSequence;
    const lessonId = String(rawLessonId || "").trim();
    view.showLookupError("");
    if (!isValidLessonId(lessonId)) {
      view.showLookupError("课程 ID 格式不正确，请检查后重试。");
      return;
    }

    view.setLookupPending(true);
    try {
      const path = lessonPath(lessonId);
      const response = await fetchImpl(
        `/api/lessons/${encodeURIComponent(lessonId)}`,
        {
          headers: { Accept: "application/json" },
          cache: "no-store",
        },
      );
      if (requestSequence !== lookupSequence) return;
      if (response.status === 404) {
        view.showLookupError("没有找到这个课程 ID。");
        return;
      }
      if (response.status !== 200) {
        view.showLookupError("暂时无法读取课程，请稍后重试。");
        return;
      }
      navigate(path);
    } catch {
      if (requestSequence === lookupSequence) {
        view.showLookupError("暂时无法读取课程，请稍后重试。");
      }
    } finally {
      if (requestSequence === lookupSequence) {
        view.setLookupPending(false);
      }
    }
  }

  async function copyLessonId(lessonId) {
    try {
      await clipboard.writeText(lessonId);
      view.showCopyStatus("课程 ID 已复制。", true);
    } catch {
      view.selectCompletedLessonId();
      view.showCopyStatus(
        "未能自动复制，请手动复制上方课程 ID。",
        false,
      );
    }
  }

  function createAnother() {
    lookupSequence += 1;
    view.restoreForm();
  }

  return {
    copyLessonId,
    createAnother,
    openExisting,
    showCompletion,
  };
}
