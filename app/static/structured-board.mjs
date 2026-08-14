const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const LINE_ROLES = new Set([
  "knowledge_anchor",
  "working",
  "summary",
  "error_tip",
  "support",
]);
const EMPHASIS_STYLES = new Set(["highlight", "underline", "red"]);
const ANNOTATIONS = new Set(["underline", "arrow", "bracket", "label"]);
const SUPPORTED_ACTIONS = new Set([
  "reveal_step_header",
  "write",
  "transform",
  "focus",
  "emphasize",
  "annotate",
  "fade",
  "reveal",
  "clear_focus",
  "complete_step",
  "scroll_to_step",
  "open_supporting_explanation",
  "close_supporting_explanation",
]);
const EXISTING_LINE_ACTIONS = new Set([
  "transform",
  "focus",
  "emphasize",
  "annotate",
  "fade",
  "reveal",
  "clear_focus",
]);


function cloneLine(line) {
  return {
    ...line,
    ...(line?.emphasis ? { emphasis: { ...line.emphasis } } : {}),
    ...(Array.isArray(line?.annotations)
      ? { annotations: line.annotations.map((item) => ({ ...item })) }
      : {}),
  };
}


function cloneStep(step, stepId) {
  const support = step?.support && typeof step.support === "object"
    ? {
      ...step.support,
      lines: Array.isArray(step.support.lines)
        ? step.support.lines.map(cloneLine)
        : [],
    }
    : null;
  return {
    ...step,
    stepId: typeof step?.stepId === "string" ? step.stepId : stepId,
    label: typeof step?.label === "string" ? step.label : "",
    status: typeof step?.status === "string" ? step.status : "questioning",
    lines: Array.isArray(step?.lines) ? step.lines.map(cloneLine) : [],
    support,
  };
}


export function emptyStructuredBoard() {
  return {
    steps: new Map(),
    activeStepId: null,
    requestedScrollStepId: null,
  };
}


export function cloneStructuredBoard(current) {
  const steps = current?.steps instanceof Map
    ? new Map(
      [...current.steps.entries()].map(([stepId, step]) => [
        stepId,
        cloneStep(step, stepId),
      ]),
    )
    : new Map();
  return {
    steps,
    activeStepId: typeof current?.activeStepId === "string"
      ? current.activeStepId
      : null,
    requestedScrollStepId: typeof current?.requestedScrollStepId === "string"
      ? current.requestedScrollStepId
      : null,
  };
}


function validId(value) {
  return typeof value === "string" && SAFE_ID.test(value);
}


function validAction(action) {
  if (
    !action
    || action.surface !== "board"
    || !SUPPORTED_ACTIONS.has(action.type)
    || !validId(action.teaching_step_id)
    || !validId(action.target)
  ) {
    return false;
  }
  if (
    ["reveal_step_header", "complete_step", "scroll_to_step"].includes(
      action.type,
    )
    && action.target !== action.teaching_step_id
  ) {
    return false;
  }
  if (
    action.type === "reveal_step_header"
    && (typeof action.step_label !== "string" || !action.step_label.trim())
  ) {
    return false;
  }
  if (action.type === "write") {
    return (
      typeof action.content === "string"
      && action.content.length > 0
      && LINE_ROLES.has(action.board_role)
    );
  }
  if (action.type === "transform") {
    return typeof action.content === "string" && action.content.length > 0;
  }
  if (action.type === "emphasize") {
    return (
      EMPHASIS_STYLES.has(action.emphasis_style)
      && (
        action.persistence === undefined
        || action.persistence === null
        || action.persistence === "transient"
        || action.persistence === "trace"
      )
    );
  }
  if (action.type === "annotate") {
    return ANNOTATIONS.has(action.annotation);
  }
  return true;
}


function newStep(stepId) {
  return {
    stepId,
    label: "",
    status: "questioning",
    lines: [],
    support: null,
  };
}


function upsertLine(lines, line) {
  const index = lines.findIndex((item) => item?.target === line.target);
  if (index < 0) {
    lines.push(line);
    return;
  }
  lines[index] = { ...lines[index], ...line };
}


function lineCollection(step, target) {
  if (step.support) {
    const supportIndex = step.support.lines.findIndex(
      (line) => line?.target === target,
    );
    if (supportIndex >= 0) {
      return { lines: step.support.lines, index: supportIndex };
    }
  }
  const index = step.lines.findIndex((line) => line?.target === target);
  return index >= 0 ? { lines: step.lines, index } : null;
}


function clearLineFocus(state) {
  for (const step of state.steps.values()) {
    const collections = [step.lines, step.support?.lines || []];
    for (const lines of collections) {
      for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index];
        if (line.focused !== true && line.focusFaded !== true) continue;
        lines[index] = { ...line, focused: false, focusFaded: false };
      }
    }
  }
}


function applyLineAction(state, step, action) {
  const located = lineCollection(step, action.target);
  if (!located) return;
  const current = located.lines[located.index];
  if (action.type === "transform") {
    located.lines[located.index] = {
      ...current,
      previousContent: current.content,
      content: action.content,
      source: action.source || current.source || null,
    };
  } else if (action.type === "focus") {
    clearLineFocus(state);
    const refreshed = lineCollection(step, action.target);
    refreshed.lines[refreshed.index] = {
      ...refreshed.lines[refreshed.index],
      focused: true,
      focusFaded: false,
    };
  } else if (action.type === "emphasize") {
    located.lines[located.index] = {
      ...current,
      emphasis: {
        style: action.emphasis_style,
        strength: "active",
        persistence: action.persistence || "transient",
      },
    };
  } else if (action.type === "annotate") {
    located.lines[located.index] = {
      ...current,
      annotations: [
        ...(current.annotations || []),
        {
          type: action.annotation,
          content: typeof action.content === "string" ? action.content : "",
          relationTarget: validId(action.relation_target)
            ? action.relation_target
            : null,
        },
      ],
    };
  } else if (action.type === "fade") {
    const next = { ...current, focused: false, focusFaded: false };
    if (current.emphasis?.persistence === "trace") {
      next.emphasis = { ...current.emphasis, strength: "trace" };
    } else {
      delete next.emphasis;
    }
    located.lines[located.index] = next;
  } else if (action.type === "reveal") {
    located.lines[located.index] = { ...current, revealed: true };
  } else if (action.type === "clear_focus") {
    clearLineFocus(state);
  }
}


export function applyStructuredBoardAction(current, action) {
  const state = cloneStructuredBoard(current);
  if (!validAction(action)) return state;

  const stepId = action.teaching_step_id;
  const existingStep = state.steps.get(stepId);
  if (!existingStep && EXISTING_LINE_ACTIONS.has(action.type)) return state;
  const step = existingStep || newStep(stepId);
  state.steps.set(stepId, step);

  if (action.type === "reveal_step_header") {
    step.label = action.step_label;
    step.status = "active";
    state.activeStepId = stepId;
  } else if (action.type === "write") {
    const line = {
      target: action.target,
      content: action.content,
      role: action.board_role,
      source: action.source || null,
    };
    if (action.board_role === "support") {
      if (!step.support) {
        state.steps.delete(stepId);
        return cloneStructuredBoard(current);
      }
      upsertLine(step.support.lines, line);
    } else {
      upsertLine(step.lines, line);
    }
  } else if (action.type === "complete_step") {
    step.status = "completed";
  } else if (action.type === "open_supporting_explanation") {
    step.status = "supporting";
    state.activeStepId = stepId;
    if (!step.support || step.support.target !== action.target) {
      step.support = { target: action.target, lines: [] };
    }
  } else if (action.type === "close_supporting_explanation") {
    step.support = null;
    step.status = "active";
    state.activeStepId = stepId;
  } else if (action.type === "scroll_to_step") {
    state.requestedScrollStepId = stepId;
  } else {
    applyLineAction(state, step, action);
  }
  return state;
}


export function stepForScroll(state) {
  const stepId = state?.requestedScrollStepId;
  if (!validId(stepId) || !(state?.steps instanceof Map)) return null;
  return state.steps.get(stepId) || null;
}
