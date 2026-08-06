import katex from "./vendor/katex/katex.mjs";


const SUPERSCRIPTS = {
  "⁰": "0",
  "¹": "1",
  "²": "2",
  "³": "3",
  "⁴": "4",
  "⁵": "5",
  "⁶": "6",
  "⁷": "7",
  "⁸": "8",
  "⁹": "9",
};

const LEGACY_CANDIDATE = /[A-Za-z0-9√⁰¹²³⁴⁵⁶⁷⁸⁹+\-−×÷*/^=()（）\s]+/g;
const LEGACY_MATH_MARKER = /[=^⁰¹²³⁴⁵⁶⁷⁸⁹√×÷]|\bsqrt\s*\(|[A-Za-z0-9)]\s*[+\-−]\s*[A-Za-z0-9(]/;


function textSegment(value) {
  return { type: "text", value };
}


function mathSegment(value, displayMode) {
  return { type: "math", value, displayMode };
}


function explicitMathSegments(value) {
  const delimiterToken = /\\([()[\]])/g;
  const tokens = [];
  let active = null;
  let match;

  while ((match = delimiterToken.exec(value)) !== null) {
    const token = match[1];
    if (token === "(" || token === "[") {
      if (active) return null;
      active = { token, index: match.index, contentStart: delimiterToken.lastIndex };
      continue;
    }

    if (!active || (active.token === "(" ? token !== ")" : token !== "]")) {
      return null;
    }

    tokens.push({
      start: active.index,
      end: delimiterToken.lastIndex,
      value: value.slice(active.contentStart, match.index),
      displayMode: active.token === "[",
    });
    active = null;
  }

  if (active) return null;
  if (tokens.length === 0) return undefined;

  const segments = [];
  let cursor = 0;
  for (const token of tokens) {
    if (token.start > cursor) segments.push(textSegment(value.slice(cursor, token.start)));
    segments.push(mathSegment(token.value, token.displayMode));
    cursor = token.end;
  }
  if (cursor < value.length) segments.push(textSegment(value.slice(cursor)));
  return segments;
}


export function normalizeLegacyMath(value) {
  let normalized = String(value).trim()
    .replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]/g, (character) => `^${SUPERSCRIPTS[character]}`)
    .replace(/−/g, "-")
    .replace(/×/g, "\\times")
    .replace(/÷/g, "\\div");

  while (/sqrt\s*\(([^()]*)\)/.test(normalized)) {
    normalized = normalized.replace(/sqrt\s*\(([^()]*)\)/g, "\\sqrt{$1}");
  }
  return normalized;
}


function legacyMathSegments(value) {
  const segments = [];
  let cursor = 0;
  let match;

  while ((match = LEGACY_CANDIDATE.exec(value)) !== null) {
    const raw = match[0];
    const leadingWhitespace = raw.match(/^\s*/)[0].length;
    const trailingWhitespace = raw.match(/\s*$/)[0].length;
    const expression = raw.slice(leadingWhitespace, raw.length - trailingWhitespace);

    if (!expression || !LEGACY_MATH_MARKER.test(expression)) continue;

    const start = match.index + leadingWhitespace;
    const end = match.index + raw.length - trailingWhitespace;
    if (start > cursor) segments.push(textSegment(value.slice(cursor, start)));
    segments.push(mathSegment(normalizeLegacyMath(expression), false));
    cursor = end;
  }

  if (cursor === 0) return [textSegment(value)];
  if (cursor < value.length) segments.push(textSegment(value.slice(cursor)));
  return segments;
}


export function mathSegments(value) {
  const source = String(value ?? "");
  const explicit = explicitMathSegments(source);
  if (explicit === null) return [textSegment(source)];
  if (explicit) return explicit;
  return legacyMathSegments(source);
}


export function renderMathText(container, value) {
  const nodes = mathSegments(value).map((segment) => {
    if (segment.type === "text") return document.createTextNode(segment.value);

    const node = document.createElement("span");
    try {
      katex.render(segment.value, node, {
        displayMode: segment.displayMode,
        throwOnError: false,
        trust: false,
        strict: "warn",
      });
    } catch {
      node.className = "math-fallback";
      node.textContent = segment.value;
    }
    return node;
  });

  container.replaceChildren(...nodes);
}
