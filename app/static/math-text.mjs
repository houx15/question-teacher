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

export const MAX_MATH_TEXT_LENGTH = 4096;
export const MAX_NORMALIZATION_PASSES = 8;
export const MAX_ACCESSIBLE_TEXT_LENGTH = 160;

const LEGACY_CANDIDATE = /[A-Za-z0-9√⁰¹²³⁴⁵⁶⁷⁸⁹+\-−×÷*/^=()（）\s]+/g;
const LEGACY_MATH_MARKER = /[=^⁰¹²³⁴⁵⁶⁷⁸⁹√×÷]|\bsqrt\s*\(|x\s*[+\-−*/]\s*(?:\d|\(|x\b)|(?:\d|\))\s*[+*/]\s*x\b/i;


function textSegment(value) {
  return { type: "text", value };
}


function mathSegment(value, displayMode) {
  return { type: "math", value, displayMode };
}


function explicitMathSegments(value) {
  const delimiterToken = /\\([()[\]])|\$\$|\$/g;
  const tokens = [];
  let active = null;
  let match;

  while ((match = delimiterToken.exec(value)) !== null) {
    const rawToken = match[0];
    if (rawToken.startsWith("$")) {
      let precedingBackslashes = 0;
      for (
        let index = match.index - 1;
        index >= 0 && value[index] === "\\";
        index -= 1
      ) {
        precedingBackslashes += 1;
      }
      if (precedingBackslashes % 2 === 1) continue;
    }

    const token = match[1] ?? rawToken;
    if (active) {
      if (token !== active.closingToken) return null;
      tokens.push({
        start: active.index,
        end: delimiterToken.lastIndex,
        value: value.slice(active.contentStart, match.index),
        displayMode: active.displayMode,
      });
      active = null;
      continue;
    }

    if (token === ")" || token === "]") return null;
    active = {
      closingToken: token === "(" ? ")" : token === "[" ? "]" : token,
      index: match.index,
      contentStart: delimiterToken.lastIndex,
      displayMode: token === "[" || token === "$$",
    };
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
    .replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹]+/g, (characters) => `^{${[...characters].map((character) => SUPERSCRIPTS[character]).join("")}}`)
    .replace(/−/g, "-")
    .replace(/×/g, "\\times")
    .replace(/÷/g, "\\div");

  for (let pass = 0; pass < MAX_NORMALIZATION_PASSES; pass += 1) {
    const next = normalized.replace(/sqrt\s*\(([^()]*)\)/g, "\\sqrt{$1}");
    if (next === normalized) break;
    normalized = next;
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
  if (source.length > MAX_MATH_TEXT_LENGTH) return [textSegment(source)];

  const explicit = explicitMathSegments(source);
  if (explicit === null) return [textSegment(source)];
  if (explicit) {
    return explicit.flatMap((segment) => (
      segment.type === "text" ? legacyMathSegments(segment.value) : [segment]
    ));
  }
  return legacyMathSegments(source);
}


function mathExpressionToPlainText(value) {
  let plain = value
    .replace(/\\(?:times|cdot)\b/g, "×")
    .replace(/\\div\b/g, "÷")
    .replace(/\\pm\b/g, "±")
    .replace(/\\neq\b/g, "≠")
    .replace(/\\leq?\b/g, "≤")
    .replace(/\\geq?\b/g, "≥")
    .replace(/\\(?:left|right)\b/g, "");

  for (let pass = 0; pass < MAX_NORMALIZATION_PASSES; pass += 1) {
    const next = plain
      .replace(/\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}/g, "$1")
      .replace(/\\sqrt\{([^{}]*)\}/g, "√($1)")
      .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "($1)/($2)")
      .replace(/\^\{([^{}]*)\}/g, "^$1")
      .replace(/_\{([^{}]*)\}/g, "_$1");
    if (next === plain) break;
    plain = next;
  }

  return plain.replace(/[{}]/g, "");
}


function truncateCodePoints(value, limit) {
  const codePoints = [];
  for (const character of value) {
    if (codePoints.length === limit) {
      return `${codePoints.slice(0, limit - 1).join("")}…`;
    }
    codePoints.push(character);
  }
  return value;
}


export function mathTextToPlainText(value) {
  const boundedSource = truncateCodePoints(
    String(value ?? ""),
    MAX_MATH_TEXT_LENGTH,
  );
  const plainText = mathSegments(boundedSource)
    .map((segment) => (
      segment.type === "math"
        ? mathExpressionToPlainText(segment.value)
        : segment.value
    ))
    .join("")
    .replace(/\s+/g, " ")
    .trim();
  return truncateCodePoints(plainText, MAX_ACCESSIBLE_TEXT_LENGTH);
}


export function renderMathText(
  container,
  value,
  { documentImpl = document, renderImpl = katex.render } = {},
) {
  const nodes = mathSegments(value).map((segment) => {
    if (segment.type === "text") return documentImpl.createTextNode(segment.value);

    const node = documentImpl.createElement("span");
    try {
      renderImpl(segment.value, node, {
        displayMode: segment.displayMode,
        throwOnError: false,
        trust: false,
        strict: "warn",
        maxSize: 20,
        maxExpand: 200,
      });
    } catch {
      node.className = "math-fallback";
      node.textContent = segment.value;
    }
    return node;
  });

  container.replaceChildren(...nodes);
}
