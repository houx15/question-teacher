import test from "node:test";
import assert from "node:assert/strict";

import {
  MAX_MATH_TEXT_LENGTH,
  MAX_NORMALIZATION_PASSES,
  mathSegments,
  normalizeLegacyMath,
  renderMathText,
} from "../app/static/math-text.mjs";


test("explicit inline and display delimiters preserve surrounding text", () => {
  assert.deepEqual(mathSegments("先算\\(x^2 - 1\\)，再看\\[\\frac{1}{2}\\]。"), [
    { type: "text", value: "先算" },
    { type: "math", value: "x^2 - 1", displayMode: false },
    { type: "text", value: "，再看" },
    { type: "math", value: "\\frac{1}{2}", displayMode: true },
    { type: "text", value: "。" },
  ]);
});


test("legacy equation after Chinese instruction becomes an inline math segment", () => {
  assert.deepEqual(mathSegments("解方程 x^2-6x+5=0"), [
    { type: "text", value: "解方程 " },
    { type: "math", value: "x^2-6x+5=0", displayMode: false },
  ]);
});


test("legacy notation normalization supports Unicode operators powers parentheses and sqrt", () => {
  assert.equal(
    normalizeLegacyMath("x² − 3 × (x + 1) ÷ sqrt(2) = 0"),
    "x^{2} - 3 \\times (x + 1) \\div \\sqrt{2} = 0",
  );
  assert.deepEqual(mathSegments("计算 x² − 3 × (x + 1) ÷ sqrt(2) = 0 后说明理由"), [
    { type: "text", value: "计算 " },
    {
      type: "math",
      value: "x^{2} - 3 \\times (x + 1) \\div \\sqrt{2} = 0",
      displayMode: false,
    },
    { type: "text", value: " 后说明理由" },
  ]);
});


test("consecutive Unicode superscripts become one exponent group", () => {
  assert.equal(normalizeLegacyMath("x¹⁰"), "x^{10}");
  assert.equal(normalizeLegacyMath("x²"), "x^{2}");
});


test("nested sqrt normalization has a fixed pass bound", () => {
  const nested = `${"sqrt(".repeat(MAX_NORMALIZATION_PASSES + 1)}2${")".repeat(MAX_NORMALIZATION_PASSES + 1)}`;
  const normalized = normalizeLegacyMath(nested);

  assert.equal(MAX_NORMALIZATION_PASSES, 8);
  assert.equal((normalized.match(/\\sqrt/g) ?? []).length, MAX_NORMALIZATION_PASSES);
  assert.match(normalized, /^sqrt\(/);
});


test("plain Chinese remains one text segment", () => {
  assert.deepEqual(mathSegments("先观察等式两边，再说明每一步的依据。"), [
    { type: "text", value: "先观察等式两边，再说明每一步的依据。" },
  ]);
});


test("authored math processing is bounded by source length", () => {
  const atLimit = `x=1${" ".repeat(MAX_MATH_TEXT_LENGTH - 3)}`;
  const overLimit = `${atLimit} `;
  const nodes = [];
  const container = { replaceChildren(...children) { nodes.push(...children); } };
  const documentImpl = {
    createTextNode(value) { return { kind: "text", value }; },
    createElement() { throw new Error("math DOM node must not be created"); },
  };

  assert.equal(mathSegments(atLimit)[0].type, "math");
  assert.deepEqual(mathSegments(overLimit), [textSegment(overLimit)]);
  renderMathText(container, overLimit, {
    documentImpl,
    renderImpl() { throw new Error("KaTeX must not run for over-limit input"); },
  });
  assert.deepEqual(nodes, [{ kind: "text", value: overLimit }]);
});


test("malformed explicit delimiters remain safe plain text", () => {
  assert.deepEqual(mathSegments("这里有未闭合的\\(x^2 + 1，原文保留"), [
    { type: "text", value: "这里有未闭合的\\(x^2 + 1，原文保留" },
  ]);
  assert.deepEqual(mathSegments("这里有未闭合的\\[x^2 + 1，原文保留"), [
    { type: "text", value: "这里有未闭合的\\[x^2 + 1，原文保留" },
  ]);
});


test("multiple legacy math fragments preserve source order", () => {
  assert.deepEqual(mathSegments("将 x+1=2 化为 x=1，然后验算 1+1=2。"), [
    { type: "text", value: "将 " },
    { type: "math", value: "x+1=2", displayMode: false },
    { type: "text", value: " 化为 " },
    { type: "math", value: "x=1", displayMode: false },
    { type: "text", value: "，然后验算 " },
    { type: "math", value: "1+1=2", displayMode: false },
    { type: "text", value: "。" },
  ]);
});


test("legacy detection keeps dates labels and prose-like hyphens as text", () => {
  for (const value of ("2026-08-06", "第1-2步", "A-B 测试", "foo-bar@example.com")) {
    assert.deepEqual(mathSegments(value), [textSegment(value)]);
  }
  assert.equal(mathSegments("先算 x-3")[1].type, "math");
  assert.equal(mathSegments("计算 2*x")[1].type, "math");
});


test("legacy detection also runs around valid explicit math delimiters", () => {
  assert.deepEqual(mathSegments("目标 \\(x=1\\)；另一个 x=2"), [
    textSegment("目标 "),
    { type: "math", value: "x=1", displayMode: false },
    textSegment("；另一个 "),
    { type: "math", value: "x=2", displayMode: false },
  ]);
});


test("renderer uses text nodes and finite safe KaTeX options", () => {
  const nodes = [];
  const calls = [];
  const container = { replaceChildren(...children) { nodes.push(...children); } };
  const documentImpl = {
    createTextNode(value) { return { kind: "text", value }; },
    createElement(tagName) { return { kind: tagName, className: "", textContent: "" }; },
  };

  renderMathText(container, "说明 \\(x=1\\)", {
    documentImpl,
    renderImpl(expression, node, options) {
      calls.push({ expression, node, options });
    },
  });

  assert.deepEqual(nodes[0], { kind: "text", value: "说明 " });
  assert.equal(calls[0].expression, "x=1");
  assert.equal(calls[0].options.trust, false);
  assert.equal(calls[0].options.maxSize, 20);
  assert.equal(calls[0].options.maxExpand, 200);
  assert.equal(Number.isFinite(calls[0].options.maxSize), true);
  assert.equal(Number.isFinite(calls[0].options.maxExpand), true);
});


test("renderer falls back to plain text when math rendering throws", () => {
  const nodes = [];
  const container = { replaceChildren(...children) { nodes.push(...children); } };
  const documentImpl = {
    createTextNode(value) { return { kind: "text", value }; },
    createElement(tagName) { return { kind: tagName, className: "", textContent: "" }; },
  };

  renderMathText(container, "\\(\\rule{100000em}{100000em}\\)", {
    documentImpl,
    renderImpl() { throw new Error("hostile input rejected"); },
  });

  assert.deepEqual(nodes, [{
    kind: "span",
    className: "math-fallback",
    textContent: "\\rule{100000em}{100000em}",
  }]);
});


test("hostile rule input receives constrained render options", () => {
  const calls = [];
  const container = { replaceChildren() {} };
  const documentImpl = {
    createTextNode(value) { return { kind: "text", value }; },
    createElement(tagName) { return { kind: tagName }; },
  };

  renderMathText(container, "\\(\\rule{100000em}{100000em}\\)", {
    documentImpl,
    renderImpl(expression, node, options) { calls.push({ expression, node, options }); },
  });

  assert.equal(calls[0].expression, "\\rule{100000em}{100000em}");
  assert.equal(calls[0].options.maxSize, 20);
  assert.equal(calls[0].options.maxExpand, 200);
  assert.equal(calls[0].options.trust, false);
});


function textSegment(value) {
  return { type: "text", value };
}
