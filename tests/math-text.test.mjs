import test from "node:test";
import assert from "node:assert/strict";

import { mathSegments, normalizeLegacyMath } from "../app/static/math-text.mjs";


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
    "x^2 - 3 \\times (x + 1) \\div \\sqrt{2} = 0",
  );
  assert.deepEqual(mathSegments("计算 x² − 3 × (x + 1) ÷ sqrt(2) = 0 后说明理由"), [
    { type: "text", value: "计算 " },
    {
      type: "math",
      value: "x^2 - 3 \\times (x + 1) \\div \\sqrt{2} = 0",
      displayMode: false,
    },
    { type: "text", value: " 后说明理由" },
  ]);
});


test("plain Chinese remains one text segment", () => {
  assert.deepEqual(mathSegments("先观察等式两边，再说明每一步的依据。"), [
    { type: "text", value: "先观察等式两边，再说明每一步的依据。" },
  ]);
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
