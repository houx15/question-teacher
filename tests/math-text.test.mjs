import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import * as mathText from "../app/static/math-text.mjs";


const {
  MAX_ACCESSIBLE_TEXT_LENGTH,
  MAX_MATH_TEXT_LENGTH,
  MAX_NORMALIZATION_PASSES,
  mathSegments,
  mathTextToPlainText,
  normalizeLegacyMath,
  problemFocusTargets,
  renderMathText,
  renderProblemMathText,
} = mathText;


const problemFocusCases = JSON.parse(readFileSync(
  new URL("./fixtures/problem-focus-cases.json", import.meta.url),
  "utf8",
));


function fakeDom() {
  const makeElement = (tagName) => ({
    kind: tagName,
    className: "",
    textContent: "",
    attributes: {},
    children: [],
    append(...children) {
      this.children.push(...children);
    },
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    },
  });
  return {
    documentImpl: {
      createTextNode(value) {
        return { kind: "text", value };
      },
      createElement: makeElement,
    },
    container: {
      children: [],
      replaceChildren(...children) {
        this.children = children;
      },
    },
  };
}


const PARAMETER_ROOT_SOURCE = String.raw`若$2n$且$n\ne0$`;
const PARAMETER_ROOT_TARGETS = [
  {
    target_id: "problem-math-001",
    math_text: "2n",
    display_mode: false,
    ordinal: 1,
  },
  {
    target_id: "problem-math-002",
    math_text: String.raw`n\ne0`,
    display_mode: false,
    ordinal: 2,
  },
];


test("focus wrapper renderer exports the synchronized problem contract", () => {
  assert.equal(typeof renderProblemMathText, "function");
});


test("focus wrapper renders exact problem formulas with safe active and trace classes", () => {
  const { documentImpl, container } = fakeDom();
  const calls = [];
  const visualState = new Map([
    [
      "problem-math-001",
      {
        style: "highlight",
        strength: "active",
        persistence: "trace",
      },
    ],
    [
      "problem-math-002",
      {
        style: "underline",
        strength: "trace",
        persistence: "trace",
      },
    ],
  ]);

  renderProblemMathText(container, PARAMETER_ROOT_SOURCE, {
    focusTargets: PARAMETER_ROOT_TARGETS,
    visualState,
    documentImpl,
    renderImpl(expression, node, options) {
      calls.push({ expression, node, options });
    },
  });

  const wrappers = container.children.filter(
    (node) => node.className?.includes("focus-target"),
  );
  assert.equal(wrappers.length, 2);
  assert.deepEqual(
    wrappers.map((node) => node.attributes["data-focus-target"]),
    ["problem-math-001", "problem-math-002"],
  );
  assert.deepEqual(
    wrappers.map((node) => node.className),
    [
      "focus-target is-highlighted is-active",
      "focus-target is-underlined is-trace",
    ],
  );
  assert.deepEqual(
    container.children
      .filter((node) => node.kind === "text")
      .map((node) => node.value),
    ["若", "且"],
  );
  assert.equal(
    container.children.some(
      (node) => node.value?.includes("$") || node.value?.includes(String.raw`\(`),
    ),
    false,
  );
  assert.deepEqual(
    calls.map((call) => call.expression),
    ["2n", String.raw`n\ne0`],
  );
  assert.deepEqual(
    calls.map((call) => call.node),
    wrappers.map((wrapper) => wrapper.children[0]),
  );
  for (const call of calls) {
    assert.equal(call.options.displayMode, false);
    assert.equal(call.options.trust, false);
    assert.equal(call.options.maxSize, 20);
    assert.equal(call.options.maxExpand, 200);
    assert.equal(Number.isFinite(call.options.maxSize), true);
    assert.equal(Number.isFinite(call.options.maxExpand), true);
  }
});


test("focus wrapper never maps hostile emphasis text into class or style", () => {
  const { documentImpl, container } = fakeDom();
  const hostile = "x; background:url(javascript:alert(1))";

  renderProblemMathText(container, PARAMETER_ROOT_SOURCE, {
    focusTargets: PARAMETER_ROOT_TARGETS,
    visualState: new Map([
      [
        "problem-math-001",
        { style: hostile, strength: "active", persistence: "trace" },
      ],
    ]),
    documentImpl,
    renderImpl() {},
  });

  const wrapper = container.children[1];
  assert.equal(wrapper.className, "focus-target");
  assert.equal("style" in wrapper, false);
  assert.equal(JSON.stringify(wrapper).includes(hostile), false);
});


test("focus wrapper fails closed for every server or local parsing mismatch", () => {
  const mismatches = [
    PARAMETER_ROOT_TARGETS.slice(0, 1),
    [
      { ...PARAMETER_ROOT_TARGETS[0], target_id: "problem-math-999" },
      PARAMETER_ROOT_TARGETS[1],
    ],
    [
      { ...PARAMETER_ROOT_TARGETS[0], math_text: "different" },
      PARAMETER_ROOT_TARGETS[1],
    ],
    [
      { ...PARAMETER_ROOT_TARGETS[0], display_mode: true },
      PARAMETER_ROOT_TARGETS[1],
    ],
    [
      { ...PARAMETER_ROOT_TARGETS[0], ordinal: 2 },
      PARAMETER_ROOT_TARGETS[1],
    ],
  ];

  for (const focusTargets of mismatches) {
    const { documentImpl, container } = fakeDom();
    const calls = [];
    renderProblemMathText(container, PARAMETER_ROOT_SOURCE, {
      focusTargets,
      documentImpl,
      renderImpl(...args) { calls.push(args); },
    });
    assert.equal(
      container.children.some(
        (node) => node.className?.includes("focus-target"),
      ),
      false,
    );
    assert.equal(calls.length, 2);
  }

  const malformed = fakeDom();
  renderProblemMathText(
    malformed.container,
    String.raw`若$2n且$n\ne0$`,
    {
      focusTargets: PARAMETER_ROOT_TARGETS,
      documentImpl: malformed.documentImpl,
      renderImpl() {},
    },
  );
  assert.equal(
    malformed.container.children.some(
      (node) => node.className?.includes("focus-target"),
    ),
    false,
  );
});


test("focus wrapper leaves legacy-only math renderable but untargeted", () => {
  const { documentImpl, container } = fakeDom();
  const calls = [];

  renderProblemMathText(container, "解方程 x^2-6x+5=0", {
    focusTargets: [],
    documentImpl,
    renderImpl(...args) { calls.push(args); },
  });

  assert.equal(calls.length, 1);
  assert.equal(
    container.children.some(
      (node) => node.className?.includes("focus-target"),
    ),
    false,
  );
});


test("focus wrapper fails closed when shared parsing finds extra legacy math", () => {
  const { documentImpl, container } = fakeDom();
  const calls = [];

  const result = renderProblemMathText(
    container,
    String.raw`解 x^2=1，再看$2n$`,
    {
      focusTargets: [PARAMETER_ROOT_TARGETS[0]],
      documentImpl,
      renderImpl(...args) { calls.push(args); },
    },
  );

  assert.deepEqual(result, { focusWrapped: false });
  assert.equal(calls.length, 2);
  assert.equal(
    container.children.some(
      (node) => node.className?.includes("focus-target"),
    ),
    false,
  );
});


test("problem focus targets match shared explicit delimiter cases", () => {
  for (const fixture of problemFocusCases) {
    if (fixture.code_points) {
      assert.deepEqual(
        Array.from(fixture.source, (character) => character.codePointAt(0)),
        fixture.code_points,
        fixture.name,
      );
    }

    const targets = problemFocusTargets(fixture.source);

    assert.deepEqual(
      targets.map((target) => target.math_text),
      fixture.math,
      fixture.name,
    );
    assert.deepEqual(
      targets.map((target) => target.target_id),
      fixture.math.map((_, index) => (
        `problem-math-${String(index + 1).padStart(3, "0")}`
      )),
      fixture.name,
    );
    if (fixture.display_modes) {
      assert.deepEqual(
        targets.map((target) => target.display_mode),
        fixture.display_modes,
        fixture.name,
      );
    }
  }
});


test("problem focus targets expose display mode and stable ordinal", () => {
  assert.deepEqual(
    problemFocusTargets(String.raw`先看\(x=2\)，再看\[\frac{1}{2}\]`)
      .map(({ display_mode, ordinal }) => ({ display_mode, ordinal })),
    [
      { display_mode: false, ordinal: 1 },
      { display_mode: true, ordinal: 2 },
    ],
  );
});


test("problem focus targets enforce 64 target boundary", () => {
  const targets = problemFocusTargets(String.raw`\(x=1\)`.repeat(64));

  assert.equal(targets.length, 64);
  assert.equal(targets.at(-1).target_id, "problem-math-064");
  assert.equal(targets.at(-1).ordinal, 64);
  assert.deepEqual(problemFocusTargets(String.raw`\(x=1\)`.repeat(65)), []);
});


test("problem focus targets count Unicode code points for budget", () => {
  const withinBudget = `${"😀".repeat(4091)}${String.raw`\(x\)`}`;
  const overBudget = `${"😀".repeat(4092)}${String.raw`\(x\)`}`;
  const withinDom = fakeDom();
  const withinCalls = [];
  const overDom = fakeDom();
  const overCalls = [];

  assert.equal([...withinBudget].length, 4096);
  assert.equal(withinBudget.length > MAX_MATH_TEXT_LENGTH, true);
  const withinTargets = problemFocusTargets(withinBudget);
  assert.deepEqual(withinTargets.map((target) => target.math_text), ["x"]);
  assert.deepEqual(
    renderProblemMathText(withinDom.container, withinBudget, {
      focusTargets: withinTargets,
      documentImpl: withinDom.documentImpl,
      renderImpl(...args) { withinCalls.push(args); },
    }),
    { focusWrapped: true },
  );
  assert.equal(withinCalls.length, 1);
  assert.equal(
    withinDom.container.children.at(-1).attributes["data-focus-target"],
    "problem-math-001",
  );

  assert.equal([...overBudget].length, 4097);
  assert.deepEqual(problemFocusTargets(overBudget), []);
  assert.deepEqual(
    renderProblemMathText(overDom.container, overBudget, {
      focusTargets: [],
      documentImpl: overDom.documentImpl,
      renderImpl(...args) { overCalls.push(args); },
    }),
    { focusWrapped: false },
  );
  assert.equal(overCalls.length, 0);
  assert.equal(
    overDom.container.children.some(
      (node) => node.attributes?.["data-focus-target"],
    ),
    false,
  );
});


test("problem focus targets exclude renderable legacy math", () => {
  const source = "解方程 x^2-6x+5=0";

  assert.equal(
    mathSegments(source).some((segment) => segment.type === "math"),
    true,
  );
  assert.deepEqual(problemFocusTargets(source), []);
});


test("plain-text accessibility helper removes visual math wrappers without duplication", () => {
  assert.equal(mathTextToPlainText(String.raw`\(9\)`), "9");
  assert.equal(
    mathTextToPlainText(String.raw`\(x=3\) 或 \(x=4\)`),
    "x=3 或 x=4",
  );
  assert.equal(
    mathTextToPlainText(String.raw`\(\text{无实数解}\)`),
    "无实数解",
  );
});


test("plain-text accessibility helper preserves ordinary option labels", () => {
  assert.equal(mathTextToPlainText("两边同时加 9"), "两边同时加 9");
});


test("plain-text accessibility helper leaves blank labels for indexed UI fallback", () => {
  assert.equal(mathTextToPlainText(" \n\t "), "");
});


test("plain-text accessibility helper truncates by code point with one ellipsis", () => {
  const atLimit = "x".repeat(MAX_ACCESSIBLE_TEXT_LENGTH);
  const overLimitWithEmoji = `${"x".repeat(158)}😀tail`;

  assert.equal(MAX_ACCESSIBLE_TEXT_LENGTH, 160);
  assert.equal(mathTextToPlainText(atLimit), atLimit);
  assert.equal(
    mathTextToPlainText(overLimitWithEmoji),
    `${"x".repeat(158)}😀…`,
  );
  assert.equal(
    [...mathTextToPlainText(overLimitWithEmoji)].length,
    MAX_ACCESSIBLE_TEXT_LENGTH,
  );
  assert.equal(mathTextToPlainText(overLimitWithEmoji).includes("�"), false);
});


test("plain-text accessibility helper bounds hostile markup-like input", () => {
  const hostilePrefix = String.raw`\(\htmlClass{evil}{x}\)<img src=x onerror=alert(1)>`;
  const hostile = `${hostilePrefix}${"😀".repeat(MAX_MATH_TEXT_LENGTH + 1)}`;
  const result = mathTextToPlainText(hostile);

  assert.equal([...result].length, MAX_ACCESSIBLE_TEXT_LENGTH);
  assert.equal(result.endsWith("…"), true);
  assert.equal(result.includes("�"), false);
});


test("explicit inline and display delimiters preserve surrounding text", () => {
  assert.deepEqual(mathSegments("先算\\(x^2 - 1\\)，再看\\[\\frac{1}{2}\\]。"), [
    { type: "text", value: "先算" },
    { type: "math", value: "x^2 - 1", displayMode: false },
    { type: "text", value: "，再看" },
    { type: "math", value: "\\frac{1}{2}", displayMode: true },
    { type: "text", value: "。" },
  ]);
});


test("dollar delimiters render common pasted LaTeX question text", () => {
  assert.deepEqual(
    mathSegments(
      String.raw`若$2n$ ($n\ne 0$)是方程$x^2-2mx+2n=0$的根`,
    ),
    [
      { type: "text", value: "若" },
      { type: "math", value: "2n", displayMode: false },
      { type: "text", value: " (" },
      { type: "math", value: String.raw`n\ne 0`, displayMode: false },
      { type: "text", value: ")是方程" },
      {
        type: "math",
        value: "x^2-2mx+2n=0",
        displayMode: false,
      },
      { type: "text", value: "的根" },
    ],
  );
  assert.deepEqual(mathSegments(String.raw`结果是$$\frac{1}{2}$$。`), [
    { type: "text", value: "结果是" },
    {
      type: "math",
      value: String.raw`\frac{1}{2}`,
      displayMode: true,
    },
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
  assert.deepEqual(mathSegments("价格$100，原文保留"), [
    { type: "text", value: "价格$100，原文保留" },
  ]);
  assert.deepEqual(mathSegments(String.raw`价格\$100 和 \$200`), [
    { type: "text", value: String.raw`价格\$100 和 \$200` },
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
  for (const value of ["2026-08-06", "第1-2步", "A-B 测试", "foo-bar@example.com"]) {
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
