# AI 讲题 Demo：Sync Cue 语音—板书同步设计

日期：2026-08-07  
状态：已确认，待实施计划

## 1. 背景

当前播放器把一个 Teaching Beat 内的 `board_actions` 按音频总时长平均分配。
这种调度只知道“本段有几个动作”，不知道教师在音频中何时说到某个条件、关键式或
推理依据，因此会出现以下问题：

- 原题高亮与教师正在说的内容不一致；
- 关键式可能在解释之前或之后出现；
- 板书看似在播放动画，但没有承担“让学生看见推理正在发生”的作用；
- 同一 Beat 内的多个动作缺少明确的语义边界；
- 暂停、重播和降级播放只能保持粗粒度顺序，不能恢复精确的教学节奏。

本设计把 Beat 保留为一个完整认知目标，在 Beat 内增加顺序明确的 `SyncCue`。
每个 Cue 同时描述一小段口语、题目区域动作、主板书动作和重点生命周期。火山 TTS
按 Cue 生成音频，播放器以 Cue 边界作为确定的动作时机。

## 2. 目标

本版本需要做到：

1. 语音说到关键条件、关键操作或关键结果时，对应题目高亮或板书同步出现；
2. 原题区域与主板书使用同一条 Cue 时间线；
3. 现有内容的聚焦可以在语音前约 0.2 秒出现，新板书在 Cue 音频开始时出现；
4. 普通板书保持原样，只有明确划过的重点在离开当前 Cue 后弱化保留；
5. 不要求每个 Beat 都有高亮或板书动作；
6. 暂停、重播、互动阻塞和音频失败回退保持确定性；
7. 旧课程没有 `sync_cues` 时仍可播放。

## 3. 非目标

本版本不做：

- 逐字字幕式高亮；
- 模型生成像素坐标、CSS 选择器或字符串偏移；
- 依赖额外语音识别服务做整段音频强制对齐；
- 任意手写笔迹模拟；
- 为每个 Beat 或 Cue 强制制造视觉动作；
- 在本次改动中重新设计互动题、近迁移题或数学验证范围。

## 4. 核心决策

### 4.1 Beat 与 Cue 的职责

`RuntimeBeat` 继续表示一个完整认知目标，例如“将已知根代回原方程”。互动仍然是
Beat 结束后的教学边界。

`SyncCue` 表示 Beat 内一个可以独立说清的语义片段，例如：

1. “因为 \(2n\) 是方程的根”；
2. “所以把 \(x=2n\) 代入原方程”；
3. “代入后得到 \(4n^2-4mn+2n=0\)”；
4. “因为 \(n\ne0\)，等式两边可以除以 \(n\)”。

每个 Beat 至少有一个 Cue。Cue 可以没有任何视觉动作。开场、过渡、总结通常只有
口语；只有具有教学信息增益的条件、操作、关键式和依据才触发画面变化。

### 4.2 不让模型生成时间戳

模型负责划分语义 Cue、选择语义目标和受限动作。系统负责：

- Cue 的播放顺序；
- 聚焦提前量；
- 颜色、动画和弱化样式；
- TTS 调用和音频 URL；
- 题目与板书对象的实际 DOM；
- 暂停、重播、回退和互动阻塞。

Cue 的音频边界就是动作边界，因此不需要模型猜测毫秒时间，也不需要按动作数量
平均分配时间。

## 5. 数据契约

### 5.1 NarrativeSyncCue

Lesson Director 输出的教学主线新增：

```text
NarrativeSyncCue
  cue_id: string
  spoken_text: string
  lead_actions: VisualAction[]
  start_actions: VisualAction[]
  end_actions: VisualAction[]
```

字段语义：

- `spoken_text`：本 Cue 唯一口语来源；
- `lead_actions`：在音频前约 0.2 秒执行，只能聚焦已经存在的题目或板书对象；
- `start_actions`：音频开始时执行，可以书写、变形、聚焦或添加语义标记；
- `end_actions`：音频结束时执行，只能取消当前聚焦，或弱化本 Cue 明确划过的重点。

Beat 的完整旁白由所有 Cue 的 `spoken_text` 按顺序拼接生成，不再同时维护一份独立、
可漂移的 Beat 旁白。公开载荷可以保留派生后的 `narration` 作为无障碍文本和旧播放器
兼容字段，但它不是生成权威源。

### 5.2 RuntimeSyncCue

音频生成完成后，运行时使用：

```text
RuntimeSyncCue
  cue_id: string
  spoken_text: string
  lead_actions: VisualAction[]
  start_actions: VisualAction[]
  end_actions: VisualAction[]
  audio_url: string | null
```

`RuntimeBeat` 新增 `sync_cues`。Beat 级 `audio_url` 在新课程中不再是权威播放源；
旧课程仍可继续使用。

### 5.3 VisualAction

Cue 动作同时覆盖题目与板书两个表面：

```text
VisualAction
  surface: "problem" | "board"
  type: allowed action type
  target: semantic target id
  content: optional math or text
  emphasis_style: "highlight" | "underline" | "red" | null
  persistence: "transient" | "trace" | null
```

动作继续使用白名单。第一版需要支持：

- `write`
- `transform`
- `focus`
- `emphasize`
- `fade`
- `reveal`
- `annotate`
- `clear_focus`

`emphasis_style` 只表达教学语义，实际颜色和视觉强度由播放器主题决定。

### 5.4 题目语义目标

模型不能输出字符偏移。生成前，服务端把题目拆成稳定的 `ProblemFocusTarget`：

```text
ProblemFocusTarget
  target_id
  display_text
  kind: "math" | "condition" | "target"
```

第一版优先为明确的数学片段、条件和待求量建立目标，例如：

- `problem-root-expression` → \(2n\)
- `problem-nonzero-condition` → \(n\ne0\)
- `problem-equation` → \(x^2-2mx+2n=0\)
- `problem-target` → \(m-n\)

这些目标作为只读上下文交给 Lesson Director。模型只能引用 `target_id`，播放器根据
服务端编译结果渲染相应范围。

主板书对象继续由先前 `write` 或 `transform` 动作建立稳定 ID，后续 Cue 可以聚焦、
标记或弱化这些对象。

## 6. 生成与编译流程

完整流程为：

```text
题目、答案、解析
→ 冻结 Teaching Route
→ 编译 ProblemFocusTarget
→ Lesson Director 生成 Beat 与 NarrativeSyncCue
→ 结构和教学质量校验
→ Materials Agent 生成互动与近迁移
→ Reviewer 整篇审稿
→ Lesson Compiler 编译 RuntimeBeat 与 RuntimeSyncCue
→ 火山 TTS 按 Cue 生成音频
→ Player 预加载并顺序播放
```

Lesson Director 的提示词必须强调：

- 先写自然、完整、学生能听懂的 Cue 口语；
- 只在具有教学信息增益时添加视觉动作；
- 关键步骤的 `write` 或 `transform` 与说出该步骤的 Cue 绑定；
- 不为了满足数量制造圈注、高亮或动画；
- 不修改冻结 Teaching Route 的步骤、条件和结论。

## 7. 播放时序

### 7.1 Cue 开始

播放器准备播放 Cue 时：

1. 确认当前 Beat、Cue 和播放 token 仍然有效；
2. 执行 `lead_actions`；
3. 等待默认 200ms；
4. 开始播放 Cue 音频；
5. 在音频成功开始后立即执行 `start_actions`。

现有内容的高亮或聚焦可以稍早于口语，让学生先知道该看哪里。新的关键式不能提前
泄露，必须在对应口语开始时出现。

### 7.2 Cue 结束

音频结束时：

1. 执行 `end_actions`；
2. 取消当前临时聚焦；
3. 仅将 `persistence=trace` 的明确重点弱化保留；
4. 普通书写和未划过的内容保持原样；
5. 预加载并进入下一个 Cue。

弱化表示视觉层级下降，不统一变成绿色。下划线、红色和高亮分别使用各自的弱化样式。

### 7.3 Beat 结束

全部 Cue 完成后：

- 有互动时显示互动并阻止进入下一 Beat；
- 无互动时开放下一 Beat；
- 临时图层按现有规则回到主板书快照；
- Beat 级重播从进入 Beat 前的快照重新开始。

## 8. 暂停、重播与导航

### 8.1 暂停

暂停必须同时停止：

- 当前 Cue 音频；
- 尚未触发的 200ms lead timer；
- Cue 切换；
- 视觉动画时间线。

继续播放时从相同 Cue 和相同音频位置恢复。

### 8.2 重播

重播当前 Beat 时：

1. 停止当前音频和所有定时器；
2. 恢复进入 Beat 前的主板书快照；
3. 清除本 Beat 产生的题目临时聚焦；
4. 从 Cue 1 重新播放。

这样不会出现音频从头开始、板书却已经显示最终结果的状态。

### 8.3 上一段和下一段

导航继续以 Beat 为单位，不暴露 Cue 级按钮。Cue 是内部同步单位，不增加学生的操作
复杂度。

## 9. 火山 TTS 策略

服务端为每个 Cue 单独请求火山 TTS：

- 使用相同音色、语速和采样率；
- 对 Cue 请求做有限并发，避免一次课程触发无界请求；
- 预加载下一 Cue 音频；
- 单个 Cue 失败后进行有限重试；
- 所有 Cue 成功后才把课程标记为完成。

Cue 不应短到只包含一个词，也不应长到包含多个不同认知动作。提示词和 Schema 共同
限制 Cue 数量与文本长度，目标是每个 Beat 2–5 个 Cue；纯口语 Beat 可以只有一个 Cue。

第一版允许预加载音频片段后顺序播放。验收需要检查片段间没有明显影响理解的空白。
如果真实体验出现明显断裂，再单独设计服务端音频拼接；本版本不提前引入音频处理依赖。

## 10. 失败回退

### 10.1 生成时

- Cue Schema 无效：向 Lesson Director 返回安全、结构化的修复信息并有限重试；
- Cue 与 Teaching Route 不一致：拒绝该讲稿并重写；
- 单个 Cue TTS 失败：有限重试；
- TTS 重试耗尽：课程生成失败，不交付缺少关键语音的半成品；
- 不允许新课程静默回退到“按动作数量平均分配音频时长”。

### 10.2 播放时

课程生成时已经确认音频存在，但浏览器仍可能临时加载失败。此时：

- 显示该 Cue 的 `spoken_text`；
- 按该文本估算持续时间；
- 保持 Cue 的动作顺序；
- 提示本段语音不可用；
- 不改变答案、互动或 Teaching Route。

这种运行时降级只保证课程仍可阅读，不能被报告为语音同步成功。

## 11. 质量门

### 11.1 结构硬门

- 每个 Beat 至少一个 Cue；
- `cue_id` 全课唯一；
- `spoken_text` 非空并满足长度预算；
- 动作类型和表面合法；
- `lead_actions` 只能引用已存在目标；
- `end_actions` 只能清理或弱化本 Cue 明确聚焦、强调过的目标；
- 所有 `target_id` 可以被服务端解析；
- 公开载荷继续隐藏答案键和内部验证信息。

### 11.2 教学硬门

- Teaching Route 的每个关键步骤都由某个 Cue 的 `write` 或 `transform` 正向呈现；
- 关键式出现的 Cue 同时说出该式或其直接推导意义；
- 题目条件的高亮发生在使用该条件的 Cue；
- 互动前不能通过 Cue 口语或板书泄露正确选项；
- 不要求每个 Beat 高亮；
- 单对象画面不能无意义地圈住整个对象；
- 不允许为了动画数量制造没有信息增益的标记。

### 11.3 Reviewer

Reviewer 以整课为单位检查：

- Cue 拼接后口语是否自然连贯；
- 题目聚焦、板书书写与口语是否指向同一语义；
- 重点是否有层级；
- 关键依据是否在学生需要时出现；
- 临时图层是否帮助理解并回到主线。

Reviewer 不能要求每个 Beat 或 Cue 都增加高亮。

## 12. 兼容策略

旧课程没有 `sync_cues` 时：

- 播放器把 Beat 的原 `narration` 包装为一个兼容 Cue；
- 原 `board_actions` 作为该 Cue 的 `start_actions`；
- 原 Beat 音频继续使用；
- 保留现有播放能力，但不声称达到新同步质量。

新课程只从 `sync_cues` 生成口语和音频，避免两套权威状态。

## 13. 测试策略

### 13.1 Python

- Schema：Cue 数量、ID、动作表面、引用目标和生命周期；
- Compiler：NarrativeSyncCue 到 RuntimeSyncCue 的无损编译；
- Audio Service：Cue 级 TTS、有限并发、重试和 URL 绑定；
- Generation：Director、Materials、Reviewer 调用顺序和冻结路线不漂移；
- Compatibility：旧 LessonDraft 仍能编译和播放。

### 13.2 JavaScript

- Cue 调度不再调用均匀动作分配；
- lead、start、end 三阶段顺序；
- 暂停会冻结音频、lead timer 和 Cue 切换；
- 重播恢复 Beat 前快照；
- 只弱化明确标记的重点；
- 普通板书不变色；
- 互动只在全部 Cue 完成后出现；
- 运行时音频失败进入可读降级。

### 13.3 浏览器验收

固定使用以下参数根题：

```text
题目：
若 2n（n≠0）是关于 x 的方程 x²−2mx+2n=0 的根，则 m−n 的值为

答案：
1/2

解析：
因为 2n（n≠0）是方程的解
所以 4n²−4mn+2n=0
所以 4n−4m+2=0
所以 m−n=1/2
```

必须观察到：

1. 说到“\(2n\) 是方程的根”时，题目中的 \(2n\) 条件被聚焦；
2. 说到“把 \(x=2n\) 代入”时，主板书出现 \(x=2n\) 和代入标记；
3. 说到结果式时，出现 \(4n^2-4mn+2n=0\)；
4. 说到 \(n\ne0\) 时，题目条件被聚焦，板书出现除以 \(n\) 的变形；
5. 当前重点样式明显，已划重点在离开 Cue 后弱化，普通板书不变；
6. 所有 Cue 火山音频可播放；
7. 暂停、继续、重播和互动阻塞保持同步；
8. 横屏 Pad 页面没有侧栏，不退化为左右分栏讲解。

## 14. 交付边界

本功能完成可以证明：

- 模型能生成结构化语义 Cue；
- 火山 TTS 音频与 Cue 边界绑定；
- 原题聚焦和主板书动作沿同一时间线执行；
- 固定验收题在浏览器中呈现可复现的同步效果。

它不能证明：

- 所有题型都能生成高质量 Cue；
- Cue 数量和重点选择已经经过教师一致性校准；
- 学生理解、迁移或保持效果得到提升；
- 不同网络和设备上的音频间隙始终不可感知。

这些需要后续题型样本、教师评审、稳定性测试和学生证据。
