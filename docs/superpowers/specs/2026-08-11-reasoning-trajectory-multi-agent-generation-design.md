# AI 讲题 Demo：思维轨迹驱动的 Multi-Agent 备课与讲解生成设计

日期：2026-08-11
状态：已完成对话设计确认，待用户审阅书面规格

## 1. 决策摘要

本版本重构课程生成链，不改变学生课堂的全屏横屏 Pad 体验。

生成流程从：

```text
参考教学路线
→ Lesson Director 自由生成整篇讲解和视觉动作
→ Reviewer 整篇审核
→ 最多两轮重写
```

调整为：

```text
题目、参考答案、参考解析
→ 还原参考解法轨迹
→ 设计可往返的教学思维轨迹
→ 编写逐句讲稿
→ 设计关键互动
→ 编排语音、题目强调与板书演出
→ 学生模拟与独立整课审核
→ 按问题定向返工直至通过或明确失败
→ 编译、火山 TTS、缓存、全屏课堂
```

本设计采用一个确定性课程主编和七个 LLM 角色，分成六个生成阶段。学生模拟器与课程
审核官在最后一个审核阶段承担不同视角，不相互代写。

本设计取代旧规格中的以下决定：

- Lesson Director 同时拥有课纲、讲稿、互动和视觉设计权；
- 教学重点只在最终整篇审核中被检查；
- 讲稿和视觉动作在同一次模型输出中生成；
- Reviewer 要求修改后由 Director 整篇重写；
- 教学审核最多进行两轮。

本设计继续保留：

- 输入题目、参考答案和参考解析；
- 参考材料作为主要教学事实源；
- 确定性工具支持时进行局部数学检查，但不建设通用自动解题和批改系统；
- 选择题互动；
- 语音与板书按语义 Cue 同步；
- 火山引擎 TTS；
- 课程 ID、SQLite 持久化和已生成课程复用；
- 学生端全屏课堂，不展示内部备课文档和审核过程。

## 2. 问题定义

当前课程虽然能够生成完整语音、公式、互动和同步板书，但教学内容容易出现四类问题：

1. 模型没有先形成权威备课成果，直接从参考解析跳到完整讲稿；
2. 解题步骤被当作教学步骤，参考解析中省略的构思、条件使用和决策理由没有补出来；
3. 讲稿和视觉动作同时生成，导致高亮、板书与实际说话内容错位；
4. Reviewer 只能对成品提出整体意见，重写时容易破坏原本正确的部分。

这会造成“数学结论大致正确，但不知道该强调什么”“机械念步骤、不解释为什么”以及
“板书看似在动，却没有跟随思考”的体验。

本版本的目标不是增加更多人格化 Prompt，而是建立可追踪的备课生产线，使每个 Agent 对
一个明确的中间成果负责。

## 3. 教学质量的第一性原则

本 Demo 对好教学采用以下最高标准：

> 教师能够在解题推进的每个时刻正确指出学生应该注意和理解的内容，并带着学生经历
> “为什么这样想、为什么这样做、做完以后得到什么、下一步为什么由此产生”的过程。

数学讲解同时包含两条线：

1. **思路线**：观察条件和目标、形成想法、探索、选择方法、根据新信息调整计划；
2. **执行线**：把已经作出的数学决定正确地代入、变形、计算、检查并表达出来。

两条线不固定为前后两个章节。课程允许以下过程反复出现：

```text
观察条件与目标
→ 形成暂时想法
→ 执行一部分
→ 观察新信息
→ 继续原计划或重新构思
→ 检查与回看
```

因此，本设计不限制一道题只能有几个重点，也不要求每道题具有相同阶段。重点由每个
思维片段当前承担的教学任务自然产生。

## 4. 相关研究与案例形成的设计依据

本设计参考以下可迁移机制，而不是照搬某个产品：

- **ScaffoldLM**：从参考解法生成显式、可检查的分步教学计划，计划作为多轮教学的稳定
  骨架；去掉计划会降低指导质量和答案准确性；
- **MathTutorBench**：解题能力不能代表教学能力，教学质量需要独立评价；
- **EduPlanner**：用分析、评价和优化角色分离教案生成与审稿；
- **EduVisAgent**：把教学规划、推理分解和视觉设计交给不同职责；
- **MATHia 与 ASSISTments**：互动和提示围绕具体步骤、条件和常见错误；
- **Pólya 与 Schoenfeld 的数学问题解决研究**：理解、构思、执行、检查不是必须一次完成
  的线性流程；执行结果可以触发新的分析和计划；
- **多媒体学习原则**：只提示关键组织信息，并让对应语音和视觉在时间、空间上邻近；
- **基于 SOP 的 Multi-Agent 流水线**：角色通过明确中间成果协作，比自由讨论更容易追踪
  和降低级联错误。

本版本不引入自由辩论、投票或自组织 Agent 群。各阶段存在明确依赖，顺序流水线更符合
任务结构。只有最终的学生模拟和整课审核可以基于同一成品独立运行。

## 5. 范围

### 5.1 本版本完成

- 把参考解析还原为显式解法轨迹；
- 把解法轨迹改写为包含构思与执行的教学思维轨迹；
- 按思维轨迹生成逐句讲稿；
- 在真实思维转换点设计选择互动；
- 将题目高亮、板书和临时图层绑定到具体讲稿句子；
- 建立学生模拟和独立课程审核；
- 支持问题定位、责任路由和局部返工；
- 保存各阶段内部成果和审核报告，供开发和教研排查；
- 建立第一版教学质量黄金题集和版本比较方法。

### 5.2 本版本不做

- 通用自动解题、正式批改或全题型证明；
- 根据长期学生画像实时改写整节课；
- 学生任意开放追问后的实时重备课；
- 让学生看到内部 JSON、Agent 分歧或审核分数；
- 多个教师 Agent 自由辩论后投票；
- 用模型生成像素坐标或毫秒级时间戳；
- 声称自动审核已经证明真实学习效果；
- 为每个思维片段强制添加互动、高亮或动画。

## 6. 总体架构

```text
生成请求
  ↓
Course Orchestrator（确定性状态机）
  ↓
Reference Material Analyst
  → SolutionTrace
  ↓
Teaching Designer
  → ReasoningTrajectory
  ↓
Script Teacher
  → TeachingScript
  ↓
Interaction Designer
  → InteractionPlan
  ↓
Classroom Director
  → PerformanceScore
  ↓
┌───────────────────┬───────────────────┐
│ Student Simulator │ Lesson Reviewer   │
│ SimulationReport  │ ReviewDecision    │
└───────────────────┴───────────────────┘
  ↓
通过：Compiler → 火山 TTS → SQLite/音频缓存 → 学生课堂
修改：Orchestrator 将问题发回责任 Agent，并重建受影响的下游成果
失败：保存审核档案，向生成页返回可重试的明确阶段错误
```

所有 LLM 角色可以使用同一个 OpenAI 兼容模型服务，但必须使用独立系统指令、独立上下文
和独立结构化输出。角色分离指责任和上下文分离，不要求部署七套不同模型。

## 7. 角色与责任

### 7.1 Course Orchestrator

Course Orchestrator 是程序状态机，不负责创作教学内容。

负责：

- 调度生成阶段；
- 保存每个版本的中间成果；
- 校验 Schema、ID、引用和状态转换；
- 收集审核问题并路由给责任角色；
- 计算哪些下游成果已经失效；
- 防止审核和修订形成无进展循环；
- 发出生成阶段进度；
- 只有审核通过后才允许编译和 TTS。

Orchestrator 不能自行补写讲稿、改变数学步骤、忽略审核问题或把失败课程标为通过。

### 7.2 Reference Material Analyst

输入：题目、参考答案、参考解析、已有的参考材料验证报告。

负责：

- 忠实还原参考解析中明确写出的数学动作；
- 标出题目条件、目标和最终结论；
- 标出每个条件被使用的位置；
- 补充解析省略但讲解需要交代的局部理由；
- 区分原文事实与教学推断；
- 标记无法确认、可能跳步或表述含混的位置。

不负责：

- 把“不支持自动验证”判断为题目错误；
- 独立发明与参考资料不同的解法；
- 决定课堂怎样开场、互动或板书。

输出：`SolutionTrace`。

### 7.3 Teaching Designer

输入：`SolutionTrace`、目标年级、可用互动和板书能力。

负责：

- 判断本题属于计划明确型、探索推进型还是混合型；
- 重构学生应该经历的思维推进过程；
- 区分理解、构思、探索、执行、监控、修订和回看；
- 为每个片段确定学生当前已知、注意对象、思考问题和决策理由；
- 确定每个执行动作的依据、结果和后续意义；
- 识别参考解析中“只写了动作、没有解释选择”的位置；
- 识别适合互动、临时图层或回看题目目标的位置；
- 写出方法总结和易错认识的来源，不直接写表演台词。

Teaching Designer 可以为了教学理解插入、合并或重新组织思维片段，但必须保留
`SolutionTrace` 中的数学依赖、题目条件和最终结论。它不能为了让故事更顺畅而交换存在
先决关系的数学动作，也不能把事后才获得的信息伪装成开题时已经知道的事实。

输出：`ReasoningTrajectory`。该成果通过设计审核后冻结，后续角色只能实现，不能擅自
改变思维片段的数学目标或顺序。

### 7.4 Script Teacher

输入：冻结的 `ReasoningTrajectory`。

负责：

- 把每个思维片段写成初中生能听懂的自然口语；
- 在构思处带学生观察、比较和作出决定；
- 在执行处讲清数学动作及其依据；
- 在得到新信息后重新连接题目目标；
- 对真实探索保留必要的不确定性，但不制造虚假装傻或无意义绕路；
- 控制句长、停顿、过渡和信息密度；
- 避免互动前泄露答案；
- 结尾回收思维轨迹、方法和易错点。

不负责视觉动作和时间。输出：`TeachingScript`。

### 7.5 Interaction Designer

输入：`ReasoningTrajectory`、`TeachingScript`。

负责：

- 选择真正值得学生作出判断的时刻；
- 区分互动检查的是构思还是执行；
- 设计单选题及唯一正确项；
- 让每个错误项对应一个可解释的误解；
- 编写正确、错误和提示反馈；
- 指定作答前隐藏的信息以及作答后返回的讲稿位置。

不要求每课固定互动数量。只有存在真实判断价值时才添加互动。输出：
`InteractionPlan`。

### 7.6 Classroom Director

输入：`TeachingScript`、`InteractionPlan`、题目语义目标、支持的板书动作。

负责：

- 将讲稿句子组织为可播放的语义 Cue；
- 为具体句子绑定题目聚焦、下划线、高亮、变色、遮罩、揭示、板书写入、变形、聚焦、
  弱化、清除和临时图层；
- 设计主板书从建立、累积、聚焦到沉淀的状态变化；
- 在探索片段中使用临时图层，并明确返回主线的时刻；
- 控制屏幕信息量，不把讲稿复制为字幕墙；
- 只在具有教学信息增益时添加视觉动作。

Classroom Director 只能引用讲稿句子 ID 和服务端提供的语义目标 ID，不能生成字符偏移、
CSS 选择器、像素坐标或毫秒时间戳。输出：`PerformanceScore`。

### 7.7 Student Simulator

输入：上述全部审核候选成果，不接触最终 Reviewer 的意见。

负责模拟：

- 不知道如何开始的学生；
- 知道方法但不理解理由的学生；
- 会构思但容易算错的学生；
- 忽略题目条件的学生；
- 被典型错误选项吸引的学生；
- 只记答案、没有形成方法认识的学生。

它逐片段回答：学生此刻能否知道看什么、想什么、做什么和为什么；错误反馈能否帮助其
继续；课程结束后能否复述方法和易错条件。输出：`SimulationReport`，只提供证据和失败
位置，不改写课程。

### 7.8 Lesson Reviewer

输入：所有中间成果、确定性校验报告和 `SimulationReport`。

负责：

- 检查参考材料、思维轨迹、讲稿、互动和演出谱之间的追踪关系；
- 检查每个思维片段是否强调了学生当前必须理解的内容；
- 检查学生能否沿着构思与执行继续前进；
- 检查关键理由是否被解释、简单计算是否喧宾夺主；
- 检查互动是否诊断真实理解；
- 检查语音意图、题目强调和板书动作是否一致；
- 汇总学生模拟暴露的问题；
- 输出通过、定向修改或明确失败。

Reviewer 不直接改写任何成果。输出：`ReviewDecision`。

## 8. 权威中间成果

### 8.1 SolutionTrace

```text
SolutionTrace
  task_target
  reference_conclusion
  assumptions[]
  source_steps[]
    source_step_id
    source_anchor
    state_before
    mathematical_action
    justification
    state_after
    new_information
    assumption_ids_used[]
    omitted_reasoning[]
    evidence_status
  audit_notes[]
```

`source_anchor` 必须能够指回题目、答案或解析中的原始文本。Agent 补出的理由标为
`inferred`，不能伪装成参考解析原文。

### 8.2 ReasoningTrajectory

```text
ReasoningTrajectory
  trajectory_type: planned | exploratory | hybrid
  lesson_purpose
  episodes[]
    episode_id
    sequence_index
    mode: understand | plan | explore | execute | monitor | revise | reflect
    source_step_ids[]
    learner_state_before
    attention_targets[]
    thinking_question
    decision
    decision_reason
    mathematical_action
    action_justification
    result
    result_meaning
    transition_reason
    must_teach[]
    likely_misconceptions[]
    interaction_intent
    visual_intent
  method_summary
  error_summary
```

`episodes` 是经过教学设计的实际播放顺序，模式可以交替。首版不建设通用分支图。探索
失败或重新构思以连续片段表达，并通过 `transition_reason` 说明为什么回到新方向。

### 8.3 TeachingScript

```text
TeachingScript
  clauses[]
    clause_id
    episode_id
    pedagogical_function
    spoken_text
    math_references[]
    learner_gain
    answer_exposure
  closing_summary_clause_ids[]
```

`pedagogical_function` 至少支持：聚焦、提问、解释、决策、执行、观察结果、纠错、过渡、
回看和总结。`spoken_text` 是后续 TTS 的唯一内容来源。

### 8.4 InteractionPlan

```text
InteractionPlan
  interactions[]
    interaction_id
    episode_id
    after_clause_id
    diagnostic_target
    diagnostic_kind: conception | execution
    prompt
    options[]
      option_id
      display_text
      misconception
    correct_option_id
    correct_feedback
    incorrect_feedback_by_option
    hint
    resume_clause_id
    concealed_targets[]
```

### 8.5 PerformanceScore

```text
PerformanceScore
  cues[]
    cue_id
    clause_ids[]
    lead_actions[]
    start_actions[]
    end_actions[]
  board_objects[]
  overlay_transitions[]
```

每个视觉动作必须引用 `clause_id`、合法题目目标或已创建板书对象。模型只决定语义边界；
播放器根据 Cue 音频确定实际时间。

### 8.6 SimulationReport 与 ReviewDecision

```text
SimulationReport
  episode_results[]
    episode_id
    learner_profile
    can_identify_attention_target
    can_explain_decision
    can_execute_action
    can_use_result_to_continue
    evidence
  interaction_results[]
  end_of_lesson_recall
  blocking_findings[]

ReviewDecision
  status: approved | revision_required | failed
  findings[]
    finding_id
    severity: blocking | material | polish
    artifact_type
    artifact_id
    criterion
    evidence
    responsible_role
    requested_change
    invalidated_downstream_artifacts[]
  retained_artifacts[]
  approval_summary
```

## 9. 教学审核标准 v0.1

审核不使用一个可以相互补偿的总分。以下两个核心门槛必须同时满足：

### 9.1 当前强调正确

对每个思维片段检查：

- 此刻真正决定学生理解的对象是什么；
- 讲稿是否把注意力引向该对象；
- 是否遗漏条件、目标、方法选择或中间结果的意义；
- 是否把普通计算讲得很重，却跳过关键构思；
- 视觉是否支持当前注意对象，而非制造装饰。

### 9.2 学生能够跟着走并理解为什么

对每个思维片段检查：

- 学生是否知道当前为什么要解决这个局部问题；
- 决策理由是否出现；
- 执行动作是否有依据；
- 执行结果是否被解释；
- 结果与下一步之间是否有可理解的过渡；
- 如果发生探索或返工，学生是否知道为什么调整方向。

### 9.3 其他硬门槛

- 与题目、参考答案和冻结的参考结论一致；
- 每个 `must_teach` 至少映射到一条讲稿句子；
- 每个关键构思至少有一个显式理由；若属于目标年级已经掌握且本题没有教学增益的常规
  动作，Reviewer 必须记录豁免理由和前文已解释的句子引用；
- 互动前不泄露所检查的答案；
- 选择题只有一个正确项，每个错误项具有明确误解含义；
- 每个视觉动作绑定具体讲稿句子和合法语义目标；
- 新板书不在对应讲稿之前泄露；
- 临时图层能够确定性返回主板书；
- 结尾能够回收方法、条件或易错认识；
- 公式文本通过现有数学文本规范化与 KaTeX 渲染检查。

### 9.4 质量观察维度

以下维度用于审稿证据和版本比较，不得弥补核心门槛失败：

- 语言是否自然、年龄适配；
- 信息密度和节奏是否合理；
- 互动是否产生真实思考；
- 错误反馈是否帮助学生重新进入轨迹；
- 板书是否逐渐形成可回看的结构；
- 总结是否支持同类题迁移。

Reviewer 必须引用具体片段、句子或视觉动作作为证据，不能只输出“清晰度 8 分”一类
抽象分数。

## 10. 审核、返工与收敛

### 10.1 分层审核

生成过程设置四个质量边界：

1. `SolutionTrace` 完成后检查参考忠实度和步骤连接；
2. `ReasoningTrajectory` 完成后检查思维轨迹是否包含必要构思与执行；
3. `TeachingScript` 和 `InteractionPlan` 完成后检查讲解覆盖与答案泄露；
4. `PerformanceScore` 完成后进行学生模拟和整课审核。

早期发现的问题在上游修复，避免等到 TTS 完成后整课重做。

### 10.2 定向返工

每个审核问题必须包含责任角色和失效范围。例如：

```text
episode_04 直接从展开结果进入约分，缺少观察公因式和使用 n != 0 的决策理由。
责任角色：Teaching Designer
保留：SolutionTrace、episode_01..03
重建：episode_04 以后受影响的 TeachingScript、InteractionPlan、PerformanceScore
```

Orchestrator 只重建被标记失效的成果。未受影响且通过审核的内容保持原版本。

### 10.3 不按固定两轮通过

教学审核没有“两轮后自动接受”的规则。课程只有在所有 blocking 和 material 问题解决后
才能进入编译。

系统通过以下机制防止无限循环：

- 同一问题每轮记录前后证据；
- 连续两次修订未改变同一问题时，切换新的 Reviewer 上下文并重新诊断一次；
- 新 Reviewer 仍判断无进展时，课程以 `review_not_converged` 失败，不生成低质量课堂；
- 单次课程设置八次定向修订的运行预算；预算耗尽只触发明确失败，不触发降级通过；
- 运行预算是 API 成本与等待时间保护，不是教学质量标准，可以通过配置调整。

### 10.4 编译后体验检查

内容审核通过后，Compiler 和 TTS 仍需执行体验检查：

- 所有 Cue 都有口语或明确静默意图；
- 音频齐全且可播放；
- Cue 顺序覆盖全部讲稿句子；
- lead/start/end 动作引用合法；
- 公式渲染无错误；
- 暂停、重播、互动阻塞和临时图层返回保持确定性。

编译后检查只修复技术表达问题；如果发现教学内容需要变化，必须退回相应 Agent 并使下游
音频失效。

## 11. 与现有 Runtime 的集成

### 11.1 Beat 与 Episode

`ReasoningEpisode` 是备课单位，`RuntimeBeat` 是课堂交互和导航单位。Compiler 可以把一个
Episode 编译为一个 Beat，也可以把紧密相连、无需互动的多个短 Episode 合并为一个 Beat。

合并不能丢失 Episode ID，公开调试元数据应保留 `episode_ids` 以便排查。

### 11.2 Script Clause 与 Sync Cue

`TeachingScript.clauses[].spoken_text` 是语音权威源。Classroom Director 只能将相邻句子
组合为 Cue，不能修改文字。Compiler 从 Cue 生成现有 `RuntimeSyncCue`，火山 TTS 按 Cue
生成音频。

### 11.3 视觉动作

继续使用现有语义动作和生命周期：

- problem：聚焦、高亮、下划线、变红、遮罩、揭示、清除聚焦；
- board：写入、变形、聚焦、标注、弱化、清除；
- overlay：进入临时图层、写入、返回主板书。

讲过的普通内容保持原状；只有明确强调过的重点在离开当前 Cue 后按其样式弱化。

### 11.4 已保存课程

已保存的旧课程继续按现有 Runtime 格式播放，不回填新中间成果。新生成课程在服务端数据库
中保存最终 RuntimeLesson，同时在内部生成记录中保存设计和审核成果。学生公开 API 继续
隐藏正确选项、私有反馈和内部审核内容。

## 12. 用户体验

生成页仍然是一次提交，不要求老师确认课纲。进度文案调整为：

```text
整理参考解析
→ 设计解题思维轨迹
→ 编写讲稿
→ 设计互动
→ 编排板书与高亮
→ 模拟学生并审核课程
→ 生成语音
→ 保存课程
```

审核返工时可以保持“正在审核和优化课程”，不向普通用户展示 Agent 内部争议。开发模式
可以查看每个阶段版本、审核问题和返工历史。

生成成功后继续显示课程 ID、复制 ID、进入课堂和继续生成。不会自动跳转。

## 13. 失败处理与可观测性

每个模型调用记录：

- lesson ID 和 generation ID；
- role；
- input artifact versions；
- output artifact version；
- duration；
- retry count；
- structured failure category；
- token usage（提供时）；
- review finding IDs（修订调用）。

不记录 API Key、访问令牌或学生私有信息。

失败分类至少包括：

- `provider_error`：模型或 TTS 服务不可用；
- `invalid_structure`：模型输出无法通过 Schema；
- `reference_trace_failed`：无法忠实整理参考解析；
- `reasoning_design_failed`：无法形成连贯思维轨迹；
- `review_not_converged`：审核与修订没有收敛；
- `compile_failed`：引用、Cue 或视觉状态无法编译；
- `tts_failed`：必要音频缺失；
- `persistence_failed`：课程无法完整保存。

Schema 失败可以对当前角色重试一次；内容问题必须进入审核修订，不能用结构重试掩盖。
任何阶段失败都不会覆盖同一 lesson ID 下已经完整保存的旧课程。

## 14. 教学标准的持续迭代

### 14.1 版本化标准

系统维护独立的 `Pedagogy Rubric` 版本。生成记录保存所用版本。标准包含：

- 核心门槛；
- 每种思维模式的审查问题；
- 正例和反例；
- 常见失败模式；
- 教师反馈沉淀出的可复用规则。

Prompt 修改不能静默改变教学标准；标准变更需要新版本和黄金题集比较。

### 14.2 第一版黄金题集

建立 18 道人工审阅的初中数学单题，覆盖：

- 概念条件转换；
- 代数运算；
- 方程与参数；
- 方法选择；
- 几何文字推理（无图）；
- 函数关系；
- 容易遗漏条件；
- 需要探索或重新构思；
- 适合临时图层解释概念；
- 不适合强加互动或视觉动作。

每道题保存人工认可的：

- `SolutionTrace` 关键锚点；
- `ReasoningTrajectory` 必须包含的构思与执行片段；
- 必须讲清的理由；
- 典型误解；
- 关键板书状态；
- 可接受和不可接受的讲解片段。

### 14.3 新旧版本比较

每个候选版本对 18 道题各生成三次，降低单次随机性的影响。比较：

- 核心门槛通过率；
- 必讲内容覆盖率；
- 讲稿句子与视觉动作有效绑定率；
- 结构、公式和 Runtime 技术通过率；
- 教师盲评的成对偏好；
- 生成成功率、时长和调用次数。

教师比较优先回答：

- 哪一版更准确地知道每一步应该强调什么；
- 哪一版更能让学生跟上并理解为什么；
- 哪一版的板书更帮助思考；
- 哪一版的互动更能诊断理解。

自动审核通过、教师认可和真实学生学会是三个不同证据层级。Demo 阶段验收前两层，不把
它们表述为已经证明学习效果。

## 15. 测试与验收

### 15.1 单元和契约测试

- 每个中间成果的 Schema、ID 和引用完整性；
- `must_teach → clause_id` 覆盖关系；
- `clause_id → cue/action` 合法关系；
- 互动唯一正确项、反馈完整性和公开载荷隐私；
- 返工失效范围和保留范围；
- 无进展检测和 `review_not_converged`；
- 旧 RuntimeLesson 向后兼容；
- 数学文本规范化和 KaTeX 渲染；
- 火山 TTS Cue 与音频资产完整性。

### 15.2 代表题验收

使用参数根示例：

```text
若 2n（n != 0）是关于 x 的方程 x^2 - 2mx + 2n = 0 的根，求 m - n。
```

课程必须呈现：

1. “是根”意味着将 `x = 2n` 代回方程；
2. 先根据题目目标构思：不必分别求出 `m,n`，需要得到二者关系；
3. 正确执行代入和展开；
4. 从展开结果观察公因式，并在约分时回看 `n != 0`；
5. 得到新关系后重新看目标，整理为 `m - n = 1/2`；
6. 语音说到相应条件或式子时，题目强调和板书才出现；
7. 互动检查至少一个真实决策或条件使用，不复述刚刚泄露的答案；
8. 结尾说明“已知某式是根先代回；约去含字母因子前确认非零”。

不能接受：

- 只朗读四行参考解析；
- 详细讲平方计算，却不解释为什么代入；
- 直接除以 `n` 而不指出非零条件；
- 从中间式直接跳到答案，不重新连接题目目标；
- 在语音之前显示完整后续推导；
- 为了制造效果圈住屏幕上唯一的大公式；
- 选择项包含未规范化、导致公式渲染警告的混合文本。

### 15.3 浏览器体验验收

- 横屏 Pad 全屏课堂保持不变；
- 真实火山语音可播放；
- 暂停、继续、重播和返回上一 Beat 正确恢复状态；
- 题目高亮与对应语句同步；
- 板书关键步骤随语音出现并保持可回看；
- 临时图层可进入并返回；
- 选择互动阻止自动推进，正确和错误反馈可听；
- 保存课程重启服务后仍可通过 ID 打开。

## 16. 实施边界与迁移顺序

实施计划应按以下顺序落地，避免一次重写全部 Runtime：

1. 新增中间成果模型和持久化的内部生成记录；
2. 实现 Reference Material Analyst 与 Teaching Designer；
3. 将现有 Narrative 生成替换为 Script Teacher；
4. 将互动素材生成迁移到 Interaction Designer；
5. 让 Classroom Director 从句子 ID 生成现有 SyncCue 和语义动作；
6. 引入 Student Simulator、Lesson Reviewer 和定向返工；
7. 接回现有 Compiler、火山 TTS、SQLite 课程缓存和前端 Runtime；
8. 建立黄金题集、离线比较工具和浏览器验收。

在新链路通过代表题之前，保留旧课程读取和播放能力。新生成 API 的公开输入、课程 ID 和
学生课堂 URL 保持不变，降低演示迁移风险。

## 17. 最终决策

- 采用顺序、可追踪的 Multi-Agent 备课流水线；
- 不采用自由讨论或投票式 Agent 群；
- 用可交替的思维片段表达构思、探索、执行和回看；
- 不限制一道题的重点数量；
- 讲稿、互动和视觉分别由不同角色完成；
- 视觉动作必须绑定具体讲稿句子；
- Reviewer 不代写，只给有证据、可路由的修改单；
- 不按两轮自动通过，未收敛时明确失败；
- “每一步强调正确”和“学生能跟上并理解为什么”是不可相互补偿的核心门槛；
- 教学标准通过版本化量表、黄金题集和教师成对比较持续迭代。

## 18. 调研来源

- ScaffoldLM：<https://aclanthology.org/2026.acl-long.325/>
- MathTutorBench：<https://aclanthology.org/2025.emnlp-main.11/>
- EduPlanner：<https://arxiv.org/abs/2504.05370>
- EduVisAgent：<https://arxiv.org/abs/2505.16832>
- MATHia 学生支持机制：
  <https://support.carnegielearning.com/help-center/math/mathia/getting-started-in-mathia/article/getting-started-mathia-students/>
- ASSISTments 研究：<https://www.assistments.org/research>
- Pólya 问题解决框架：<https://www.utsc.utoronto.ca/learningstrategies/polyas-problem-solving>
- Schoenfeld 问题解决阶段相关研究：
  <https://www.sciencedirect.com/science/article/pii/S0360131505001806>
- Productive Failure：<https://onlinelibrary.wiley.com/doi/10.1111/cogs.12107>
- 多媒体学习的信号与邻近原则：
  <https://www.cambridge.org/core/books/abs/cambridge-handbook-of-multimedia-learning/principles-for-reducing-extraneous-processing-in-multimedia-learning/F29A19FCD34C542806F736E0661C05F5>
- MetaGPT 的 SOP Multi-Agent 流水线：<https://arxiv.org/abs/2308.00352>
- Anthropic Orchestrator-Worker 生产实践：
  <https://www.anthropic.com/engineering/multi-agent-research-system>
