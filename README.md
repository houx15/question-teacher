# 无图初中数学 AI 讲题 Demo

这是一个面向现场演示的教学运行时：输入无图初中数学题目、简短参考答案和可选的多段参考解析，系统以这三项参考材料依据组织教学路线，经过整篇审稿和修订后生成语音、板书和选择式互动，最后进入横屏全屏课堂。这个 Demo 用于展示参考材料依据的 AI 讲解能达到什么质量，不是自动批改系统，也不承担通用自动解题。课程完成后会进行 SQLite 持久化，可以用页面显示的课程 ID 在服务重启后恢复；正在运行的生成任务仍只保存在内存。

## 支持范围

当前的产品范围是“有参考答案，可选有题库解析”的无图初中数学讲解。质量机制分为两条路径：

- 受支持的一元一次和一元二次方程继续执行严格符号校验，包括答案、关键等价变形和数学路线；
- 超出当前符号工具能力的题目使用参考答案与解析生成结构化教学路线，再经过结构化模型审阅和有边界的局部检查；
- 工具报告 `unsupported` 或模型请求的局部检查 `failed` 都只形成 warning，不会单独阻断讲解；只有安全/格式输入无效，或严格符号路径重现了数学矛盾时才阻断；
- `symbolic_verified`、`model_cross_checked` 和 `reference_grounded` 等验证模式及其检查证据仅保留在服务端，不进入学生页面或公开课程 API。

课堂展示与互动当前支持：

- 一元一次方程；
- 一元二次方程；
- 因式分解法、公式法和配方法；
- 指定方法，或由服务端验证过的数学路线稳定确定方法；
- 精简、标准两种讲解长度；
- 新课程中可自动评分的数学互动：`choice`，以及选择式近迁移互动；
- 运行时支持 `free_text`；API 对它始终返回 `needs_review`，不自动评分；
- 本地 KaTeX 公式渲染（JavaScript、CSS 与字体均随应用发布）。

每一节新课程会先用一个独立的“先认识方法”节拍说明接下来要用
什么方法、目标形式和它为何有用，再进入具体运算。配方法课程的该节拍
固定强调“今天用配方法”，首个板书动作写出“配方法”；它的目的不是抢先
展示计算，而是让学生先获得接下来变形的观察框架。

新生成的可自动评分数学互动由确定性门禁限制为 `choice`。`point_select`、
表达式输入、`free_text` 和旧式 `transfer` 互动仅为读取已有课程的兼容类型，
生成链路不会再产出它们；`free_text` 在运行时固定返回 `needs_review`。
确定性门禁负责结构、类型与数学硬校验，不能替代对 prompt 的语义判断。
近迁移也以三到四个带公式显示标签的选择项呈现；正确答案和选项的内部
判定值不会发送给课堂浏览器。

每个选择项都有针对该选项的诊断反馈：页面显示反馈，同时播放相同反馈的
语音。这样学生选择错误后得到的是可定位的提示，而不是只看到“错误”。
Lesson Director 负责教学主线，Materials Agent 负责选择题与近迁移素材，
Reviewer 只审查服务端合成并通过硬质量门后的完整讲稿。

在严格符号校验路径中，数学路线优先由服务端确定性规划器生成。目前它覆盖配方法的安全首一二次方程、
公式法二次方程、未指定方法的二次方程，以及未指定方法的基础一次方程。确定性
规划器返回 `unsupported` 时才调用 Math Route Agent；例如当前的因式分解路线
需要尚未纳入操作词汇的零积求根过渡，因此不会伪造该过渡。无论路线来自确定性
规划器还是 Agent，所有步骤仍逐步通过同一个 Math Engine 硬校验，并在进入
Lesson Director 前冻结和生成指纹。验证报告用 `math_route_source` 区分
`deterministic` 与 `agent`，但两者的数学状态都只有通过硬校验后才是 `verified`。
因式分解路线明确以经验证的因式乘积方程为终态；Lesson Director 再用零乘积
性质和独立校验的全部根完成教学解释，Reviewer 专门检查这一段是否漏根或混用方法。

当前不处理需要图形或函数图像作为输入的题目，也不包含账号与学习历史、数字人或 MP4 合成。严格符号检查仍只接受受限的实数一元代数表达式和有限解集；超出这个工具范围不等于已证明题目或解析正确。

### 输入格式与参数根示例

HTTP `POST /api/lessons/generate` 的三个核心字段是：

| 字段 | 含义 |
| --- | --- |
| `problem_text` | 题目原文，可以包含 `$...$` LaTeX |
| `reference_answer` | 简短参考答案，必填 |
| `reference_solution_text` | 题库提供的可选多段解析；建议保留条件、中间式和最终结论 |

例如，当已知根是含参数的式子时，可以输入：

```json
{
  "problem_text": "若$2n$ ($n\\ne 0$)是关于x的方程 $x^2-2mx+2n=0$的根，则$m-n$的值为",
  "reference_answer": "$\\frac{1}{2}$",
  "reference_solution_text": "因为 $2n$ 是方程的根，所以 $4n^2-4mn+2n=0$\n因为 $n\\ne0$，所以 $4n-4m+2=0$\n所以 $m-n=\\frac{1}{2}$"
}
```

这类题应先讲清“根代回原方程后等式成立”，再展示代入 `2n`、提取 `2n`、使用 `n\ne0` 和得到 `m-n=\frac12` 的板书。它会进入参考材料依据的教学路线，不要求本地一元方程引擎先完整求解它。

## 运行环境

项目按 Python 3.9 和 `general` conda 环境验证。首次安装与启动使用以下命令：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env` 后，将变量导出到当前 shell，再启动服务：

```bash
set -a
source .env
set +a
python -m uvicorn app.main:app --reload
```

打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)。`.env` 只用于本机配置，不要提交真实密钥。

## 文本与语音服务配置

| 变量 | 必需 | 用途 |
| --- | --- | --- |
| `OPENAI_BASE_URL` | 是 | Chat Completions 服务根地址，例如 `https://api.openai.com/v1`；程序追加 `/chat/completions` |
| `OPENAI_API_KEY` | 是 | 文本模型鉴权密钥 |
| `OPENAI_MODEL` | 是 | 生成与整篇审稿使用的模型名 |
| `OPENAI_TIMEOUT_SECONDS` | 否 | 正数超时秒数，默认 `90` |
| `TTS_PROVIDER` | 否 | `volcengine` 或 `openai_compatible`；未设置时兼容原有 OpenAI 语音配置，`.env.example` 默认使用 `volcengine` |

### 火山引擎豆包语音 V3

Demo 使用 HTTP Chunked 单向流式 V3 接口。先在豆包语音控制台开通对应资源并取得有权限的音色 ID；`.env.example` 中的音色只是占位符，不代表已经授权。

| 变量 | 必需 | 用途 |
| --- | --- | --- |
| `VOLCENGINE_TTS_ENDPOINT` | 否 | 默认 `https://openspeech.bytedance.com/api/v3/tts/unidirectional` |
| `VOLCENGINE_TTS_API_KEY` | 是 | 控制台生成的 API Key；V3 只使用 `X-Api-Key`，绝不会继承 `OPENAI_API_KEY` |
| `VOLCENGINE_TTS_RESOURCE_ID` | 是 | 已开通的资源 ID；示例为 `seed-tts-2.0` |
| `VOLCENGINE_TTS_VOICE` | 是 | 该资源下已授权的音色 ID |
| `VOLCENGINE_TTS_SPEED_RATIO` | 否 | 对外语速倍率，范围 `0.5`–`2.0`，默认 `1.0`；请求时转换为 V3 的 `speech_rate`（`-50`–`100`） |
| `VOLCENGINE_TTS_SAMPLE_RATE` | 否 | MP3 采样率，只支持 `8000`、`16000`、`24000`，默认 `24000` |
| `VOLCENGINE_TTS_UID` | 否 | 非空请求用户标识，默认 `ai-math-demo`；生产环境应改为可追踪且不含敏感信息的标识 |

参考火山引擎官方的[语音合成大模型 API 列表](https://www.volcengine.com/docs/6561/2228192?lang=zh)确认 V3 接口，并通过[大模型音色列表 API](https://api.volcengine.com/api-docs/view?action=ListSpeakers&serviceCode=speech_saas_prod&version=2025-05-20)核对资源与音色。开通资源、鉴权信息和音色授权需要在当前火山引擎账号下完成。

### OpenAI 兼容语音回退

将 `TTS_PROVIDER=openai_compatible` 后使用以下变量：

| 变量 | 必需 | 用途 |
| --- | --- | --- |
| `TTS_BASE_URL` | 否 | OpenAI 兼容语音服务根地址；为空时继承 `OPENAI_BASE_URL`，程序追加 `/audio/speech` |
| `TTS_API_KEY` | 否 | 语音服务密钥；只有语音与文本端点相同时才允许继承 `OPENAI_API_KEY` |
| `TTS_MODEL` | 是 | 语音模型名 |
| `TTS_VOICE` | 是 | 语音音色名 |

`GET /api/health` 只返回文本模型和语音是否配置完成的布尔值，不返回密钥、模型请求或响应正文。浏览器页面也不接收这些服务端配置。

## 从输入到课堂

生成链路如下：

```text
题目、简短参考答案、可选多段参考解析、指定方法与讲解长度
→ Capability Probe 识别能否进入严格符号校验
→ 受支持方程：Math Engine 独立验证答案与关键变形，然后冻结 symbolic route
→ 其他题目：模型把参考材料整理为结构化 grounding brief，局部 Claim Checker 只检查它能安全处理的声明
→ grounding 局部检查用于提升置信度或形成 warning，不承担自动批改和阻断权
→ 审查结果为 consistent 或 warning 时冻结 teaching route
→ 将已解析的方法族及展示名传给 Lesson Director 和 Materials Agent
→ Lesson Director 生成教学主线、板书和 1–3 个互动意图
→ schema 与已冻结路线的教学一致性校验
→ Materials Agent 为已声明的互动意图生成选择题，并生成近迁移题
→ 服务端按稳定 moment_id 合成完整讲解并运行全部硬质量门
→ Reviewer 整篇审稿
→ 最多两轮“主线修订—互动素材全量重建—合成—复审”
→ 编译为 RuntimeLesson 与 Teaching Beats
→ 以 Cue 级语音承载语义讲解片段，并为每个选择项生成专属诊断反馈语音
→ 进入全屏课堂
```

Runtime Beat 仍是认知与互动边界，Cue 是语音与视觉动作的同步边界。每个 Cue
使用配置的 TTS provider 生成一段语音，火山引擎是当前默认路径，也可显式选择
OpenAI 兼容语音；运行时在语义讲解片段的 lead、start、end 阶段执行题目强调
或板书动作。当前 Cue 完成后才进入下一 Cue，全部 Cue 完成后才结束 Beat。
互动出现时，主讲语音停止，学生通过互动后才能继续。

公式渲染与强调是相互独立的白名单合同：公式只进入受约束的 KaTeX 渲染路径；
强调只能引用服务端编译的题目公式 ID 或已经写入的板书 ID，并使用预定义样式，
不接受模型生成的 HTML、CSS、选择器或字符串偏移。

## 保存课程与课程 ID

完整课程保存在 SQLite 数据库 `var/lessons.sqlite3`，已生成的语音保存在
`var/audio/{lesson_id}/`。生成任务仍只保存在内存：如果服务在生成期间重启，
该任务不能恢复。课程 ID 只会在持久化保存成功后显示；看到完成状态和 ID
才表示课程已写入数据库。

生成完成页会显示课程 ID，可以先复制课程 ID，再点击进入课堂。首页的
“已有课程 ID”也可以直接打开已保存的课程。服务重启后，同一课程 ID 仍可用于
恢复对应的课堂。即使输入完全相同，每次都会分配新的课程 ID，不按题目内容复用课程。
账号、课程列表、删除和搜索不在当前 Demo 范围内。

### 完整性与备份

一节已保存课程由数据库记录和对应音频共同组成。删除数据库中的课程记录，
或删除对应的音频目录，都会使这节课程不完整。备份前先停止服务，然后同时备份
`var/lessons.sqlite3` 和整个 `var/audio/` 目录。不要单独复制 SQLite sidecar 文件
（例如 `-wal`、`-shm` 或 `-journal`）作为课程备份。

## 演示操作

1. 在生成页填写方程和简短参考答案；如有题库解析，可把多段文字与公式完整粘贴到“参考解析”。再按需选择指定方法和讲解长度。
2. 点击生成，观察“理解题目—核对题目材料—设计讲解—整篇审稿—修订并编译—生成语音”的任务阶段。符号能力探测、参考解析审阅与 grounding 的内部模式都使用同一个对外阶段，学生页不展示模型分歧、检查请求或供应商响应。
3. 生成完后不会自动进入课堂。先保存生成完成页显示的课程 ID，再点击“进入课堂”；也可以回到首页，在“已有课程 ID”中输入它。课堂使用横屏 16:10 舞台，输入表单和调试信息不会出现在学生视野。
4. 首次点击“开始讲解”以解锁浏览器音频。使用上一段、重播、暂停或继续控制教学节拍，不能跳到尚未讲解的完整答案。
5. 新生成课程中的选择题由服务端判定。每个选择项会显示并朗读其专属诊断反馈；错误后继续按钮保持禁用，直到完成当前互动。
6. `point_select`、表达式、`free_text` 和旧式 `transfer` 输入仅作已有课程的兼容读取；新生成的自动评分数学作答固定使用 `choice`。
7. 完成总结后处理选择式近迁移题，检查同一方法能否迁移到表面不同的新题。

## 本地公式资源

课堂不从 CDN 加载公式资源。`app/static/vendor/katex/` 中提交了 KaTeX 的
JavaScript、CSS 与字体，`package-lock.json` 固定依赖版本。刷新或升级本地
vendor 资源时，先在项目根目录执行：

```bash
npm install
mkdir -p app/static/vendor/katex
cp node_modules/katex/dist/katex.mjs app/static/vendor/katex/katex.mjs
cp node_modules/katex/dist/katex.min.css app/static/vendor/katex/katex.min.css
rm -rf app/static/vendor/katex/fonts
cp -R node_modules/katex/dist/fonts app/static/vendor/katex/fonts
cp node_modules/katex/LICENSE app/static/vendor/katex/LICENSE
npm test
```

这些命令只替换受版本控制的 KaTeX vendor 文件；`package-lock.json` 固定来源版本。
确认测试通过后，再把本地 KaTeX 文件与 lockfile 一并审阅。

## 自动化验证

完整验证在 `general` 环境中运行：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests
python -m compileall -q app scripts tests
npm test
node --check app/static/generate.js
node --check app/static/generation-flow.mjs
node --check app/static/lesson.js
node --check app/static/cue-player.mjs
node --check app/static/runtime-core.mjs
node --check app/static/math-text.mjs
```

### 教学质量黄金题集与版本比较

`tests/fixtures/pedagogy_golden_cases.json` 保存了 18 道经过人工编写与审阅的
无图初中数学题。每道题都记录了思维轨迹锚点、必讲内容、典型误解、关键板书状态，
以及可接受和不可接受的讲解片段。这些内容是教师给评估器的质量预期，不代表模型
已经达到预期，也不代表学生已经学会。

离线检查只验证题集元数据、评估器输出合同和其他确定性程序行为，不调用模型：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q tests/test_pedagogy_evaluation.py
```

真实评估必须显式开启集成模式，并且使用与当前代码一致的 rubric 版本。标准比较为
每题生成三次。输出目录必须是新目录或空目录；脚本不会覆盖已有评估结果：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
set -a
source .env
set +a
RUN_INTEGRATION=1 python scripts/run_pedagogy_evaluation.py \
  --rubric-version 0.1 \
  --runs-per-case 3 \
  --output-dir /tmp/ai-math-pedagogy-v01
```

如果只修改 prompt 而 rubric 仍为 `0.1`，为两个候选分别增加不同的稳定标签，例如
`--candidate-version prompt-a` 与 `--candidate-version prompt-b`。比较器以候选标签区分
版本，并同时核对两次运行的题集指纹；这样 prompt-only 比较不会被误认为同一个候选。

真实模式只调用课程生成服务，不生成语音，也不启动 Web 服务。`private/records/`
保存完整的内部生成记录，`public/runtime/` 保存移除参考答案、参考解析、正确选项、
诊断反馈和内部验证报告后的课堂内容。`manifest.json` 只汇总确定性合同指标：生成
成功、硬门槛审稿状态、必讲内容覆盖、讲稿与视觉动作绑定、Schema/Runtime 通过、
时长和调用次数。日志不写模型正文、内部审稿反馈或服务密钥；失败项只记录受限的
失败类别和阶段。每个成功运行在清单中保存公开课堂 JSON 的 SHA-256；比较器会先
核对哈希，再按固定公开 Schema 重建内容。参考答案、参考解析、正确选项、内部反馈、
验证报告和候选版本字段无法进入盲评文件。

评估清单必须完整包含“所有题目 × 所有运行轮次”，成功和失败都占一个明确位置。
比较报告分别统计双方成功、双方失败、单边失败和被排除的盲评对，不会把失败运行
悄悄删掉后把剩余结果当成完整比较。输出根目录及脚本控制的子路径不能是符号链接，
脚本会在创建和写入前后检查目录仍指向同一位置。

比较两个 rubric 或 prompt 版本时，应在对应版本的代码中分别生成独立目录，且两边
使用同一题集和相同的 `--runs-per-case`。随后离线生成盲评对：

```bash
python scripts/run_pedagogy_evaluation.py \
  --compare-run /tmp/ai-math-pedagogy-v01 /tmp/ai-math-pedagogy-v02 \
  --output-dir /tmp/ai-math-pedagogy-blind
```

给教师的文件是 `public/blind_pairs.json`，其中只有随机化后的 candidate A/B，
没有版本标签。版本对应关系单独保存在 `private/candidate_mapping.json`；收集盲评时
不要把该私有文件交给评审教师。脚本不会自动填写教师偏好，也不会从自动指标推断
偏好。

证据边界分为三层：自动审核和确定性指标只能说明生成物符合当前合同；教师成对盲评
可以说明教师更认可哪一版及其理由；真实学生是否理解、迁移和保持，需要独立的学生
任务、过程证据和学习效果研究。前两层都不能改写成已经证明学生学会。

配置真实端点后，默认运行不包含参考解析审阅的 core smoke：

```bash
python scripts/smoke_live.py
```

如需单独验证可选的多段参考解析审阅链路，显式运行：

```bash
python scripts/smoke_live.py --with-reference-audit
```

参数根 live smoke 的 `--grounded-parameter-root` 参数已实现。下列命令会使用
真实文本模型与语音端点运行它；Task 8 的 live 状态仍取决于本轮真实执行结果，
本地自动化测试通过不表示该 live smoke 已经通过：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
set -a
source .env
set +a
python scripts/smoke_live.py --grounded-parameter-root
```

该 smoke 使用上文的 `2n` 参数根题，断言生成过程不被“符号工具不支持”阻断，
服务端模式为 `model_cross_checked` 或 `reference_grounded`，一致性状态有效且
冻结教学路线带有指纹；讲解保持纯选择式互动、完成整篇审稿、每个 Beat 都有
同步 Cue、每个 Cue 都有语音，语义动作包含题目首个公式强调与带 `4n` 的板书，
并且板书中存在参考结论。它只输出课程 ID、Beat/Cue 数量、互动类型、审稿状态、音频就绪和
`conclusion_present` 布尔值，不输出 prompt、题目、
答案、选项文本、密钥、端点或供应商响应正文。

两种 invocation 分开提供证据：可选审阅失败不会抹去一次已成功的 core 结果。
脚本会在创建客户端前检查 Chat 与 TTS 配置，并将本次语音资产写入自动清理的
临时目录；无论成功、Provider 失败或结构断言失败，均不会向 `var/audio/` 留下
资产。core 还会断言配方法题没有调用 Math Route Agent，且核心模型调用从
Lesson Director、Materials Agent 到 Reviewer 的顺序正确。生成并配音后，core
会断言第二个 Beat 是“先认识方法”的配方法介绍、
选择互动都有 3–4 个选项及已生成的选项诊断反馈语音、近迁移的选项 ID/顺序及
公式标签与内部 canonical answer 的确定性格式一致；有同步 Cue 的 Beat 要求
每个 Cue 都有音频，未带 Cue 的兼容课程仍要求 Beat 音频。
`--with-reference-audit` 额外要求审阅状态为 `approved`。成功时只打印课程 ID、
模式、结构计数、状态与上述布尔摘要；不会打印密钥、题目、选项、答案、prompt、
端点或供应商响应正文。未配置真实密钥时不要运行网络 smoke。

## 证据边界

- **Schema 校验**证明数据满足字段、枚举、长度和跨字段不变量，不能证明数学结论正确或讲解有效。
- **能力分类**只识别当前本地符号工具能否严格处理输入；`unsupported` 不能推出题目错误，也不能推出参考解析正确。
- **严格符号校验**证明回归题及受支持代数范围内的答案、关键变形和表达式等价关系通过 SymPy 检查，不能覆盖开放题型、自然语言讲解中的全部数学含义或模型供应商行为。
- **结构化模型审阅与局部检查**约束提取出的方法、假设、推理步骤、结论和检查请求。检查请求本身来自模型，因此通过时可以提升服务端置信度，失败或不支持时只记录 warning，不能据此自动判定参考材料错误。
- **服务端验证报告**记录路线模式、一致性、检查结果和审稿状态。公开课程 payload 会移除该报告、参考答案、参考解析、正确选项和未选中选项的反馈。
- **视觉验证**需要页面测试和真实浏览器中的横屏、板书、遮罩、音频、互动联动检查。DOM 或状态机测试通过，不能证明不同设备上的视觉重点始终清楚。
- **教学质量**目前只有结构化规则、模型整篇审稿和 Demo 人工检查。Reviewer 尚未经过教师一致性校准，也没有学生实验，因此不能据此声称提升理解、近迁移、保持或长期学习效果。
- **同步证据**中的固定参数根 smoke 与浏览器检查只覆盖一个既定题目及其 Cue 合同，不能证明所有题型都能稳定生成，也不能证明学习效果。
- **音频降级**在媒体缺失、拒播或超时时提供按讲稿长度计算的可读降级，使课堂能够继续；出现该降级不代表同步语音成功，验收仍必须要求每个 Cue 的真实语音就绪。

core smoke 只能说明某次配置下完成了不含参考解析审阅的“生成—审稿—语音”
链路及其有限的结构断言；带 flag 的 smoke 才额外说明可选审阅链路在该次运行中
通过。它们都不能替代稳定性压测、隐私审查、教师评价、浏览器人工验收或学生
学习效果研究。浏览器验证仍需要在真实横屏课堂中检查公式排版、板书层、音频、
点选与选择反馈的联动；教学证据仍需要教师审阅和学生研究，不能由本地测试或
一次 live smoke 推断。
