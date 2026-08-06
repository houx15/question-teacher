# 无图初中数学 AI 讲题 Demo

这是一个面向现场演示的窄范围教学运行时：输入一道无图的一元一次或一元二次方程、简短参考答案和可选的多段参考解析，系统先独立校验数学结果，再审阅参考解析、生成、整篇审稿和修订讲解，随后按教学节拍生成语音，最后进入横屏全屏课堂。课程与生成任务只保存在当前服务进程的内存中，重启服务后会丢失。

## 支持范围

当前支持：

- 一元一次方程；
- 一元二次方程；
- 因式分解法、公式法和配方法；
- 指定方法或由 Lesson Director 选择方法；
- 精简、标准两种讲解长度；
- 点选、选择、表达式、简短文字和近迁移互动。

当前不支持几何图形、函数图像、证明题、不等式、多问组合题、高于二次的方程、账号与学习历史、数字人和 MP4 合成。数学引擎只接受受限的实数一元代数表达式和有限解集。

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
→ Math Engine 独立验证题目和参考答案
→ Reference Material Auditor 审阅解析中的结论、步骤与教学素材
→ Math Engine 核对解析中可验证的结论与关键变形
→ Lesson Director 生成完整讲解
→ schema 与数学步骤校验
→ Reviewer 整篇审稿
→ 最多两轮整体修订
→ 编译为 RuntimeLesson 与 Teaching Beats
→ 按 Beat 生成讲解、提示和正确反馈语音
→ 进入全屏课堂
```

语音与板书以 Beat 为同步单位。每个 Beat 只承担一个主要认知动作；开始播放该段语音时执行对应板书动作，语音结束后才允许自动进入下一段。互动出现时，主讲语音停止，学生通过互动后才能继续。

## 演示操作

1. 在生成页填写方程和简短参考答案；如有题库解析，可把多段文字与公式完整粘贴到“参考解析”。再按需选择指定方法和讲解长度。
2. 点击生成，观察“理解题目—验证数学路线—审阅解析—设计讲解—整篇审稿—修订并编译—生成语音”的任务阶段。没有填写参考解析时会跳过审阅解析。
3. 生成完成后进入 `/lesson/{lesson_id}`。课堂使用横屏 16:10 舞台，输入表单和调试信息不会出现在学生视野。
4. 首次点击“开始讲解”以解锁浏览器音频。使用上一段、重播、暂停或继续控制教学节拍，不能跳到尚未讲解的完整答案。
5. 选择或点选按去除首尾空格后的精确值判断；表达式与近迁移答案由 Math Engine 判断等价性。错误答案显示下一条提示并保持继续按钮禁用。
6. 简短文字回答在 v0.1 中固定标记为“待确认”，不会发送给模型，也不会自动判错。
7. 完成总结后处理近迁移题，检查同一方法能否迁移到表面不同的新题。

## 自动化验证

所有 Python 命令都在 `general` 环境中运行：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate general
pytest -q
```

聚焦运行数学回归和 OpenAI 兼容客户端测试：

```bash
pytest -q tests/test_math_engine.py tests/test_llm_client.py
```

课堂状态机使用 Node.js 内置测试运行器：

```bash
node --test tests/runtime-core.test.mjs
```

配置真实端点后，可以运行一次可选 smoke：

```bash
python scripts/smoke_live.py
```

脚本会在创建客户端前检查 Chat 与 TTS 配置。成功时只打印课程 ID、Beat 数量、音频是否齐全、数学与审稿状态、修订次数；不会打印密钥、题目、prompt 或供应商响应正文。未配置真实密钥时不要运行网络 smoke。

## 证据边界

- **Schema 校验**证明数据满足字段、枚举、长度和跨字段不变量，不能证明数学结论正确或讲解有效。
- **符号校验**证明六道回归题及受支持代数范围内的答案、关键变形和表达式等价关系通过 SymPy 检查，不能覆盖开放题型、自然语言讲解中的全部数学含义或模型供应商行为。
- **参考解析审阅**由模型提取结论、关键步骤和可用教学素材，并对可符号化步骤再次校验。它可以阻止已经识别的答案冲突和错误变形，但不能证明发现了多段自然语言中的所有错误。
- **视觉验证**需要页面测试和真实浏览器中的横屏、板书、遮罩、音频、互动联动检查。DOM 或状态机测试通过，不能证明不同设备上的视觉重点始终清楚。
- **教学质量**目前只有结构化规则、模型整篇审稿和 Demo 人工检查。Reviewer 尚未经过教师一致性校准，也没有学生实验，因此不能据此声称提升理解、近迁移、保持或长期学习效果。

真实端点 smoke 只能说明某次配置下完成了“生成—审稿—语音”链路。它不能替代稳定性压测、隐私审查、教师评价、浏览器人工验收或学生学习效果研究。
