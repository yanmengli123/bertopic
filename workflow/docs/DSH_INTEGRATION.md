# DSH 接入契约 v1

DSH 专用 SDK 尚未提供。本文件定义已经实现、可以测试的通用进程协议；`plugin_contract.json` 不冒充官方插件 manifest。官方规范到位后，需适配注册、生命周期、权限、密钥库、文件选择器、报告 WebView 和打包格式，再进行宿主验收。

## 宿主界面流程

1. 文件选择支持多库、多批次；每个文件可指定来源。
2. 调用 inspect，显示真实读取格式、工作表、表头行、字段映射、原始记录数和缺摘要提示。冲突时让用户选择映射，不隐藏失败。
3. 读取配置 schema 生成参数表单；区分数据清理、嵌入模型、聚类、时间范围和绘图。按用户需求保留或去重，不能默认把 WoS 去重。
4. validate 只读检查，通过后展示远程嵌入目的地及将发送的标题/摘要。宿主按自身规则取得同意，从安全密钥库注入 worker 环境。
5. 分配新的 run_id，启动 worker，展示阶段及进度。run_prepared 只表示部分阶段完成；只有 run_completed 和通过校验的 result.json 表示全流程成功。
6. 取消时调用 cancel；UI 显示“正在取消”直到 worker 退出。强制退出保留已发布阶段，重新启动时显式恢复锁和 resume。
7. 显示报告前调用 verify。只打开 run_directory 内 manifest 列出的产物，WebView 禁止任意文件访问、导航和脚本桥提升权限。
8. package 生成 ZIP 和 SHA-256。结果含文献文本，分发范围由宿主/用户决定；不会自动上传结果或发送消息。

## 进程

宿主配置可信的 Python 绝对路径。安装 wheel 与依赖后执行：

```text
python -u -B -m bertopic_workflow run --config <absolute-project.json> --run-id <unique-id>
```

必须使用参数数组，不拼接 shell 字符串。设置 `PYTHONUTF8=1`、`PYTHONUNBUFFERED=1`。`desktop/dsh_adapter.mjs` 使用 Node 原生模块，无 npm 依赖，兼容支持 child_process 的桌面宿主；它不是浏览器 renderer 中可直接运行的库。子进程仅继承基本操作系统环境及显式注入的 API 凭据，不继承宿主的其他 token 或 PYTHONPATH。桥接凭据变量需以 `_API_KEY` 或 `_TOKEN` 结尾。

stdout 每行是一条 UTF-8 JSON 事件，格式如下：

```json
{"schema_version":1,"event":"stage_progress","run_id":"project-001","sequence":4,"at":"2026-10-15T00:00:00+00:00","data":{"phase":"embed","completed":16,"total":100,"cache_hits":8,"requests":1}}
```

schema 位于 Python 包的 `schemas/event.schema.json`。sequence 在每个进程内递增，恢复后的新进程从 1 开始；持久事件文件可以包含多个尝试，不能假设整个文件的 sequence 全局唯一。进度 total 描述当前操作（文献/候选/图片），不是剩余耗时估计。每条事件最多 2 MiB，桥接 stderr 只保留 128 KiB。错误事件不回显无效配置原值、API 响应或凭据。

主要事件：run_started、stage_started、stage_progress、stage_completed、run_prepared、validation_started、run_completed、run_failed、run_cancelled、error。其他只读命令分别返回 capabilities、inspected、config_validated、doctor、job_status、release_verified 等事件。

退出码：0 成功（包括 until 指定的部分流程）；2 配置/输入问题或缺依赖；3 运行错误；4 任务锁冲突；5 取消。宿主还应处理进程无法启动或被强制结束的情况，不能只检查有没有输出文件。

## Python SDK

```python
from bertopic_workflow.sdk import run_workflow, verify_release

result = run_workflow("project.json", run_id="project-001", on_event=handle_event)
if result["status"] == "completed":
    verify_release(result["run_directory"])
```

回调在执行线程同步触发；GUI 应把事件转发到自己的消息队列，不能在回调里阻塞分析。一个任务由一个 worker 写入。可并发运行不同任务，但宿主应限制 CPU、内存和远程 API 并发，避免多个本地大模型争抢显存。

## 状态和最终文件

每个任务的 config.json 为解析后的完整配置，provenance.json 为原始输入和代码指纹，job.json/events.jsonl 为可变运行信息；ingest/embed/model/report 是整体发布的阶段目录，各自有 `_stage_manifest.json`。release_manifest.json 索引最终文件，result.json 是完成标记。可变日志、取消请求、锁、未发布的暂存目录不进入最终 ZIP。

配置、输入、引擎版本或运行依赖变化会拒绝恢复，应新建任务。代码按阶段计算指纹，只复用相关代码及上游产物均未变化的阶段；已存在但指纹不匹配的阶段会明确报错，不静默覆盖。完成任务的结果保持不可变。异常遗留的暂存目录可用于诊断；宿主可在任务不运行时按自己的保留策略清理，不能把它们列为最终结果。

## 凭据与网络

凭据由宿主密钥库管理，按照 api_key_env 指定名称注入。不能置于参数、配置 JSON、日志或 UI 错误详情。远程接口要求 HTTPS，不跟随重定向转发 Authorization；只有显式启用的本机调试服务可以 HTTP。离线模式不发嵌入请求，也不下载模型/tokenizer。

生产环境应将 embedding endpoint 加入宿主允许列表，并固定模型修订/依赖镜像；本地模型禁用 trust_remote_code。结果哈希提供完整性核验，不等于可信发布者的数字签名；分发签名和插件权限验证由宿主接入完成。
