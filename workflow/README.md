# 多数据库 BERTopic 工作流

把文献导出文件转换为可核验的主题分析，供桌面程序通过 CLI、Python SDK 或 Node 子进程接口调用。原项目的四库分析保持原样；新引擎没有固定文件名、记录数、主题数、研究领域或本机 Python/Node 路径。DSH 官方插件注册等待宿主 SDK 规范，本目录先交付独立引擎与通用调用契约。

## 工作流

文件检查与映射 → 联合语料及异常隔离 → 单一模型空间的嵌入 → UMAP/HDBSCAN/BERTopic → 候选与种子诊断 → 15 类图及离线报告 → 独立数值验收 → 完成标记 → 可验证 ZIP。

每次运行使用新的 `run_id`。各阶段先在任务内部临时目录生成和校验，再整体重命名发布；最终 `result.json` 只在全部验收通过后出现。桌面端应以它作为完成标记。中途失败不会把半成品当成成功结果；`--resume` 先检查配置、原始文件、引擎版本与运行依赖，再按阶段核对相关代码及产物哈希。绘图代码修改不使已完成的嵌入失效；已发布的完成任务保持不可变，重新绘图请创建新任务。

## 安装

支持 Python 3.11/3.12；当前实测环境为 Windows、Python 3.11。其他操作系统需要按相同验收流程验证。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\bertopic-workflow.exe doctor
```

发布包提供 wheel，可用 `python -m pip install 路径\dsh_bertopic_workflow-0.1.0-py3-none-any.whl` 安装。wheel 不包含 Python、第三方依赖、模型权重或私有文献；离线部署需提前准备依赖 wheelhouse、字体和本地模型/向量缓存。依赖快照与构建 SHA-256 随发布包提供。安装本地嵌入能力可选择 `pip install '.[local]'`。所有嵌入模式均可在 CPU 运行，GPU 为本地模型的可选加速设备。

远程 API 模式不执行本地嵌入推理。BERTopic 的上游依赖仍可能经 Sentence Transformers 安装 PyTorch；这不表示需要 NVIDIA GPU 或 CUDA。`doctor` 会递归检查依赖版本，完整运行在任何 API 请求之前执行同样检查；本地模式还检查 `local` 依赖组。中文图形需要中文字体，可在 `report.font_path` 指定字体文件。

## 输入格式与字段

| 数据库 | 已实现读取路径 | 说明 |
|---|---|---|
| 维普 | CSV、TSV、XLSX、真实 XLS、HTML 表格、RIS | 中文常见表头及显式映射；实际模板差异需用 inspect 核对 |
| 万方 | 同上 | 自动寻找前 30 行中的真实表头，支持上方说明行 |
| 知网 | 同上 | 支持以 `.xls` 命名的 HTML，以及 Title-题名等双语列名 |
| WoS | CSV、TSV、XLS/XLSX、RIS | 分开保留 WoS Categories、Research Areas、Publication Type 和 Document Type |
| PubMed | CSV、TSV、Excel、MEDLINE/TXT、NBIB、PubMed XML | CSV 不一定含摘要；缺摘要会明确报告，不会伪造摘要 |
| 其他表格 | 同类表格格式 | 配置 mapping；无需改代码 |

支持一种数据库的单个文件、多个分批文件，以及不同数据库的联合文件。不能保证所有数据库历年所有自定义模板均可自动识别。未知表头、多候选标题/摘要列、多工作表都会报出具体问题，需显式指定 `mapping`、`sheet` 或 `header_row`；不猜测或静默读取错误列。

```powershell
bertopic-workflow inspect --input "D:\data\CNKI.xls" --source CNKI
bertopic-workflow inspect --input "D:\data\万方.xlsx" --source WanFang --sheet "Sheet1"
bertopic-workflow schema --output workflow.schema.json
bertopic-workflow validate --config project.json
```

配置示例见 `examples/project.template.json`；现有工作区迁移示例见 `examples/current_workspace.json`。所有相对路径相对于配置文件所在目录。标准 mapping 字段由 `config.py` 的 `FIELDS` 定义，包括 title、abstract、year、date、doi、journal、authors、keywords、institutions、countries、category、research_areas、document_type、publication_type、pmid、wos_id、addresses。例如 `"mapping":{"title":"自定义文献标题","abstract":"中文内容摘要"}`。不需要同时导入五个数据库。

原文件只读；导入保留文件 SHA-256、表格/工作表、物理行号、逐条 ID 与文本指纹。空题名且空摘要记录按配置报错或隔离；隔离及去重均输出明细，输入总量必须等于保留＋隔离＋去重。默认不去重，包括 WoS；可选择按全文精确一致或有效 DOI 去重，并保留被删除记录到保留记录的映射。CSV 导出对可能被表格程序执行的文本公式加前置单引号，JSON 保持原值。

## 嵌入与运行

默认硅基流动 `BAAI/bge-m3`，也支持 `BAAI/bge-large-zh-v1.5`、OpenAI 兼容接口、本地 Sentence Transformers 和预计算向量。一项任务只使用一个模型空间；不拼接不同模型向量。远程模型凭据只从环境变量读取，不能写入 JSON、源码或命令行参数。

```powershell
# 由宿主密钥库注入 SILICONFLOW_API_KEY，再启动进程。
bertopic-workflow run --config project.json --run-id project-001
bertopic-workflow status --run-dir .\runs\project-001
bertopic-workflow cancel --run-dir .\runs\project-001
bertopic-workflow run --config project.json --run-id project-001 --resume
bertopic-workflow verify --run-dir .\runs\project-001
bertopic-workflow package --run-dir .\runs\project-001 --destination .\deliveries\project-001.zip
```

先做导入可使用 `run --until ingest`。预计算模式需要 `.npy` 和 JSON，其中 `record_ids`、`text_sha256` 必须与这次导入的 `ingest/corpus.json` 逐条相同，`model` 必须等于配置声明；仅有等长矩阵不能证明对齐。

长文本通过对应 tokenizer 分块，不静默截断；分块向量按 token 数加权平均后 L2 归一化。缓存隔离 endpoint、模型/修订、维度、tokenizer 指纹、分块和池化规则。已成功的缓存会在下一次恢复时复用；仅对网络与限流错误执行有上限的重试，不把鉴权错误反复重试。`--offline` 禁止未命中的远程计算。浮动模型版本不能保证再次请求得到相同向量；复现以已保存的向量及哈希为准，本地生产部署应固定模型修订。

取消为协作式：嵌入批次、候选模型、种子及绘图之间检查取消；正在执行的单次 UMAP/HDBSCAN 调用不会即时中断。桌面桥提供超时终止兜底；强制中止后的未发布阶段保留诊断，需确认原进程退出后显式 `--recover-lock --resume`。不能并发写同一任务。

## 分析与图片口径

候选模型只复用相同降维坐标比较聚类参数；输出选择规则、实际应用参数、逐条主题、代表文献、c-TF-IDF 关键词、年度/数据库计数与比例、中心相似度、层级、原始向量轮廓系数、文档共现 NPMI、关键词多样性与多个种子 ARI/NMI。领域词和额外停用词由配置提供，没有水稻专用词典。

实现主题总览、关键词条形、词云、层级、文档分布、关键词热力、词秩、时间趋势、分类对比、相似度、关系网络、规模、离群比例、质量诊断、时期到主题构成桑基图。动态适配主题数量；静态部分选择主要主题，交互版及数值文件保留完整范围。缺年份或元数据、只有一个主题、全部离群时，明确记录 `not_applicable` 及原因。不会生成貌似有数据的空图。

气泡/网络节点面积按记录数成比例；帕累托曲线明确使用已归类记录作分母。MDS 提供 stress；主题很多时改用明确标注的 PCA。文档图使用聚类 UMAP 的前两坐标；超过绘图上限会记录实际抽样 ID。大规模绘图与完整记录导出是两个范围，不能混用。

国家/机构仅使用已导出字段或保守地址解析，每条记录注明来源；国家无法识别时留空，机构不做实体消歧。多组记录在每组各计一次，不把组数之和当作唯一论文数。时间截止年、不完整年、时间区间宽度均可配置。桑基图是各时期的主题构成，不能解释为同一论文跨主题迁移或主题分裂/合并。

年度趋势对无记录年份保留断点，不补零或跨空年连线；年度构成使用逐年堆叠柱，避免稀疏年份被连续面积误导。占比的坐标范围固定为 0–100%。小样本年份的比例不代表稳定趋势，需结合 evidence 文件中的年度分母判断。

独立验收重新计算记录守恒、ID/向量对齐、中心/余弦矩阵、年度/来源计数、轮廓系数、NPMI、关键图的数值证据与离线链接。发布清单列出所有产物的 SHA-256，ZIP 再核对 CRC 和内容哈希。安全模型导出使用 safetensors，不加载任意 pickle；它不包含原 UMAP/HDBSCAN，新文献的中心相似度分配不等同于原聚类预测。[BERTopic 官方序列化说明](https://maartengr.github.io/BERTopic/getting_started/serialization/serialization.html)。

## DSH 接入与验收

见 `docs/DSH_INTEGRATION.md`、`desktop/dsh_adapter.mjs`、`desktop/plugin_contract.json`。配置和事件 JSON Schema 随 Python 包提供。桥接代码使用参数数组与 `shell:false`，凭据通过子进程环境注入；不会把宿主账号或密钥写入结果包。宿主负责文件授权、网络数据提示、密钥库及结果 WebView 隔离。

```powershell
python -m pip install '.[test]'
python -m pytest
python -m ruff check src tests
python tools/build_release.py
```

工程机制和自动测试不代表已取得企业认证。发布进入 DSH 生产环境前仍需按实际 SDK 完成宿主集成验收、目标系统安装/依赖扫描、资源压力测试及业务主题人工评审。供应商新模板以实际文件为验收依据，无法仅凭数据库名称保证字段含义。详细已测范围及未测项见 `docs/ACCEPTANCE.md`。

`ci/github-actions.yml` 是可选 CI 模板，尚未在远端执行；确认仓库策略后可移至 `.github/workflows/` 启用。`tools/generate_schemas.py` 从配置模型和项目依赖生成随包协议文件；修改配置或依赖时应重新生成。代码发布包不包含文献或历史分析结果。

方法参考：[BERTopic 自定义嵌入](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html)、[PubMed 导出说明](https://pubmed.ncbi.nlm.nih.gov/help/)、[硅基流动嵌入接口](https://siliconflow.readme.io/reference/createembedding)。
