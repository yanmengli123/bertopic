# 四库联合 BERTopic 文献分析

主模型将 CNKI、万方、PubMed、WoS 全部记录合在同一语料中共同拟合。打开 `outputs/analysis_report.html` 查看主题、代表文献、交互图、年份变化和质量诊断。图表使用随目录保存的 Plotly 文件，浏览结果无需 API 或互联网。

## 已完成的结果

联合 bge-m3 模型得到 42 个非离群主题，13,987 条记录归类，8,562 条记录为离群（37.97%）。主要主题包括大米与淀粉加工理化性质（3,404 条）、QTL 定位（818 条）、粒型粒重调控（539 条）和高温胁迫（514 条）。2016–2020 与 2021–2025 相比，氮肥管理、粒型调控、光谱与机器学习检测的样本占比上升较明显。

原始向量轮廓系数 0.0429，主题存在重叠。种子 7 得到 37 个主题，离群率 26.83%；全部记录 ARI 为 0.5164，两次均归类的 13,402 条记录 ARI 为 0.8205。结果对离群边界和主题粒度有敏感性。

689/715 条中文文本（96.4%）集中在联合模型 T2，存在语言分组倾向。T2 是多个中文方向的混合，不能理解为单一实质主题。在相同中文聚类参数下，中文专用模型得到 20 个主题，中文 bge-m3 对照得到 18 个主题，均有 118 条离群记录。两模型 ARI=0.2917、NMI=0.5696，说明嵌入模型选择仍影响具体分组。

逐条结果、全部主题、代表文献、完整参数表和限制均在报告及 CSV 中。所有 32 个 CSV 经独立回读核对，四个输入文件哈希保持一致；完整模型和 safetensors 模型均已加载验证。结果属于这批未去重语料的探索性分析。

## 数据口径

| 来源 | 记录数 | 有摘要 | 无摘要 |
| --- | ---: | ---: | ---: |
| CNKI | 215 | 212 | 3 |
| 万方 | 500 | 495 | 5 |
| PubMed | 1,742 | 1,718 | 24 |
| WoS | 20,092 | 19,914 | 178 |
| 合计 | 22,549 | 22,339 | 210 |

原始文件位于上一级目录，未修改。WoS 按要求不去重，跨库重复记录也全部保留。WoS 的 20,092 条记录对应 19,092 个唯一 WoS ID。相同文本只在嵌入 API 缓存阶段复用计算，仍作为多条记录参与建模和统计。输出的 `doi_record_count`、`title_key_record_count`、`text_sha256_record_count`、`wos_id_record_count` 用于识别重复，未据此删除文献。

没有主题范围筛选。WoS 占样本约 89%，且内容包含其他谷物研究。这是所提供四库检索合集的分析，频数不能直接解释为去重后的独立论文数或整个水稻胚乳领域的研究规模。`rice_endosperm_term_match` 只是文本匹配提示，不能代替相关性审查。

建模文本为标题加摘要，无摘要时只用标题并保留 `has_abstract` 标记。文本清理仅处理 HTML、明确版权尾注和多余空白。语义嵌入保留基因名称、数字和科学符号。语言按实际文本识别，中文文本 715 条，不能以 PubMed 的原文语言字段替代摘要实际语言。

CNKI 的 `.xls` 实际为 HTML 表格。万方导出含两层表头。年份从年份字段提取，CNKI 缺失年份使用发表时间补齐，同时保存原始值和来源。一个原始 2027 年记录保留建模并标记复核，年度趋势截止 2026 年时排除；2026 年尚不完整。两个完整窗口 2016–2020 和 2021–2025 用于近期占比比较。

## 模型与诊断

联合模型使用硅基流动 `BAAI/bge-m3`，整个语料使用同一种 1,024 维向量。中文验证在同一组 715 条中文文本上分别拟合 `BAAI/bge-large-zh-v1.5` 和 `BAAI/bge-m3`，没有拼接不同模型的向量。中文模型的主题编号与联合模型、另一中文模型均不直接对应。

用官方 tokenizer 对长文本完整切片：bge-m3 每段上限 8,000 token，中文模型每段 480 token。每段向量先 L2 归一化，按 token 数加权平均，再 L2 归一化。该方案保留超长摘要全文，但长文本表示不同于一次整体嵌入，切片数量写入结果。SQLite 缓存按模型和实际输入文本哈希保存，可中断后继续。

主模型 UMAP 使用余弦距离、30 邻居、5 维、min_dist=0 和随机种子 42。HDBSCAN 使用欧氏距离、min_samples=10、eom，并比较最小簇规模 30、60、100。中文模型使用 15 邻居、min_samples=3，并展示最小簇规模 10、15、25 的敏感性。双模型正式比较都固定最小簇规模为 15，以控制聚类参数差异。候选在各自相同的降维坐标上比较。

主题表示使用 jieba 中文分词与英文词项规则，保留科研实体，使用中英文停用词、1–2 gram 和 c-TF-IDF 高频词调整。每主题输出 20 个词项。主题标签是关键词和代表文献的概括，属于解释性命名。

候选选择分数为 `0.30×覆盖率 + 0.25×(轮廓系数+1)/2 + 0.25×(NPMI+1)/2 + 0.20×关键词多样性`。这是本次候选比较的启发式规则，没有通用合格阈值。轮廓系数在原始嵌入空间按余弦距离计算，最多抽样 1,500 条已归类文献。NPMI 使用文档级前十关键词共现，不采用滑动窗口；无共现词对记 -1。多样性是所有主题前十词中不同词项的比例。

主模型另用随机种子 7 复拟合 UMAP/HDBSCAN，报告 ARI、NMI 和离群率变化。ARI/NMI 不受主题编号置换影响。种子比较和中文模型比较都是敏感性诊断，不能证明所有主题都正确。HDBSCAN 成员强度不是主题正确性的概率。

主题 -1 是保留的离群记录，没有自动改写为最近主题。`nearest_semantic_topic` 与 `nearest_topic_cosine` 是原始向量的辅助建议。代表文献按与本主题中心的余弦相似度排列，示例中相同文本只展示一次。主题频数仍包含全部原始重复记录。

逐年与数据库分布都包含离群记录作为分母。近期比较使用每个五年窗口的全部记录作为分母，反映样本占比变化。BERTopic 五年区间关键词演化沿用联合主题定义，不逐年独立建模。数据库、语言与主题的 NMI 只表示描述性关联。

## 输出文件

- `outputs/merged_four_databases.csv`：统一字段的四库合集，全部 22,549 条。
- `outputs/joint_bge_m3/document_topics.csv`：完整逐条主题、成员强度、最近主题、元数据与重复标记。
- `topic_summary.csv`、`topic_keywords.csv`、`representative_documents.csv`：主题汇总、词权重与代表文献。
- `topics_by_year.csv`、`topics_by_source.csv`、`topics_by_language.csv`：逐年、数据库与语言分布。
- `recent_topic_changes.csv`、`topic_keywords_over_time.csv`：近期占比变化与五年区间关键词演化。
- `parameter_comparison.csv`、`stability.json`、`seed_stability_assignments.csv`：候选参数与稳定性结果。
- `topic_similarity.csv`、`topic_similarity.npy`、`hierarchy_linkage.npy`：主题中心相似度与层次关系。
- `visualizations/`：离线交互图及 Plotly 运行文件。`figures/`：PNG 和 SVG 科研图。
- `figure_gallery.html`：15 张主题图库索引（主题总览、关键词、词云、层级、语义分布、关键词热力图、词秩、时间趋势、分类对比、相似度、关系网络、规模分布、噪声占比、质量诊断、演化桑基图），每张含 PNG/SVG 与交互版。生成脚本为 `make_figures.py`，产物清单见 `figures_manifest.json`。分类对比图的国家与机构取自 WoS 地址字段，其他数据库无该信息。
- 每个模型的 `bertopic_model.pkl` 保存完整降维与聚类状态；`portable_model/` 保存 safetensors 主题表示与 c-TF-IDF，可移植模型不保留原始聚类预测状态。
- `outputs/chinese_bge_zh/` 与 `outputs/chinese_bge_m3_comparison/`：两种中文子集拟合的完整结果。
- `outputs/chinese_model_assignments.csv` 与 `chinese_model_comparison.json`：同一中文文献的双模型对照。
- 嵌入矩阵、实际 API 用量、文本切片和记录顺序保存在 `outputs/`，原始文件 SHA-256 记录于 `data_manifest.json`。
- CSV 为 UTF-8 BOM 编码，JSON 是分析中间数据。CSV 内容经 Artifact Tool 写入、回读校验后导出。

## 运行

本次通过远端 API 生成嵌入，本地 CPU 完成降维聚类。没有使用本地 GPU，也无需配置 CUDA。API 密钥只从 `SILICONFLOW_API_KEY` 环境变量读取，没有写入任何分析文件。

使用已经完成并缓存的向量，离线重新分析：

```powershell
powershell -ExecutionPolicy Bypass -File D:\bertopic\bertopic_analysis\run.ps1 -Offline
```

只重新生成报告和 CSV：

```powershell
powershell -ExecutionPolicy Bypass -File D:\bertopic\bertopic_analysis\run.ps1 -ReportOnly
```

正常全流程执行同一命令并去掉 `-Offline`。只有缺失向量需要接口密钥。现有完整矩阵自动复用；输入文本或顺序改变时，必须重新生成对应向量，禁止用旧向量静默匹配新记录。

运行环境在 `D:\bertopic\.venv_bertopic`。为绕开本机已有 Transformer/GPU 依赖冲突，启动器在子进程开始前设置 `USE_TORCH=0` 和 `USE_TF=0`。实际算法仍使用正式 BERTopic、UMAP 与 HDBSCAN。

新机器可创建干净 Python 3.11 环境，先安装 `requirements-core.txt`，再用 `python -m pip install --no-deps bertopic==0.17.4` 安装 BERTopic。没有本地嵌入模型时不需要安装 PyTorch 或 SentenceTransformers。CSV 导出还使用本机 Codex 的 bundled Node 和 `@oai/artifact-tool`；移植时应按该工具的运行环境配置提供依赖路径，不能只复制 Node 可执行文件。已导出的结果与离线报告可以直接阅读。

完整 pickle 模型包含中英文 tokenizer。读取可移植模型后继续做 c-TF-IDF 操作时，需要恢复 Python tokenizer：`from text_utils import tokenize`，然后 `model.vectorizer_model.set_params(tokenizer=tokenize, token_pattern=None, lowercase=False)`。新文献预测时显式提供相同模型生成的向量；不要让 BERTopic 自动下载一个不同的嵌入模型。

## 图库运行与验收

入口是 `outputs/joint_bge_m3/figure_gallery.html`，支持离线搜索、15 类主图、全部 42 个主题的独立关键词图和真实分词频次词云。静态图包含 300 dpi PNG、SVG 和 PDF；独立词频词云的 SVG 使用原生文字。总览词云仍使用 c-TF-IDF 权重，不能称为原始词频。散点与总览词云的 SVG/PDF 可能含栅格层，清单中的 `contains_raster` 可用于判断。

```powershell
# 只读取已有模型结果：不调 API，不重新训练，不改主题分配。
D:\bertopic\.venv_bertopic\Scripts\python.exe D:\bertopic\bertopic_analysis\make_figures.py

# 首次完整生成后，按需重绘；保留其他图与元数据清单。
D:\bertopic\.venv_bertopic\Scripts\python.exe D:\bertopic\bertopic_analysis\make_figures.py --only 1,8,14

# 验证每个文件的大小、SHA-256、主题覆盖与数据/代码版本。
D:\bertopic\.venv_bertopic\Scripts\python.exe D:\bertopic\bertopic_analysis\make_figures.py --check-only

# 回归检查：地址解析、完整文档覆盖、并发锁、提交回滚与篡改检出。
Set-Location D:\bertopic\bertopic_analysis
D:\bertopic\.venv_bertopic\Scripts\python.exe -m unittest test_figure_ops
```

`figures_manifest.json` 是发布版本的最终标记：记录输入/代码 SHA-256、运行环境、随机种子、每个产物的哈希/大小/PNG 尺寸、42 个主题的全部独立图、离线 Plotly 共享脚本和文档交互图覆盖。`figures_validation.json` 保存发布前自动验收结果；`figure_build_status.json` 保存最近构建状态。

构建先校验文档 ID、嵌入顺序、主题计数、向量规范和原始来源，再在临时目录生成全部请求产物。渲染或验收失败时，已有发布图集保持可用；发布过程遇到 Python 可捕获的异常时自动回滚已替换文件。停电/强制终止可能发生于文件提交中间，应先运行 `--check-only` 检查，失败则完整重绘。并发构建由排他锁阻止；只有旧进程已退出时才允许 `--recover-lock` 清理遗留锁。数据或绘图代码变化时禁止混用旧版图，要求完整重绘。原始输入文件需与 `data_manifest.json` 中的哈希匹配。

国家/地区与机构图使用 WoS 全部地址，并按“同一文献在同一组只计一次”完整计数；国际合作文献可以进入多个国家/地区组，不能将各组计数相加后解释为去重总量。无法识别的国家尾缀保持缺失；国家与机构覆盖在 manifest 的 `extras.metadata_coverage` 中报告。机构仅规范空白，没有实体消歧；期刊原始名称也没有做缩写归并。

质量诊断修正了原有两个混淆：噪声图两组均使用“最近主题中心余弦相似度”，不再混合余弦和成员强度；主题关键词独享占比按词是否仅出现在该主题前十词计算，与全局词多样性分别说明。NPMI 均值、轮廓均值放在轴标签处，42 行主题刻度放大。工程验收通过不能替代主题人工评审；仅两次种子比较不足以给出稳定性的置信区间。主模型中中文主题混合明显，中文细分解释仍应结合报告中的补充模型。

桑基图显示“时期/数据库 → 主题”的构成，不能解释为同一文献跨主题迁移。WoS 的学科字段缺失，文献类型不能替代真实学科分类；CNKI/万方原始中图分类补充图展示分类前缀与联合/中文补充主题的对应（各自编号独立），仅覆盖原始分类有值的中文记录，未凭模型填充学科值。2026 年不完整，趋势图中的年份范围与排除数在 manifest 中说明。

## 方法来源

- [BERTopic 算法](https://maartengr.github.io/BERTopic/algorithm/algorithm.html)
- [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)
- [硅基流动嵌入接口](https://api-docs.siliconflow.cn/docs/api/embeddings-post)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
