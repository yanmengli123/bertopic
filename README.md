# 四库联合 BERTopic 文献分析

通用工作流已整理到 [workflow/README.md](workflow/README.md)：支持维普、万方、知网、WoS、PubMed 的可配置导入、独立任务、嵌入、BERTopic、15 类图、验收和打包，并提供 CLI、Python SDK 与桌面进程桥。DSH 官方注册仍需按宿主 SDK 规范接入。

下文记录原四库项目的历史运行方式及分析口径，原始数据和既有结果继续保留。

将 CNKI、万方、PubMed 和 Web of Science 文献合并为一个语料，使用硅基流动 `BAAI/bge-m3` 嵌入与 BERTopic 完成主题分析。中文子集使用 `BAAI/bge-large-zh-v1.5` 作补充比较；两个向量空间分别建模。

本仓库管理源代码、依赖版本和方法说明。原始文献、API 向量缓存、训练模型、CSV、图片、HTML 报告与压缩包保存在本地，由 `.gitignore` 排除。API 密钥从环境变量读取。

## 代码结构

| 路径 | 用途 |
| --- | --- |
| [bertopic_analysis/README.md](bertopic_analysis/README.md) | 数据口径、完整运行说明及图库验收机制 |
| [bertopic_analysis/prepare_and_embed.py](bertopic_analysis/prepare_and_embed.py) | 四库读取、文本清理、API 嵌入及缓存 |
| [bertopic_analysis/analyze.py](bertopic_analysis/analyze.py) | 联合建模、候选参数、主题指标与种子比较 |
| [bertopic_analysis/report.py](bertopic_analysis/report.py) | 分析报告与补充可视化 |
| [bertopic_analysis/make_figures.py](bertopic_analysis/make_figures.py) | 15 类主题分析图及全部主题独立图 |
| [bertopic_analysis/figure_ops.py](bertopic_analysis/figure_ops.py) | 输入/产物哈希、并发锁、增量保护和发布回滚 |
| [bertopic_analysis/test_figure_ops.py](bertopic_analysis/test_figure_ops.py) | 覆盖与构建安全的回归检查 |
| [.pubmed_merge_support/merge_pubmed.mjs](.pubmed_merge_support/merge_pubmed.mjs) | PubMed 导出文件合并 |

## 本地运行

当前运行环境为 Windows、Python 3.11；入口使用仓库根目录下的 `.venv_bertopic`。新环境安装 `bertopic_analysis/requirements-core.txt` 后，再安装 `bertopic==0.17.4`（详见分析目录说明）。CSV 导出和 PubMed 合并依赖 `@oai/artifact-tool` 与 Node.js，需要在新机器配置对应依赖；不能仅靠安装 Python 完成全部复现。

原始文件放在仓库根目录：`CNKI.xls`、`WanFangdata.xlsx`、`pubmed_merged_bertopic.csv`、`wos.csv`。PubMed 合并脚本读取 `PubMed/`。部分路径和中文字体按现有 Windows 工作区配置，迁移机器时需要检查。

```powershell
# 已有匹配向量：离线分析
powershell -ExecutionPolicy Bypass -File .\bertopic_analysis\run.ps1 -Offline

# 已有分析产物：重新生成报告、图和 CSV
powershell -ExecutionPolicy Bypass -File .\bertopic_analysis\run.ps1 -ReportOnly

# 只核验已发布图库
.\.venv_bertopic\Scripts\python.exe .\bertopic_analysis\make_figures.py --check-only
```

正常全流程去掉 `-Offline`，并在当前会话设置 `SILICONFLOW_API_KEY`。不要把实际密钥写入代码、Git 配置、提交记录或仓库文件。使用远端嵌入时，本地 CPU 完成分析，不要求本地 GPU。

回归检查在 `bertopic_analysis/` 下运行 `python -m unittest test_figure_ops`。其中部分检查依赖已有分析产物，克隆代码后未准备数据时不能直接执行全部检查。

## 当前结果与解释边界

本地一次分析保留全部 22,549 条未去重记录，其中 WoS 20,092 条。主模型得到 42 个非离群主题、8,562 条离群记录。保持不去重是本次分析要求，频数表示输入记录数，不能解释为去重后的独立论文数或领域总体。

结果属于探索性分析：原始向量空间轮廓系数约 0.043，主题重叠明显；689/715 条中文文本集中于联合模型 T2；更换随机种子后主题数量与离群边界发生变化。中文细分解释应结合补充模型和代表文献审阅。

已实现文件校验、增量保护、排他锁和可捕获异常的发布回滚；尚未完成严格企业生产验收。当前已知绘图局限包括：

- MDS 二维布局 normalized stress 约 0.270，只适合作探索性距离展示。
- 图 7 的 42 行主题刻度仍有重叠。
- 图 11 节点面积采用压缩缩放，不能当作与文献数量等比例的面积。
- 图 12 帕累托曲线的分母是已归类记录，轴标签仍需进一步明确。
- 图 15 显示时期/数据库到主题的构成，不表示同一文献迁移或已验证的主题分裂、合并。
- 国家/机构仅来自 WoS 地址，机构未实体消歧；中图分类补充图只覆盖原始分类有值的中文子集。

逐文件提交无法保证断电或强制终止时的目录级原子性。遇到异常退出，应先检查图库，再按说明完整重绘。跨环境复现、逐图数值断言和主题人工评审仍需继续完善。
