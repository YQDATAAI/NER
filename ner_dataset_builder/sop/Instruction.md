# Instruction（构建SOP）

本文件是“新行业初始化构建SOP”，用于规范从 GEO 迁移到任意新行业时的配置准备与生成流程。

本文件聚焦两件事：

1. 明确新行业初始化必须先确定的三项核心输入。  
2. 提供固定模板，指导 AI 生成可直接落地的 `system_instruction`、`task_description`、`few_shots`。

执行方式采用“清单式/填空式”步骤，目标是降低理解成本并保证流程一致性。

---

## 1. 新行业初始化只做三件事

当你从 GEO 扩展到其他行业（例如 ROCKET）时，先只准备以下三项：

- `system_instruction`
- `task_description`
- `few_shots` 样例

其余流程（数据放置、命令启动、结果迭代）按：

- `DEV_PLAN_GEO.md` -> 复制并改成 `DEV_PLAN_<行业>.md`
- `MANUAL_GEO.MD` -> 直接复用运行与迭代 SOP

---

## 2. 让 AI 生成 `system_instruction` 的固定模板

### 2.1 你发给 AI 的提示词（可直接复制）

```text
你要为NER项目生成 system_instruction。

需要做的项目内容为：{行业名称/项目内容}

产出示例如下：
- 示例1：你是一个严谨的地质与石油领域数据标注专家。你的任务是从文本中抽取特定的实体，并输出带有严格格式的JSON。
- 示例2：你是一个严谨的航天与火箭工程领域数据标注专家。你的任务是从技术文档中抽取指定实体并输出严格JSON。你必须只输出纯JSON，不得包含解释、思考过程或Markdown代码块。对于不确定或文本中不存在的实体，必须丢弃，不得臆造。

请基于“项目内容”生成最终 system_instruction（一段中文文本，不要解释）。

产出要求必须包含：
1) 专家角色定义
2) 仅输出纯JSON，不得输出解释/Markdown代码块
3) 不确定时宁缺毋滥，不要编造实体
```

### 2.2 产出示例（GEO -> ROCKET）

示例1（GEO）：

```text
你是一个严谨的地质与石油领域数据标注专家。你的任务是从文本中抽取特定的实体，并输出带有严格格式的JSON。
```

示例2（ROCKET）：

```text
你是一个严谨的航天与火箭工程领域数据标注专家。你的任务是从技术文档中抽取指定实体并输出严格JSON。你必须只输出纯JSON，不得包含解释、思考过程或Markdown代码块。对于不确定或文本中不存在的实体，必须丢弃，不得臆造。
```

---

## 3. few_shots 样例准备模板（必须给具体样例）

### 3.1 通用规则

- 每个实体类型至少准备 2 条正例。
- 至少准备 3 条“易错边界样例”（让模型学会不乱标）。
- 文本尽量来自真实文档句式，避免纯造句。
- 样例里的索引可先粗写，实际训练前会由解析器回溯纠偏。

### 3.2 `few_shots.yaml` 模板

```yaml
examples:
  - text: "<示例文本1>"
    label: '{"<entity_key>": {"<实体原文>": [[0, 0]]}}'
  - text: "<示例文本2>"
    label: '{"<entity_key>": {"<实体原文>": [[0, 0]]}}'
```

### 3.3 GEO 样例（当前项目可复用）

```yaml
examples:
  - text: "SN0015-08 井录井完井报告"
    label: '{"well_name": {"SN0015-08": [[0, 8]]}}'
  - text: "评审单位：长庆油田苏里格南作业分公司地学部"
    label: '{"block": {"苏里格南作业分公司": [[9, 17]]}}'
  - text: "古生界二叠系下统山西组"
    label: '{"strat_unit": {"山西组": [[8, 10]]}}'
  - text: "云岩：成份中白云石约占85.2~90.4%"
    label: '{"lithology": {"云岩": [[0, 1]]}}'
```

### 3.4 ROCKET 样例（泛化参考）

```yaml
examples:
  - text: "长征五号B运载火箭执行近地轨道任务"
    label: '{"rocket_model": {"长征五号B": [[0, 5]]}}'
  - text: "一级采用YF-100K发动机，二级采用YF-75D发动机"
    label: '{"stage": {"一级": [[0, 1]], "二级": [[15, 16]]}, "engine_model": {"YF-100K": [[4, 10]], "YF-75D": [[21, 26]]}}'
  - text: "该发动机海平面推力约1200kN"
    label: '{"thrust": {"1200kN": [[10, 15]]}}'
  - text: "推进剂组合为液氧煤油"
    label: '{"fuel_type": {"液氧煤油": [[6, 9]]}}'
```

---

## 4. 让 AI 生成 `task_description` 的固定模板

### 4.1 你发给 AI 的提示词（可直接复制）

```text
请基于下面提供的 few_shots.yaml 内容，自动归纳实体类型并生成 task_description。
要求输出中文文本，不要解释，不要多余内容。

few_shots.yaml 内容如下：
{few_shots_yaml完整内容}

请按以下要求生成 task_description：
1) 先列出“请抽取以下N类实体：”及实体清单（entity_key + 中文解释）
2) 严格要求必须包含：
   - 必须输出纯JSON格式，不要包含任何额外解释或Markdown代码块
   - 输出格式为：{"实体类别": {"实体原文": [[起始索引, 结束索引]]}}
   - 如果未找到任何实体，请输出 {}
3) 语气与格式参照下面示例风格

示例风格：
请抽取以下四类实体：
1. well_name (井名)
2. block (区块)
3. strat_unit (层位名称)
4. lithology (主岩性)

严格要求：
- 必须输出纯JSON格式，不要包含任何额外的解释或Markdown代码块。
- 输出格式为：{"实体类别": {"实体原文": [[起始索引, 结束索引]]}}。
- 如果未找到任何实体，请输出 {}。
```

### 4.2 产出示例（GEO）

```text
请抽取以下六类实体：
1. well_name (井名)
2. block (区块)
3. strat_unit (层位名称)
4. lithology (主岩性)
5. lithology_color (岩性颜色)
6. lithology_texture (岩性结构描述)

严格要求：
- 必须输出纯JSON格式，不要包含任何额外的解释或Markdown代码块。
- 输出格式为：{"实体类别": {"实体原文": [[起始索引, 结束索引]]}}。
- 如果未找到任何实体，请输出 {}。
```

---

## 5. 最终落地步骤（按顺序）

1. 明确“行业名称/项目内容”。  
2. 用第 2 节模板让 AI 生成 `system_instruction`。  
3. 先按第 3 节准备 `few_shots.yaml`。  
4. 用第 4 节模板让 AI 生成 `task_description`。  
5. 复制 `DEV_PLAN_GEO.md` 为 `DEV_PLAN_<行业>.md`，替换行业实体与目标。  
6. 按 `MANUAL_GEO.MD` 的命令与 SOP 启动、复核、迭代。  

到这里，你就完成了“新行业初始化”。

---

## 6. 三部分内容如何与 `prompt_config.json` 交互

### 6.1 `prompt_config.json` 的作用

`prompt_config.json` 是推理提示词的总装配置，决定模型“如何被提问”与“按什么约束输出”。

核心字段：

- `system_instruction`：定义模型角色与行为边界。
- `task_description`：定义本次抽取任务的实体类型与输出格式规则。
- `template`：把 `system_instruction`、`task_description`、`few_shots_str`、`input_text` 拼接成最终 prompt。
- `generation_config`：控制生成参数（如 `max_new_tokens`、`temperature`）。

### 6.2 上面三部分与 `prompt_config.json` 的映射关系

- 第 2 节产出的 `system_instruction` -> 直接写入 `prompt_config.json.system_instruction`
- 第 4 节产出的 `task_description` -> 直接写入 `prompt_config.json.task_description`
- 第 3 节准备的 `few_shots.yaml` -> 由程序读取并拼接为 `few_shots_str`，再注入 `template`

### 6.3 实际运行时的交互流程

1. `PromptBuilder` 读取 `prompt_config.json` 与 `few_shots.yaml`。  
2. `few_shots.yaml` 被转成 `few_shots_str`。  
3. 程序将 `system_instruction`、`task_description`、`few_shots_str`、`input_text` 注入 `template`。  
4. 得到最终 prompt，送入模型推理。  

结论：`system_instruction`、`task_description`、`few_shots` 三者都通过 `prompt_config.json` 的模板机制进入最终推理链路，缺一会影响抽取质量或格式稳定性。
