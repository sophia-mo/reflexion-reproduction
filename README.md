# [NeurIPS 2023] Reflexion: Language Agents with Verbal Reinforcement Learning

This repo holds the code, demos, and log files for [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) by Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao. 

![Reflexion RL diagram](./figures/reflexion_rl.png)

![Reflexion tasks](./figures/reflexion_tasks.png)

We have released the LeetcodeHardGym [here](https://github.com/GammaTauAI/leetcode-hard-gym)

## To Run: reasoning (HotPotQA)

We have provided a set of notebooks to easily run, explore, and interact with the results of the reasoning experiments. Each experiment consists of a random sample of 100 questions from the HotPotQA distractor dataset. Each question in the sample is attempted by an agent with a specific type and reflexion strategy.

### Setup

To get started:

1. Clone this repo and move to the HotPotQA directory:

```bash
git clone https://github.com/noahshinn/reflexion && cd ./hotpotqa_runs
```

2. Install the module dependencies into your environment:

```bash
pip install -r requirements.txt
```

3. Set `OPENAI_API_KEY` environment variable to your OpenAI API key:

```bash
export OPENAI_API_KEY=<your key>
```

#### Agent Types

Agent type is determined by the notebook you choose to run. The available agent types include:

- `ReAct` - ReAct Agent

- `CoT_context` - CoT Agent given supporting context about the question 

- `CoT_no_context` - CoT Agent given no supporting context about the question

The notebook for each agent type is located in the `./hotpot_runs/notebooks` directory.

#### Reflexion Strategies

Each notebook allows you to specify the reflexion strategy to be used by the agents. The available reflexion strategies, which are defined in an `Enum`, include:

- `ReflexionStrategy.NONE` - The agent is not given any information about its last attempt. 

- `ReflexionStrategy.LAST_ATTEMPT` - The agent is given its reasoning trace from its last attempt on the question as context.

- `ReflexionStrategy.REFLEXION` - The agent is given its self-reflection on the last attempt as context. 

- `ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION` -  The agent is given both its reasoning trace and self-reflection on the last attempt as context.

### To Run: decision-making (AlfWorld)

Clone this repo and move to the AlfWorld directory

```bash
git clone https://github.com/noahshinn/reflexion && cd ./alfworld_runs
```

Specify the run parameters in `./run_reflexion.sh`.
`num_trials`: number of iterative learning steps
`num_envs`: number of task-environment pairs per trial
`run_name`: the name for this run
`use_memory`: use persisting memory to store self-reflections (turn off to run a baseline run)
`is_resume`: use logging directory to resume a previous run
`resume_dir`: the logging directory from which to resume the previous run
`start_trial_num`: if resume run, then the trial number of which to start

Run the trial

```bash
./run_reflexion.sh
```

The logs will be sent to `./root/<run_name>`.

### Another Note

Due to the nature of these experiments, it may not be feasible for individual developers to rerun the results as GPT-4 has limited access and significant API charges. All runs from the paper and additional results are logged in `./alfworld_runs/root` for decision-making, `./hotpotqa_runs/root` for reasoning, and `./programming_runs/root` for programming

### Other Notes

Check out the original implementation [here](https://github.com/noahshinn/reflexion-draft)

Read one of the original blog posts [here](https://nanothoughts.substack.com/p/reflecting-on-reflexion)

Check out an [Appl](https://github.com/appl-team/appl) implementation [here](https://github.com/appl-team/reppl/tree/main/reflexion).

Check out an interesting type-prediction implementation here: [OpenTau](https://github.com/GammaTauAI/opentau)

For all questions, contact [noahrshinn@gmail.com](noahrshinn@gmail.com)

### Cite

```bibtex
@misc{shinn2023reflexion,
      title={Reflexion: Language Agents with Verbal Reinforcement Learning}, 
      author={Noah Shinn and Federico Cassano and Edward Berman and Ashwin Gopinath and Karthik Narasimhan and Shunyu Yao},
      year={2023},
      eprint={2303.11366},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## ALFWorld Reflexion 2 个任务 x 2 轮尝试
### 适配新模型的代码修改
#### Modification 1

在 `generate_reflections.py` 中，将导入、函数签名和反思调用分别改为：
```python
from utils import get_chat

def update_memory(trial_log_path, env_configs, model):
    # 保留原有函数体，仅替换生成 reflection 的那行
    reflection = get_chat(reflection_query, model=model, max_tokens=256)
```
原代码调用的 get_completion() 写死了 text-davinci-003，该模型已退役

#### Modification 2

在 `main.py` 中，将更新记忆的两行改为：
```python
if args.use_memory and trial_idx + 1 < args.num_trials:
    env_configs = update_memory(trial_log_path, env_configs, args.model)
```
这同时避免最后一轮结束后生成不会再使用的反思。所选模型需要兼容当前代码的 Chat Completions 接口及 stop、temperature、max_tokens 参数。

#### Modification 3

单次尝试步数把 `alfworld_trial.py` 的 `while cur_step < 49` 改成30。这里 think: 也占一次循环和模型调用，不要把上限压得过低。

#### Modification 4

新版 ALFWorld 通过 `get_environment()` 获取环境类，不能再直接从模块中查找 `AlfredTWEnv`

### 运行
#### 配置环境

在 Ubuntu 中
```bash
conda create -n reflexion python=3.9 -y
conda activate reflexion
cd /mnt/d/Reproductions/reflexion-reproduction/alfworld_runs
python -m pip install -r requirements.txt

# 安装 make, GCC 等编译工具
sudo apt update
sudo apt install -y build-essential
# 检查
gcc --version
make --version

python -m pip install "alfworld==0.4.2" pyyaml "spacy==3.7.5" "thinc==8.2.5" "numpy==1.26.4" --only-binary=spacy,thinc,numpy
```

#### 下载任务数据到 D:\Reproductions\alfworld-data

```bash
export ALFWORLD_DATA="/mnt/d/Reproductions/alfworld-data"
mkdir -p "$ALFWORLD_DATA"
alfworld-download
```

| 下载文件 | 用途 |
| ------- | ---- |
| json_2.1.1_json.zip | 任务描述、轨迹等 JSON 数据 |
| json_2.1.1_pddl.zip | 任务的 PDDL 规划描述 |
| json_2.1.2_tw-pddl.zip | TextWorld 使用的环境文件 |
| mrcnn_alfred_objects_sep13_004.pth | 视觉环境的物体检测模型 |

```
alfworld-data\
├── json_2.1.1\
│   ├── train\
│   ├── valid_seen\
│   └── valid_unseen\
├── detectors\
│   └── mrcnn_alfred_objects_sep13_004.pth
└── logic\
    ├── alfred.pddl
    └── alfred.twl2
```

如果连接超时可从浏览器下载，原脚本最后的 logic 文件复制步骤可执行：
```bash
python - <<'PY'
import os
import shutil
from pathlib import Path
from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH

logic = Path(os.environ["ALFWORLD_DATA"]) / "logic"
logic.mkdir(parents=True, exist_ok=True)
for source, name in [
    (ALFRED_PDDL_PATH, "alfred.pddl"),
    (ALFRED_TWL2_PATH, "alfred.twl2"),
]:
    target = logic / name
    if not target.exists():
        shutil.copyfile(source, target)
    print(target)
PY
```

#### 运行脚本
```bash
read -rs OPENAI_API_KEY
# 输入 API KEY
export OPENAI_API_KEY

cd /mnt/d/Reproductions/reflexion-reproduction/alfworld_runs
export ALFWORLD_DATA="/mnt/d/Reproductions/alfworld-data"
python main.py --num_trials 2 --num_envs 2 --run_name "smoke_reflexion_2x2" --use_memory --model "gpt-4o-mini"
```

### 问题
1. 模型生成的动作带了多余的 >
2. 模型编造观察结果

有待探究原因，检查 paper 中有没有说到这个问题

**主要是模型把输入当成了“继续写一段交互记录”，而程序期待的是“只返回下一条动作”。** 两者的输出约定没有对齐。

你输入的 few-shot 是这样的：

```text
> go to drawer 1
The drawer 1 is closed.
> open drawer 1
You open the drawer 1...
```

这里同时包含了动作前缀、动作和环境反馈。模型能模仿整段格式，但不一定知道哪些部分只能由程序提供。

**为什么多输出一个 `>`？**

代码把提示词拼成：

```python
str(env_history) + ">"
```

它期待模型直接续写 `go to drawer 1`。但 Chat 模型是在生成一个**新的回复消息**，不保证将回复当作用户消息最后那个 `>` 的直接延续，因此可能完整输出：

```text
> go to drawer 1
```

之后日志程序又加一个 `>`，就显示成：

```text
> > go to drawer 1
```

**为什么会编造动作或环境反馈？**

模型本来就负责生成动作，例如 `go to drawer 1`。真正的问题分为两种：

| 输出 | 问题 |
|---|---|
| `open desk 1` | 生成了环境可能不支持的动作 |
| `You open drawer 1. Inside, you see...` | 越过职责，生成了本该由环境返回的观察 |

当前调用把示例和历史都放在一条 `user` 消息中，没有明确区分“模型负责动作，环境负责观察”；程序也没有校验返回内容，就直接传给 `env.step()`。

从日志看，还出现了一个恶性循环：

```text
输出带 > 的动作
  ↓
环境返回 Nothing happens.
  ↓
模型没有正确纠错，反而自行描述“打开了抽屉”
  ↓
程序又把这段描述当成动作
  ↓
环境再次返回 Nothing happens.
```

**`stop=['\n']` 只能限制输出到换行处，不能保证这一行是动作。** 一行编造的观察同样可以通过。

所以需要分两层处理：

- **格式层**：去掉前导 `>`，明确要求只输出一条动作或 `think:`。
- **语义层**：检查输出是不是允许的动作形式，不能把 `You open...` 之类的叙述交给环境执行。

这些是根据输入结构和日志得出的解释，无法从日志直接确认模型内部的具体原因。换成 `gpt-4o-mini` 后，原来依赖模型自行遵守的格式约定，需要更明确地落实到提示和代码中。