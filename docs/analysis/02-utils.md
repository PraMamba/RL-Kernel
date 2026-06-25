# 工具层（Utils）

> 源码位置：`rl_engine/utils/`
> 文件数：1 个 | 总行数：75 行
> 最后更新：2026-06-13

## 1. 模块职责概述

工具层当前仅包含日志子系统，为整个 RL-Kernel 引擎提供统一的日志基础设施。核心功能是在标准 `logging.Logger` 基础上注入三个扩展方法：`info_once`（去重日志）、`warn_once`（去重警告）、`info_on_rank`（分布式 rank 过滤）。

在整体流水线中，日志模块处于第二底层——仅依赖 `platforms/device`（且为延迟导入），但被 8 个上层生产文件直接导入，是被依赖最多的单一模块。

上游消费者：`platforms/device.py` 在 `DeviceContext.__init__` 中调用 `logger.info_once` 和 `logger.warning`。

下游消费者：`kernels/ops/base.py`、`kernels/ops/cuda/*`、`kernels/registry.py`、`kernels/sampling.py`、`executors/bridge.py`、`executors/rollout.py` 均导入并使用 `logger` 单例。

## 2. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `logger.py` | 75 | 日志初始化、扩展方法注入、全局 logger 单例 |

## 3. 核心数据结构与接口

### RLEngineLogger

- **类型**：class（类型提示存根，实际是 `logging.Logger` 的运行时打补丁版本）
- **字段数**：0 个（存根类，无实际字段）
- **关键方法**：
  - `info_once(msg, *args)` → `None` — 对相同消息仅记录一次（通过 `lru_cache` 去重）
  - `warn_once(msg, *args)` → `None` — 对相同警告仅记录一次
  - `info_on_rank(msg, rank=0, *args)` → `None` — 仅在指定分布式 rank 上记录

### init_logger

- **类型**：function
- **签名**：`init_logger(name: str) -> RLEngineLogger`
- **职责**：获取或创建命名 logger，配置 StreamHandler（输出到 stdout），注入三个扩展方法

## 4. 算法与逻辑详解

### 去重日志实现

1. **入口** (`logger.py:20`): `_info_once` 被绑定为 `logger.info_once` 实例方法
2. **去重核心** (`logger.py:14-17`): `_log_once_impl` 使用 `@lru_cache(maxsize=None)` 装饰器，以 `(logger, level, msg, *args)` 元组为缓存键
3. **首次调用** (`logger.py:17`): 缓存未命中时调用 `logger.log(level, msg, *args, stacklevel=3)`
4. **后续调用**: 缓存命中直接返回 `None`，不执行任何日志操作

### 分布式 rank 过滤

1. **入口** (`logger.py:30`): `_info_on_rank` 在函数体内延迟导入 `device_ctx`
2. **过滤** (`logger.py:37`): 读取 `device_ctx` 的 `rank` 属性（可能不存在，`getattr` 默认为 0）
3. **日志** (`logger.py:38`): 仅当当前 rank 等于目标 rank 时调用 `self.info(msg, *args)`

### 关键常量与阈值

| 常量名 | 值 | 位置 | 用途 |
|--------|---|------|------|
| `_DEFAULT_FORMAT` | `"%(levelname)s %(asctime)s [%(name)s]: %(message)s"` | `logger.py:10` | 日志格式模板 |
| `_DATE_FORMAT` | `"%m-%d %H:%M:%S"` | `logger.py:11` | 日期格式 |
| `logger` | `init_logger("RL-Kernel")` 单例 | `logger.py:75` | 全局日志实例 |

## 5. 数据流（输入/输出）

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  调用方      │────▶│   logger     │────▶│   stdout     │
│ (registry,   │     │              │     │              │
│  bridge 等)  │     │ 方法：info,  │     │ 格式化输出   │
│              │     │ info_once,   │     │              │
│ 输入：消息   │     │ warn_once    │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 输入契约

- `msg` 参数应为 `%s` 风格的格式字符串（非 f-string），以便 `lru_cache` 去重正确工作

### 输出保证

- 日志输出到 `sys.stdout`，不使用 `stderr`
- `info_once` 和 `warn_once` 对相同 `(msg, *args)` 组合在进程生命周期内仅输出一次
- 日志级别默认为 `INFO`

## 6. 关键设计决策与不变量

1. **运行时方法注入而非子类** — 使用 `MethodType` 在标准 `logging.Logger` 实例上注入扩展方法，`RLEngineLogger` 仅用于类型提示。原因：避免 `logging.getLogger` 返回自定义类带来的兼容性问题，保持与标准日志基础设施的互操作性。

2. **lru_cache 去重而非集合** — 使用 `@lru_cache(maxsize=None)` 而非显式 `set()` 追踪已记录消息。原因：`lru_cache` 自动处理线程安全（CPython GIL 下），且无需手动管理缓存清理。

3. **延迟导入 device_ctx** — `_info_on_rank` 在函数体内导入 `device_ctx` 而非模块顶层。原因：打破 `logger.py` ↔ `device.py` 之间的循环导入（`device.py` 也导入 `logger`）。

### 不变量

- `logger` 名称固定为 `"RL-Kernel"`，全局唯一
- `propagate` 设为 `False`，不向父 logger 传播消息

## 7. 已知问题与限制

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | `info_on_rank` 依赖 `device_ctx.rank` 属性，但 `DeviceContext` 类未定义 `rank` 字段 | `logger.py:37` | `getattr` 默认返回 0，实际无法感知分布式 rank |

## 8. 相关测试覆盖

### 直接测试

| 测试文件 | 测试函数数 | 测试焦点 |
|---------|----------|---------|
| `rl_engine/tests/test_dispatch.py` | 1 (`test_logger_enhancements`) | 验证 logger 具有 info_once/warn_once 方法 |

### 间接测试

无专门的间接测试。所有使用 `logger` 的模块的测试间接验证了日志系统的可用性。
