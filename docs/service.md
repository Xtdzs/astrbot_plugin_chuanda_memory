# 独立部署与 API v1

## 启动

在仓库目录安装依赖，设置环境变量后启动。模型名称填写你实际提供商支持的名称：

```bash
python -m pip install -r requirements.txt
export CHUANDA_MEMORY_TOKEN='replace-with-a-long-random-token'
export CHUANDA_MODEL_BASE_URL='https://api.deepseek.com/v1'
export CHUANDA_MODEL_API_KEY='your-model-api-key'
export CHUANDA_MODEL='your-available-model-name'
python -m chuanda_memory serve --data-dir /srv/chuanda-memory
```

PowerShell 对应语法为 `$env:CHUANDA_MEMORY_TOKEN = '...'`。服务默认监听 `127.0.0.1:8765`；远程机器使用受控网络或 HTTPS 反向代理。

AstrBot 插件配置改为：

```json
{
  "mode": "remote",
  "service_url": "http://127.0.0.1:8765",
  "service_token": "replace-with-the-same-long-random-token"
}
```

也可以在 AstrBot 进程中设置 `CHUANDA_MEMORY_TOKEN` 替代配置项。独立服务使用自己的 `CHUANDA_MODEL_*`，不会读取 AstrBot 的模型配置；插件中的 `memory_provider_id` 仅作用于 local 模式。

若 AstrBot 在容器里，`127.0.0.1` 指 AstrBot 容器自己。应将两个容器放入同一网络并用服务名访问，或使用可达的宿主地址。

## Docker

复制 `.env.example` 为 `.env` 并填入真实配置，再运行：

```bash
docker compose up -d --build
```

Compose 将记忆写到命名持久卷，宿主端口只绑定 `127.0.0.1`。更新时重建容器即可保留数据；不要执行删除数据卷的操作。`.env` 与数据库都不进入发布包。

## 接口

除 `GET /health` 外，都要求请求头 `Authorization: Bearer <CHUANDA_MEMORY_TOKEN>` 和 JSON 请求体，最大64 KiB。

令牌代表可信适配器权限，拥有该令牌即可访问服务数据。不要把 API 或令牌直接作为让群友随意传入身份参数的模型工具。成员身份、范围、管理员标志必须由适配器从真实消息事件绑定，不能采信模型或聊天文本提供的值。

| 路径 | JSON 参数 | 返回 |
| --- | --- | --- |
| `POST /v1/observe` | namespace, scope, message_id, sender_id, sender_name, text；可选 timestamp, provider_id | accepted：是否新入队 |
| `POST /v1/context` | namespace, scope, sender_id, query | current_sender_id, person_names, memories |
| `POST /v1/profile` | namespace, scope, sender_id | 当前范围内该人物档案和有效记忆 |
| `POST /v1/forget` | namespace, scope, sender_id, memory_id；可选 admin:boolean | deleted |
| `POST /v1/pause` | namespace, scope, sender_id；可选 erase:boolean | ok |
| `POST /v1/resume` | namespace, scope, sender_id | ok |

`@group` 为群体记忆的保留主体，不可作为消息发送者。读取当前范围的共同事件可向 profile 传 `sender_id: "@group"`。API v1 不接受客户端传入候选记忆或直接写任意人物事实，后台模型输出统一经过校验。

观察示例：

```json
{
  "namespace": "[\"aiocqhttp\",\"my-adapter-id\",\"my-bot-id\"]",
  "scope": "group:123456",
  "message_id": "987654321",
  "sender_id": "12345",
  "sender_name": "小王",
  "text": "技术问题回答短一点，先给结论就好"
}
```

独立服务不会发送 QQ 消息。其他宿主（包括 DSH）要接入时，适配器应：

1. 观察真实入站消息并提交 observe。
2. 请求模型前，用同一 namespace/scope 与真实当前发送者 ID 请求 context。
3. 将结果作为低信任、临时资料加入当轮上下文；人格规则独立配置，不把历史消息升级成系统指令。
4. 服务不可用时继续正常聊天；由适配器执行查看、删除与管理员授权。

## 迁移和恢复

local 模式迁移到独立服务：停止 AstrBot 插件 → 执行备份 → 将数据库或完整数据目录复制到服务的数据目录 → 启动独立服务 → 将插件切到 remote。保持平台/适配器/机器人身份配置不变。

恢复备份前停止所有连接该数据库的进程，将选中的备份恢复为 `memory.sqlite3`。不要在有写入进程时单独替换 SQLite 主文件；在线备份用提供的 backup 命令。

`profile` 离线查询可用于运维诊断：

```bash
python -m chuanda_memory profile --data-dir /srv/chuanda-memory \
  --namespace '["aiocqhttp","my-adapter-id","my-bot-id"]' \
  --scope group:123456 --person-id 12345
```
