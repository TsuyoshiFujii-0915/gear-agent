# Gear Agent

Gear Agent は、Responses API 互換エンドポイントを使う最小構成の学習用コーディングエージェントです。
Codex の中核動作を理解しやすい Python コードとして表現することを目的にしています。

現在の実装は、対話型 TUI、モデル呼び出し、ツール実行、JSONL 形式のセッション保存、明示的な履歴コンパクションを備えています。

## 特徴

- モデル通信とツール実行は Python 標準ライブラリ中心の小さな実装
- Textual による対話型 TUI
- `uv` によるプロジェクト実行
- `gear` コマンドによる起動
- Responses API 互換の non-stream HTTP POST
- OpenAI と LM Studio などの互換エンドポイントを設定で切り替え
- 関数ツール呼び出しを含むエージェントループ
- ファイル読み取り、ファイル書き込み、シェル実行、パッチ適用ツール
- Tavily による Web 検索、Web ページ本文取得ツール
- JSONL によるセッションイベント保存

## 必要なもの

- Python 3.11 以上
- uv
- Docker
- Responses API 互換エンドポイント

Docker は `shell` ツールで使います。

## セットアップ

依存関係を同期します。

```bash
uv sync
```

プロジェクトスコープの設定ファイルを作成します。

```bash
uv run gear init
```

ユーザースコープの設定ファイルを作成する場合は次を使います。

```bash
uv run gear init --scope user
```

## 設定

設定ファイルは、カレントディレクトリから親ディレクトリへ向かって
プロジェクトスコープの `.gear/config.toml` を探索し、最初に見つかったものを読み込みます。
見つからない場合はユーザースコープの `~/.gear/config.toml` を読み込みます。

LM Studio など、API キーなしのローカル互換エンドポイントを使う例です。

```toml
[model]
url = "http://localhost:1234/v1/responses"
model = "local-model-id"
api_key_env = ""

[tool]
shell_tool = true
file_read = true
file_write = true
apply_patch = true
glob = true
grep = true
web_search = false
web_fetch = false

[web_search]
api_key_env = "TAVILY_API_KEY"
search_depth = "basic"
max_results = 5
timeout_seconds = 20
include_answer = true
include_raw_content = false

[web_fetch]
api_key_env = "TAVILY_API_KEY"
extract_depth = "basic"
content_format = "markdown"
timeout_seconds = 20
include_images = false
include_favicon = true
max_content_chars = 20000

[runtime]
workdir = "."
session_dir = ".gear/sessions"
network = "disabled"
max_iterations = 8
model_timeout_seconds = 120
```

OpenAI の Responses API を使う例です。`model` は利用可能なモデル ID に置き換えてください。

```toml
[model]
url = "https://api.openai.com/v1/responses"
model = "gpt-5.5"
api_key_env = "OPENAI_API_KEY"

[tool]
shell_tool = true
file_read = true
file_write = true
apply_patch = true
glob = true
grep = true
web_search = false
web_fetch = false

[web_search]
api_key_env = "TAVILY_API_KEY"
search_depth = "basic"
max_results = 5
timeout_seconds = 20
include_answer = true
include_raw_content = false

[web_fetch]
api_key_env = "TAVILY_API_KEY"
extract_depth = "basic"
content_format = "markdown"
timeout_seconds = 20
include_images = false
include_favicon = true
max_content_chars = 20000

[runtime]
workdir = "."
session_dir = ".gear/sessions"
network = "disabled"
max_iterations = 8
model_timeout_seconds = 120
```

`api_key_env` が空文字の場合、認証ヘッダーは送信しません。
環境変数名が指定されているのに値が存在しない場合は、設定エラーとして起動時に失敗します。

`[tool]` はモデルへ公開するツールを明示的に制御します。すべてのキーは必須の真偽値です。
未定義のキー、欠けているキー、真偽値以外の値は設定エラーとして起動時に失敗します。

Tavily Search を使う `web_search` を有効にする例です。

```toml
[tool]
shell_tool = true
file_read = true
file_write = true
apply_patch = true
glob = true
grep = true
web_search = true
web_fetch = true

[web_search]
api_key_env = "TAVILY_API_KEY"
search_depth = "basic"
max_results = 5
timeout_seconds = 20
include_answer = true
include_raw_content = false

[web_fetch]
api_key_env = "TAVILY_API_KEY"
extract_depth = "basic"
content_format = "markdown"
timeout_seconds = 20
include_images = false
include_favicon = true
max_content_chars = 20000
```

`web_search = true` の場合、`[web_search]` は必須です。`web_fetch = true` の場合、
`[web_fetch]` は必須です。`api_key_env` が指す環境変数に Tavily API キーが存在しない場合は、
設定エラーとして起動時に失敗します。

`[web_search]` の `search_depth` は `basic`、`advanced`、`fast`、`ultra-fast` のいずれかです。
`max_results` は 1 以上 20 以下の整数です。
`timeout_seconds` は 1 以上の整数です。

`[web_fetch]` の `extract_depth` は `basic` または `advanced` です。
`content_format` は `markdown` または `text` です。
`timeout_seconds` と `max_content_chars` は 1 以上の整数です。
各 Web 設定テーブルで未定義のキー、欠けているキー、型が合わない値は設定エラーとして起動時に失敗します。

## 使い方

通常起動します。

```bash
uv run gear
```

保存済みセッションを完全なセッション ID または一意な短縮 prefix で再開できます。
TUI に表示される先頭 8 文字も、一意であれば利用できます。

```bash
uv run gear resume 12345678
uv run gear resume 12345678-1234-5678-1234-567812345678
```

有効な `session_dir` で最後に更新されたセッションを再開する場合は、次を使います。

```bash
uv run gear resume --latest
```

設定ファイルや実行時設定は CLI オプションで一時的に上書きできます。

```bash
uv run gear --config custom.toml
uv run gear --workdir ../target-project --network enabled
uv run gear --max-iterations 4 --model-timeout-seconds 30
uv run gear --session-dir ../sessions resume --latest
```

Shell tool の Docker image はコード側で `python:3.11-slim` に固定しています。

対話中に使えるコマンドです。

```text
/compact
/quit
/exit
```

`/compact` は現在のセッション履歴をモデルへ送り、継続用の要約を `.gear/sessions` に保存します。

## ツール

モデルには次の関数ツールを渡します。

| ツール        | 役割                                                                      |
| ------------- | ------------------------------------------------------------------------- |
| `shell`       | Docker コンテナ内でシェルコマンドを実行します。`shell_tool` で制御します。 |
| `file_read`   | ワークスペース内の UTF-8 テキストファイルを読み取ります。                 |
| `file_write`  | ワークスペース内の既存親ディレクトリ配下へ UTF-8 テキストを書き込みます。 |
| `apply_patch` | ワークスペース内に unified diff パッチを適用します。                      |
| `glob`        | ワークスペース内のファイルとディレクトリを glob パターンで検索します。    |
| `grep`        | ワークスペース内の UTF-8 テキストファイルを正規表現で検索します。         |
| `web_search`  | Tavily Search API で Web 検索します。`web_search` で制御します。          |
| `web_fetch`   | Tavily Extract API で URL の本文を取得します。`web_fetch` で制御します。  |

ファイル操作とパッチ適用は、ワークスペース外のパスを明示的に拒否します。
ファイル検索ツールも、ワークスペース外のパスを明示的に拒否します。

## ディレクトリ構成

```text
.
|-- docs/
|   |-- PLAN.md
|   `-- decisions/
|-- src/
|   `-- gear_agent/
|       |-- __init__.py
|       |-- agent/
|       |   |-- compaction.py
|       |   |-- events.py
|       |   |-- history.py
|       |   `-- loop.py
|       |-- cli.py
|       |-- config.py
|       |-- errors.py
|       |-- model/
|       |   |-- client.py
|       |   |-- responses.py
|       |   `-- transport.py
|       |-- store/
|       |   |-- base.py
|       |   |-- jsonl.py
|       |   `-- memory.py
|       |-- tools/
|       |   |-- base.py
|       |   |-- configured.py
|       |   |-- filesystem.py
|       |   |-- filesystem_search.py
|       |   |-- patch.py
|       |   |-- registry.py
|       |   |-- runtimes.py
|       |   |-- shell.py
|       |   |-- web_fetch.py
|       |   |-- web_search.py
|       |   `-- validation.py
|       |-- tui.py
|       `-- tui_app.py
|-- tests/
|-- pyproject.toml
`-- uv.lock
```

`cli.py` と `config.py` は入口として読みやすいようにパッケージ直下へ残し、
エージェント実行、モデル通信、ツール、保存処理は責務ごとに分けています。

## テスト

標準の `unittest` でテストを実行します。

```bash
uv run python -m unittest discover -s tests
```

## 設計方針

- Responses API 互換エンドポイントへ、指定された URL、モデル、入力を明示的に送信します。
- stream 出力は扱わず、`stream = false` の non-stream リクエストだけを送ります。
- プロバイダごとの互換差分を自動吸収しません。
- Responses API で失敗した場合に Chat Completions API へ切り替えません。
- Docker が使えない場合にローカル実行へ暗黙フォールバックしません。
- 設定やレスポンス形状の不備は、原因と発生元を含む明示的なエラーとして扱います。

## 関連資料

- `docs/PLAN.md`: プロジェクトの目的、範囲、初期アーキテクチャ
- `docs/decisions/`: 採用済みの設計判断
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses/create
