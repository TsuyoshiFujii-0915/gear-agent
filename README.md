# Gear Agent

Gear Agent は、Responses API 互換エンドポイントを使う最小構成の学習用コーディングエージェントです。
Codex の中核動作を理解しやすい Python コードとして表現することを目的にしています。

現在の実装は、対話型 TUI、モデル呼び出し、ツール実行、JSONL 形式のセッション保存、明示的な履歴コンパクションを備えています。

## 特徴

- モデル通信とツール実行は Python 標準ライブラリ中心の小さな実装
- Textual による対話型 TUI
- `uv` によるプロジェクト実行
- `gear` コマンドによる起動
- Responses API 互換の JSON / SSE HTTP POST
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
reasoning_replay = "none"
stream = false

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
model_stream_idle_timeout_seconds = 60
```

OpenAI の Responses API を使う例です。`model` は利用可能なモデル ID に置き換えてください。

```toml
[model]
url = "https://api.openai.com/v1/responses"
model = "gpt-5.5"
api_key_env = "OPENAI_API_KEY"
reasoning_replay = "encrypted"
stream = true

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
model_stream_idle_timeout_seconds = 60
```

`api_key_env` が空文字の場合、認証ヘッダーは送信しません。
環境変数名が指定されているのに値が存在しない場合は、設定エラーとして起動時に失敗します。

`stream = true` は Responses API のtyped SSE eventsを逐次受信し、モデル層で完成済みの
Responses objectへ組み立ててからAgentLoopへ返します。`false` は従来のJSON responseを
使用します。既存設定との互換性のため、`stream` 欠落時は `false` です。
streamingを有効にする場合、`runtime.model_stream_idle_timeout_seconds` は必須です。
これはstream bytesを受信しない最大秒数で、接続・書き込み等に使う
`model_timeout_seconds` とは独立しています。途中切断、idle timeout、terminal errorから
non-stream requestへの自動再送は行いません。

OpenAI、Bearer token認証を受け付けるAzure OpenAI、LM Studio等で同じResponses SSE
event schemaが提供される場合はprovider固有分岐なしで処理します。未知の追加eventは無視しますが、`response.failed`、
`response.incomplete`、`error`、成功terminal前のEOFは明示的なエラーになります。

`reasoning_replay` は `none` または `encrypted` を指定します。
`none` は opaque な reasoning state を要求・再送しません。`encrypted` は Responses リクエストへ
`include = ["reasoning.encrypted_content"]` 相当の指定を加え、暗号化された reasoning item を
手動履歴で再送します。既存の設定との互換性のため、設定欠落時は `none` として
読み込みます。明示した未対応値は設定ロード時に失敗します。

暗号化された state は、protocol、credential を除いた設定 endpoint URL の SHA-256
identity、model ID が保存時と現在で完全一致する場合にだけ再利用されます。endpoint
または model を変更した場合や、scope
metadata を持たない旧JSONLセッションでは `encrypted_content` のみをモデル入力から除去し、
reasoning summary、メッセージ、ツール履歴は維持します。保存済みJSONL自体は変更しません。
query はendpoint identityに含めますが、設定した API key を含むquery componentは
固定表現に置き換えてから計算します。そのため、Azure Responses API の `api-version`
のようなqueryは利用でき、credential の派生値はscope metadataに残りません。userinfoと
fragment を含むendpoint URLは、`encrypted` の場合は設定エラーとして拒否します。
この設定は OpenAI Responses API の
[`reasoning.encrypted_content`](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
に対応しています。

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

## リポジトリの指示（AGENTS.md）

Gear は、設定と `--workdir` を反映したワークスペースを探索の境界として、
ルートの `AGENTS.md` を毎回のエージェントリクエストに含めます。
ワークスペースより上のディレクトリは探索しません。

ツール実行後は、次の情報から確認できるパスについて、ルートから対象ディレクトリまでの
`AGENTS.md` を次のリクエストに追加します。

| ツール | 指示の適用範囲を決める情報 |
| --- | --- |
| `file_read` / `file_write` | 結果の `path` が指すファイルの親ディレクトリ |
| `apply_patch` | 結果の `changed_files` に含まれる各ファイルの親ディレクトリ |
| `glob` | 結果に含まれるディレクトリ、またはファイルの親ディレクトリ |
| `grep` | 結果に含まれる一致ファイルの親ディレクトリ |
| `shell` | 実行した呼び出しの `workdir`（非ゼロ終了・タイムアウトを含む） |

たとえば `backend/src/service.py` を読み取った後は、存在するものに限り
`AGENTS.md` → `backend/AGENTS.md` → `backend/src/AGENTS.md` の順に含めます。
重複は除去し、浅い階層から深い階層へ、同じ深さではワークスペース相対パス順に並べます。
各指示は自身のディレクトリ配下にだけ適用され、同じ範囲では深い階層を優先します。

構造化された `error` を返したツールや未実行の呼び出しは範囲に加えません。
シェルのコマンド本文・標準出力・検索で返されなかったパスからは範囲を推測しません。
一度確認した範囲はセッション中保持し、`/compact` 後や再開時も元のツール履歴から復元します。
各ツールは実行時に確定した実体のディレクトリを、ワークスペース相対パスの
`resolved_scope_paths` として結果へ保存します。元のリンクが後で付け替え・削除されても、
過去の操作に対応する範囲は変わりません。保存された実体のディレクトリ自体が
シンボリックリンクに置き換わった場合は、別の指示へ転送せず明示的なエラーにします。
最初のリクエストや未知のパスに直接行う編集には、そのパスの深い階層の指示はまだ含まれません。

指示は UTF-8 の通常ファイルで、**1ファイル32 KiB、合計128 KiB** までです
（ファイル内容のバイト数、上限値を含む）。超過時は切り詰めずにエラーとします。
文字コード不正、権限エラー、ワークスペース外への参照も明示的なエラーとなります。
内部を指すものも含め、`AGENTS.md` 自体のシンボリックリンクは拒否します。
対象パスの内部シンボリックリンクは実体のパスへ解決し、その階層の指示を適用します。
指示ファイルが存在しない場合や、以前参照したディレクトリが削除済みの場合は正常に扱います。

Gear 本来の指示の後に、相対 `path`・`scope` を持つ `repository_instructions` ブロックを
追加します。本文と属性を XML エスケープし、区切りを明確にします。
各リクエストで再読み込みするため、実行中の作成・変更・削除も次のリクエストに反映されます。
指示の自動読み込みはチャットやセッションイベントに保存せず、API レスポンスが返す
リクエスト由来の `instructions` フィールドも保存対象から除外します。
ツールによる明示的な読み書きやモデルの応答は、従来どおり履歴に残ります。
既存セッションの移行は不要です。`resolved_scope_paths` のない旧形式の結果に限り、
元のパスを現在のワークスペースで解決します。旧形式には実行時の実体が記録されていないため、
当時のリンク先は復元できません。フィールドが存在する場合は空配列も有効な記録として扱い、
不正な値ではエラーとし、旧形式の処理には切り替えません。
`/compact` 自体は従来の要約専用指示を使用します。

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
|       |-- repository.py
|       |-- model/
|       |   |-- client.py
|       |   |-- events.py
|       |   |-- responses.py
|       |   |-- streaming.py
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

`pytest` で `unittest` 形式を含む全テストを実行します。

```bash
uv run pytest -q
```

## 設計方針

- Responses API 互換エンドポイントへ、指定された URL、モデル、入力を明示的に送信します。
- `model.stream` によりnon-stream JSONとSSE streamingを明示的に選択します。
- streamingでもAgentLoopには1つのcanonical completed responseだけを返します。
- プロバイダごとの互換差分を自動吸収しません。
- Responses API で失敗した場合に Chat Completions API へ切り替えません。
- Docker が使えない場合にローカル実行へ暗黙フォールバックしません。
- 設定やレスポンス形状の不備は、原因と発生元を含む明示的なエラーとして扱います。

## 関連資料

- `docs/PLAN.md`: プロジェクトの目的、範囲、初期アーキテクチャ
- `docs/decisions/`: 採用済みの設計判断
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses/create
