# Gear Agent

Gear Agent は、Responses API 互換エンドポイントを使う最小構成の学習用コーディングエージェントです。
Codex の中核動作を理解しやすい Python コードとして表現することを目的にしています。

現在の実装は、対話型 TUI、非対話の単一タスク実行、モデル呼び出し、ツール実行、JSONL 形式のセッション保存、手動・自動の履歴コンパクションを備えています。

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

## コンテキスト予算と自動コンパクション

既存設定との互換性のため、`[context_budget]` がない場合は自動コンパクションを
無効にします。`gear init` も `auto_compaction = false` を明示して生成します。
無効時は、従来どおりの入力・イベント・送信動作を維持します。

有効にするには、使用するモデルのコンテキスト上限と、出力・推論用に確保する
トークン数を設定してください。以下の数値は設定例であり、特定モデルの仕様ではありません。

```toml
[context_budget]
auto_compaction = true
context_window_tokens = 128000
reserved_tokens = 16000
max_input_tokens = 80000
```

| キー | 意味 |
| --- | --- |
| `auto_compaction` | 自動要約と予算超過時の送信停止を有効にする真偽値。テーブルがある場合は必須。 |
| `context_window_tokens` | モデルのコンテキスト上限。正の整数で、有効時は必須。モデル名から推測しません。 |
| `reserved_tokens` | 出力・推論用の余裕。0以上かつ上限未満の整数で、有効時は必須。APIの出力上限設定は変更しません。 |
| `max_input_tokens` | 自動要約を開始する入力予算。省略時は上限から予約分を引いた値。指定時はその値以下の正の整数。 |

未知のキーや不正な型・範囲は起動時の設定エラーです。無効時は容量を未指定にできます。
`reserved_tokens` を無効時に省略すると0とし、`max_input_tokens` の省略は上記の計算規則を
使います。これらは既存設定を維持するための明示的な規則で、容量不明のモデルに上限値を
割り当てるものではありません。

各モデル反復の直前に、アダプターへ渡す実際の入力・指示・ツール定義を
`ContextBudgetManager` が計測します。基本指示、現在のリポジトリ指示、
履歴・チェックポイント、実際に再送する推論情報、ツール呼び出しと結果、
現在のユーザー入力、最終回答を促す再試行指示を含みます。
過去のツール結果は従来の履歴短縮処理後に計上し、同じターン内の結果は従来どおり全文を計上します。

初期の `ByteTokenEstimator` は、各成分を空白なしのJSONにしたUTF-8バイト数を
1バイト＝1トークンとして数え、25%を加えて切り上げます。さらにリクエストの枠組み用に
256トークンを加算します。安全側の見積もりであり、未知のトークナイザーや暗号化された
状態の実際の消費量を保証するものではありません。外部トークナイザーへの通信や、
前回レスポンスのusageを次回入力サイズとして使う処理はありません。
見積もり方式と容量判定は `TokenEstimator` インターフェースで分離しています。

入力予算を超えると、その反復で一度だけ既存のテキスト要約を行います。
最新チェックポイント以降の有効なセッション内容を要約し、現在のユーザー入力を
そのまま残して入力を再構築します。再試行用指示と最新のリポジトリ指示も反映して
再計測し、まだ超過していれば `context_budget_exceeded` エラーで送信を止めます。
要約対象には現在のユーザー入力も含まれるため、要約内の目標と末尾のユーザー入力が
意味的に重複する場合があります。現在の要求を逐語的に保持するための挙動であり、
予算計測には両方を含めます。評価時にもこのコンテキスト構成を前提にしてください。
要約自体のリクエストも「上限 − 予約分」に収まる必要があります。ツール結果などで
急激に増大し、要約リクエストも収まらない場合は送信せず失敗します。
`max_input_tokens` を低めに設定すると、テキスト要約の追加情報に使える余裕が増えます。

自動要約も `compaction_summary` チェックポイントを保存し、`trigger = "automatic"` で
識別できます。元のJSONLは書き換えず、チェックポイント以前の暗号化推論状態は再送しません。
`gear resume` も起動時の有効な設定で同じ予算判定を行います。
手動の `/compact` は従来どおり明示的な要約コマンドとして利用できます。

有効時は `ContextBudgetEvaluated` イベントで、反復番号、判定段階
（`before` / `compaction` / `after`）、成分別見積もり、入力上限、予約分、
自動要約の発動と予算エラーを取得できます。合計入力見積もりは予約分を含まず、
予約分を差し引いた入力上限と比較します。診断にはコンテキスト本文を含めず、
ストリーム断片ごとの保存も行いません。現在のTUIは診断を受信しますが、表示を追加しません。
`ReasoningReplayEvaluated` は予算判定が完了した後、モデルへの送信直前に通知します。
自動要約で破棄された候補の推論再利用数は通知せず、再構築した入力の診断で置き換えます。
予算エラーで送信しない場合も、その候補の再利用を通知しません。

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

### 非対話で1タスクを実行する

`run` は新規セッションで1回のユーザーターンを実行します。モデルのツール呼び出しは、
最終回答またはエラーになるまで通常の `AgentLoop` 内で続きます。

```bash
uv run gear run --prompt "Fix the failing test"
uv run gear run --prompt-file task.md
uv run gear --workdir ../target-project --max-iterations 12 run --prompt-file task.md
uv run gear --config custom.toml --session-dir ../sessions --model-timeout-seconds 60 run --prompt "Explain this project"
```

`--prompt` と `--prompt-file` はどちらか一方が必須です。ファイルはUTF-8として読み、
空白だけのタスクは拒否します。設定・実行時オプションは従来どおり `run` の前に置きます。
設定探索、相対パスの基準（起動ディレクトリ）、workspace境界、Docker shellとnetwork設定、
有効なツール、モデルadapter、AGENTS.md、context budgetと自動compactionはTUIと共通です。
`model.stream = true` も利用でき、Textualや対話イベントループは起動しません。

出力の契約は次のとおりです。

- 成功時のstdout: 確定した最終回答を1回だけ出力し、改行を1つ付加します。
- stderr: 実行前に `session_id=<UUID>` を1行出力します。失敗時はさらに
  `{"error":{"type":"...","origin":"...","message":"..."}}` のJSON診断を出します。
- 進捗、推論、streamの途中テキストは出力しません。進捗は常に無効なのでquiet指定は不要です。
- 失敗時のstdoutは空です。部分実行後にタスク全体を自動再試行しません。

| 終了コード | 意味 |
| --- | --- |
| 0 | 最終回答を得て正常終了 |
| 1 | 設定、promptファイル、実行環境、ローカルI/Oのエラー |
| 2 | CLI引数のエラー（入力の未指定・重複など） |
| 3 | 実行中のモデル、agent、tool、repository context、context budgetの構造化エラー |

回復可能なtoolエラーは通常どおりモデルに返し、訂正して完了できれば成功です。
中断時は通常のプロセス中断動作に従います。

```bash
uv run gear run --prompt-file task.md > answer.txt 2> run.log
```

履歴は通常の `session_dir/<UUID>.jsonl` に保存され、`gear resume <UUID>` でTUIから
再開できます。診断は認証情報を伏せ、任意のレスポンス本文を含むerror detailsは出しません。
実行中の構造化エラーは、同じ診断内容を既存の `turn_error` として一度だけ保存し、
再開時にも停止理由を表示します。失敗の記録自体を書き込めない場合は、元のタスクエラーと
終了コードを維持し、stderrのJSONに `persistence_error` を追加します。
回答とセッション内のタスク・モデル・tool本文には、通常の保存・出力ルールが適用されます。

Pythonからは `gear_agent.headless.run_task` の `RunResult` と `RunSpec` で、
モデル名・credentialを除いたendpoint fingerprint、adapterとcapabilities、workspace、
tool設定、runtime制限、context budget、prompt source、session IDを参照できます。
API keyと生のendpoint URLはこのmetadataに含めません。ベンチマーク用artifactファイルの
生成やheadless resumeはこのコマンドの対象外です。

### 対話中の操作

Shell tool の Docker image はコード側で `python:3.11-slim` に固定しています。

対話中に使えるコマンドです。

```text
/compact
/quit
/exit
```

`/compact` は現在のセッション履歴をモデルへ送り、継続用の要約を `.gear/sessions` に保存します。

## 評価実行の成果物

`gear run` は実行ごとに `<workspace>/.gear/runs/<run-id>/` を作成し、
設定・結果の `run.json`、計測値の `metrics.json`、独立したセッション履歴
`events.jsonl`、成功時の `final.txt`、Git 状態と追跡済みファイルの差分を保存します。
実行 ID はセッション ID と独立しています。保存先を指定する例です。

```bash
uv run gear run --prompt-file task.txt --run-dir ../results/case-001
```

`--run-dir` は今回の実行専用ディレクトリです。既存の保存先は上書きしません。
`run.json` の `status` が `success` または `failure` のとき記録が完了しています。
`running` のままの記録は未完了です。実行に失敗した場合も履歴と計測値を保存し、
モデルの途中出力を最終回答にはしません。

使用量の未報告項目は `null` とし、モデル要求・ツール実行時間は単調時計で計測します。
自動コンパクションのモデル要求も計測対象です。Git の差分は開始時 HEAD 基準なので、
開始時に未コミット変更があれば既存の変更も含みます。秘密情報を除去した設定・診断を
保存しますが、会話のコピーには既存仕様の暗号化 reasoning state が残ることがあります。
詳細は [成果物のスキーマと制約](docs/run-artifacts.md) を参照してください。

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
