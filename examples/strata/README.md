# Strata の Qwen モデルを Gear から使う

推論先を選び、対応する TOML をコピーして使う。どの配置でも Gear は同じ Responses adapter でリクエストする。Gear のファイル操作は **クライアントの workspace**、shell は Gear が使う Docker、Web ツールも Gear 側で実行する。SSH 先のディスクを編集する機能はない。この検証では Gear 側の `file_read` だけを有効にし、Strata の MCP サーバーと内蔵ツールは無効のままにする。

| Strata の配置 | Gear から見た接続先 | 設定例 |
| --- | --- | --- |
| 同一ホストで直接起動 | `http://127.0.0.1:8080/v1/responses` | [local.toml](local.toml) |
| 同一ホストの Docker | `http://127.0.0.1:8080/v1/responses` | [docker.toml](docker.toml) |
| SSH 先で直接起動 | `http://127.0.0.1:18080/v1/responses` | [ssh.toml](ssh.toml) |
| SSH 先の Docker | `http://127.0.0.1:18080/v1/responses` | [ssh-docker.toml](ssh-docker.toml) |

Google Colab は外部クライアントからの利用条件を確認する段階で実機検証を停止した。[Colab の成立条件と検証記録](colab.md)

`model.url` は `/v1/responses` まで含める。HTTP(S) ゲートウェイを使う場合も、Bearer ヘッダー、POST 本文、JSON/SSE が同じ経路を通ることが前提になる。追加のゲートウェイ認証が必要な経路は個別に検証する。`runtime.network = "disabled"` は Gear の Docker shell の設定で、モデル HTTP には適用されない。

## 推論ホストを用意する

手順の基準は Strata **v0.2.7**、commit `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d` の Qwen Responses 実装。実機記録には実際に使った版を書く。Strata の Terminal、engine、モデルにはそれぞれ独立したバージョンや識別子がある。[Strata v0.2.7](https://github.com/Niko1221/Strata/releases/tag/v0.2.7) · [対応機種とインストール](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/INSTALL.md)

| 推論ホスト | 使用する経路と条件 |
| --- | --- |
| Apple Silicon Mac、macOS 15 以降 | ネイティブの Metal engine。Mac の Docker は Metal GPU にアクセスできない。 |
| Linux x86-64、対応 NVIDIA | ネイティブまたは GPU コンテナ。下の CUDA 13 例は RTX 20〜50 系、driver 580 以降、NVIDIA Container Toolkit を前提とする。旧カード用 CUDA 12 の対応表は公式資料を確認する。 |
| Linux、対応 Radeon / Intel Arc | GPU 世代に合う ROCm / SYCL イメージ、デバイス・グループ設定が必要。Arc は準備済み pack / MTP も必要。 |

ドライバーとコンテナランタイムは推論ホストで確認する。Gear クライアントにモデルを保存する必要があるのは同じホストで推論する場合だけである。[GPU 別の公式 Docker 手順](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/DOCKER.md) · [NVIDIA Container Toolkit のセットアップ](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 直接起動（同一ホスト・SSH 先ホスト）

1. 推論ホストに v0.2.7 の Terminal を用意する。Linux / Mac の新規導入では、リリースの `install.sh` を確認し、`STRATA_FEED=https://github.com/Niko1221/Strata/releases/download/v0.2.7 sh install.sh` で取得元も固定する。既存環境では先に `strata --version` と `strata launch --status` を確認する。[インストーラー](https://github.com/Niko1221/Strata/blob/v0.2.7/packaging/install.sh)
2. サーバー管理者が推論用の十分に長いキーを選び、推論ホストと Gear 側の環境変数 `STRATA_API_KEY` に同じ値を設定する。これは自分のサーバーを保護するキーで、クラウド API キーの購入は不要。秘密管理ツールや非表示入力を使い、キーをコマンド履歴・設定例・ログに書かない。
3. 次の例は Qwen IQ3_XXS、32K を選ぶ。まず plan だけを実行し、取得容量・保存先・GPU/ホストメモリ・SSD 空き容量を確認する。モデルは量子化によって数十 GB に達する。Terminal は既存ライブラリを探索する。新規取得先を指定する場合は plan と起動の両方に `--data-dir DIR` を加える。[Terminal の起動オプション](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/TERMINAL_LAUNCH.md)

```sh
strata launch --plan --family qwen --model iq3_xxs --context 32k \
  --access local --port 8080 --api-key-env STRATA_API_KEY
```

管理者が表示された取得とロードを許可した場合に限り、同じ引数から `--plan` を外して対話で起動する。`--restart` は既存の推論を中断するのでこの手順では使わない。すでにロード済みなら下のメタデータ確認へ進む。

### Docker（同一ホスト・SSH 先ホスト）

Linux NVIDIA の例。ホストで `nvidia-smi` と Docker の GPU アクセスを確認する。管理者が用意した既存の IQ3_XXS GGUF 全シャードを含むディレクトリを `GGUF_DIR` に、リポジトリ外の読み取り可能な秘密ファイルの絶対パスを `KEY_FILE` に設定する。ファイルのキー（16文字以上）を Gear の `STRATA_API_KEY` に渡す。コンテナは `API_KEY_FILE` と `API_KEY` の両方の方式に対応する。

初回は image の取得、pack の作成（IQ3_XXS の例で約 40 GB）、追加の draft layer（約 6 GB の取得）があり得る。管理者が容量と取得を許可してから実行する。`MODEL_PATH` は既存の重みを指定し、専用 volume `/data` はモデル関連ファイル・preset・キーを永続化する。[公式イメージのストレージと認証](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/DOCKER.md#files-you-already-have)

```sh
docker run -d --name strata-gear-check --gpus all -p 127.0.0.1:8080:8080 \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-gear-check-data:/data \
  --mount "type=bind,src=${GGUF_DIR:?},dst=/models,readonly" \
  --mount "type=bind,src=${KEY_FILE:?},dst=/run/secrets/strata,readonly" \
  -e FAMILY=qwen -e MODEL=IQ3_XXS -e MODEL_PATH=/models -e CONTEXT=32k \
  -e API_KEY_FILE=/run/secrets/strata ghcr.io/niko1221/strata:0.2.7-cuda13
docker inspect --format '{{.State.Health.Status}}' strata-gear-check
docker image inspect --format '{{json .RepoDigests}}' ghcr.io/niko1221/strata:0.2.7-cuda13
```

公式コンテナ内は全インターフェースで待ち受け、ホスト側の公開を loopback に絞る。Gear の接続先は `127.0.0.1`。初回は `starting` から `healthy` になるまでログとロード状態を確認し、次に実 API を診断する。Docker Engine 28.0.0 未満には localhost 公開ポートへ同一 L2 から到達できる既知の制約がある。[Docker のポート公開](https://docs.docker.com/engine/network/port-publishing/)

### SSH トンネル（直接起動・Docker 共通）

SSH 先のホストで上のいずれかの手順を済ませる。Gear 側の別ターミナルでホスト鍵の指紋を確認し、通常の SSH 認証で転送を維持する。SSH 鍵と Strata の Bearer は別々に管理する。

```sh
ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  user@remote-host
```

右側の `127.0.0.1:8080` は **リモートホスト**から見た宛先。Docker の場合はリモートホストの loopback に公開したポートに届く。開放するのは既存の SSH 経路だけでよい。`ExitOnForwardFailure` は転送の設定失敗を検出する。転送先がロード中・停止中でも転送自体は成立し得るため、次の診断をクライアントから実行する。[OpenSSH のローカル転送](https://man.openbsd.org/ssh.1#L)

## 各構成で JSON と SSE を確認する

以下は Gear リポジトリのルートで実行する Bash の例。`route` に表のファイル名の幹を指定し、配置ごとに新しい保存先を作る。生成物には会話・平文 reasoning・ツール結果が含まれる。Git が無視する非公開ディレクトリ内で扱う。

```bash
umask 077
route=local
mkdir -p test_outputs
case_dir=$(mktemp -d "test_outputs/strata-${route}.XXXXXX")
cfg="$case_dir/sse.toml"
cp "examples/strata/$route.toml" "$cfg"
uv run gear --config "$cfg" model check --profile strata --json > "$case_dir/metadata.json"
```

1. **実ロード状態を照合する。** `metadata.json` の `server.load_state`、`reported_model`、`model_ids`、`context_window_tokens` を確認する。矛盾や未報告なら推論ホストの monitor / 起動設定とも照合し、実値が分かるまで未検証とする。Strata は要求の model 名で自動ロード・切り替えをしない。コピーした TOML の `model` をロード済み ID、`context_window_tokens` を実ロード容量に合わせる。例の 32768 は全モデル共通ではない。
2. **予算と形式を選ぶ。** 例では生成と reasoning の合計上限が 512、予約が 2048、入力予算は window − 予約。HTTP timeout は 120 秒、SSE のバイト受信 idle は 60 秒、effort は `low`。実機に合わせて決め、両形式で揃える。`reasoning_replay = "none"` では opaque state を要求・再送しない。平文 reasoning は JSONL と再送履歴に残り、effort は別に適用される。[Strata の Responses 仕様](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/DETAILS.md#the-responses-api-and-codex-cli)
3. **設定を確定してから** `sed 's/^stream = true$/stream = false/' "$cfg" > "$case_dir/json.toml"` で JSON 用を作る。以下は `mode=sse` と `mode=json` で一度ずつ実行する。推論は明示実行で計算資源を消費する。

```bash
mode=sse
cfg="$case_dir/$mode.toml"
uv run gear --config "$cfg" model check --profile strata --json > "$case_dir/$mode-metadata.json"
uv run gear --config "$cfg" model check --profile strata --infer --deadline-seconds 120 --json \
  > "$case_dir/$mode-infer.json"
STRATA_API_KEY=deliberately-invalid-probe-key uv run gear --config "$cfg" \
  model check --infer --deadline-seconds 120 --json > "$case_dir/$mode-denied.json"
```

正しいキーの POST は 2xx かつ `inference.status = "succeeded"` と短い通常回答で合格（診断は空の回答やツール呼び出しを拒否する）。不正キーのコマンドは**終了コード 1** と `endpoint.http_status` 401/403、`authentication = "denied"` を確認する。接続失敗や公開 `/health` の成功は認証の証拠に使えない。小規模推論の cap は最大 32。上限に達したら `response_incomplete` を失敗として記録する。[Gear のモデル接続診断](../../README.md#モデル接続を診断する)

### クライアントのファイル読み取りと再開

```bash
uv run python -m examples.strata_probe --live --config "$cfg" --deployment "$route" \
  --output "$case_dir/$mode-probe"
```

ランナーは隔離 workspace にランダムな値だけの `marker.txt` を作り、値を教えずに読み取りを依頼する。成功には Gear の `file_read` 結果と、その値だけを返す最終回答が必要。同じ設定で runtime と JSONL store を開き直し、ツールを再実行せずに値を答えられること、元の履歴が保存されて 1 ターンだけ追記されたことを確認する。回帰テストでは送信した `function_call` と `function_call_output` の `call_id`、内容、再開時の件数も照合する。

`result.json` の `stage`、`status`、`error`、`metrics.model.requests` を読む。usage はサーバーが報告した値で、未報告は null。推測回答・ツール失敗・応答途絶は合格にならない。未完了や不明のターンは `sessions/probe.jsonl` を確認し、自動で再実行しない。新しい検証には新規ディレクトリを指定する。

### 長い入力、heartbeat、出力上限、切断

同じ `cfg` と経路で、合成入力だけを使う。下の上限は 512 語で、token 数ではない。実機の入力量・時間の上限を先に決め、Gear の保守的な入力見積もりが予算に入る大きさで行う。256K は不要。独立した空の Git workspace を作るのは、headless 証跡に親リポジトリの差分を含めないため。

```bash
work="$case_dir/$mode-work"
mkdir "$work"
git -C "$work" init -q
uv run python -c 'print("sample " * 512 + "\nReply with just OK. Do not use tools.")' > "$case_dir/$mode-long.txt"
uv run gear --config "$cfg" --workdir "$work" --session-dir "$case_dir/$mode-sessions" \
  run --prompt-file "$case_dir/$mode-long.txt" --run-dir "$case_dir/$mode-long"
```

JSON では通常の完了と usage、SSE では加えて prefill 中の転送を調べる。Gear が完了しただけでは heartbeat の受信時刻は分からない。SSE の追加試験では同じ loopback 経路に次のリクエストを 1 回送り、クライアントでイベント名と到着時刻だけを記録する。これは転送の観測用で、Gear の長文試験とは別に採点する。

```bash
sse_cfg="$case_dir/sse.toml"
uv run python - "$sse_cfg" "$case_dir/sse-long.txt" > "$case_dir/heartbeat-request.json" <<'PY'
import json, sys, tomllib
from pathlib import Path
model = tomllib.loads(Path(sys.argv[1]).read_text())["model"]
print(json.dumps({"model": model["model"], "stream": True, "input": Path(sys.argv[2]).read_text(),
                  "reasoning": {"effort": model["reasoning_effort"]}, "max_output_tokens": model["max_output_tokens"]}))
PY
url=$(uv run python -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["model"]["url"])' "$sse_cfg")
set -o pipefail
printf 'Authorization: Bearer %s\n' "$STRATA_API_KEY" | \
  curl -fsS -N --max-time 120 -H @- -H 'Content-Type: application/json' \
  --data-binary "@$case_dir/heartbeat-request.json" "$url" | \
  uv run python -u -c 'import sys,time
for line in sys.stdin:
    if line.startswith("event: "):
        print(time.monotonic(), line.strip())' > "$case_dir/heartbeat-events.txt"
```

開始直後の `response.in_progress` に加えて、長い prefill 中に約 15 秒間隔で後続の `response.in_progress` が届き、最後に `response.completed` が届くことを確認する。短時間で prefill が終わった場合は heartbeat を未検証とする。全イベントが末尾で一度に届く場合はプロキシのバッファリングを調べる。

* **cap** は形式ごとに設定コピーの `max_output_tokens` を 1 にし、長い回答を要求する。期待値は終了コード 3、`response_incomplete` と成功回答なし。サーバーが cap 到達前に正常終了したら上限到達は未検証。推論・出力を合わせた上限を記録する。

```bash
sed -E 's/^max_output_tokens = [0-9]+$/max_output_tokens = 1/' "$cfg" > "$case_dir/$mode-cap.toml"
uv run gear --config "$case_dir/$mode-cap.toml" --workdir "$work" --session-dir "$case_dir/$mode-sessions" \
  run --prompt "List the integers from 1 to 100 in full." --run-dir "$case_dir/$mode-cap"
```

* **切断** は専用の試験用接続に限る。新しい `--output` で probe を起動し、`sessions/probe.jsonl` に `tool_result` が出た後、次の応答完了前に自分の試験用 SSH トンネルを終了する。直接接続では管理者と調整した試験用サーバー停止や中継の切断を使う。失敗 stage、JSONL の `turn_error`、応答途中の EOF・ネットワーク失敗・idle timeout 等の実際のエラー、ツール実行件数を記録する。切断が間に合わなければ未検証。Gear はツールを含むターン全体をリトライしない。共有サーバーの再起動・model unload はこの手順の自動処理に含まれない。

### 代表構成での summary / Jev 0.3

1 構成を選び、実ロード context に要約用の余裕があることを確認する。Strata と Jev の両方を使える場合は既存の比較ランナーで同じ履歴を両戦略に渡す。実行時の `TYPESAFE_API_KEY` と利用許可が必要。生成した `results.json` の summary・Jev の完了、実効 threshold 0.3、各段階の usage を記録する。[比較ランナーの手順と判定範囲](../README.md)

```bash
uv run python -m examples.compaction_comparison --generate --config "$cfg" \
  --output "$case_dir/compaction" --jev-model jev-1.13.0 --jev-timeout 10
```

Jev を利用できない場合はその段階を未実施と記録する。summary 単独は同じ設定で `uv run gear --config "$cfg" --workdir "$work" --session-dir "$case_dir/summary-sessions"` を起動し、短いターンの後に `/compact` を実行する。JSONL の `compaction_summary` と再開後の回答を確認する。継続品質の評価は別途扱う。[Issue #29](https://github.com/TsuyoshiFujii-0915/gear-agent/issues/29)

## 実機検証表

今回の作業環境は Apple Silicon Mac。Strata の起動・ロード済み Qwen・指定ポートの待受けを確認できず、対応 Linux GPU コンテナと検証用 SSH 接続先の提供もない。4 構成の実機試験と、Strata を使う代表構成の summary / Jev 0.3 は未実施。通常の pytest は疑似 HTTP サーバーで JSON/SSE を検証する。

| 配置 | JSON | SSE | 実測に必要なもの |
| --- | --- | --- | --- |
| 同一ホスト直接 | 未検証 | 未検証 | ネイティブ engine、ロード済み Qwen、Bearer |
| 同一ホスト Docker | 未検証 | 未検証 | 対応 GPU ホスト、固定 image、モデル volume |
| SSH 先直接 | 未検証 | 未検証 | SSH 接続先、転送許可、起動済み Strata |
| SSH 先 Docker | 未検証 | 未検証 | SSH 接続先、対応 GPU コンテナ、loopback 公開 |

追試時は構成・形式ごとに、次の項目を `case_dir` に記録する。各段階を **実測成功 / 実測失敗（エラー種別）/ 未検証（不足条件）** のいずれかにし、実施した段階だけ更新する。

| 記録する内容 | 取得元 |
| --- | --- |
| Gear commit・dirty、配置ラベル、adapter、JSON/SSE、予算・予約、effort・cap、timeout | probe の `result.json`。probe 前に止まった場合は `git rev-parse HEAD`、`git status --porcelain` の有無と診断の `settings` |
| Strata tag・実 commit・engine version、Docker の実 image digest、OS/GPU、モデル ID・量子化・実ロード context | 実行中の版・起動記録・モデル pack、公式リリースの対応 commit（ソースビルドは checkout の HEAD）、image inspect、`metadata.json`。取得できない値は未確認 |
| 合否、所要秒数、provider 報告 usage | 正常/不正キー、短文、tool、resume、長文、heartbeat（SSE のみ）、cap、切断、代表構成の summary/Jev を個別に。診断 JSON、probe の `metrics`、headless の `metrics.json` と上記の観測を対応付ける |

公開するのは必要な集計と不足条件だけにする。認証付き URL、Bearer、SSH 接続情報、生成した設定・JSONL・モデルの本文はリポジトリや PR に添付しない。
