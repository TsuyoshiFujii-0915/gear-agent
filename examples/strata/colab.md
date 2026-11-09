# Google Colab で Strata を利用するための成立条件

**2026-11-09 時点の判定：managed Colab での実験は外部接続の前に停止した。** 想定したのは、利用者が Notebook を開いている間だけ Strata を動かし、手元の Gear から推論する構成。外部クライアントと中継経路をこの用途で利用できる根拠を確認できず、対象ランタイムの割当情報もない。モデル取得、外部公開、実機推論はいずれも未実施。[Issue #40](https://github.com/TsuyoshiFujii-0915/gear-agent/issues/40)

## 環境別の判断

| 環境 | 公式資料で確認したこと | 今回の判断 |
| --- | --- | --- |
| 個人向け managed Colab（無料・有料） | 全 managed runtime で、対話的計算と無関係なサービス提供や remote proxy への接続が制限される。無料利用や compute unit 残高がない場合は、SSH 等の遠隔操作にも追加制限がある。 | 有料プランと正の残高だけでは、手元の Gear に API を提供する経路の適合性は確認できない。対象用途と具体的な経路について Google の根拠を得るまで外部公開用 Notebook・トンネル手順の提供を見送る。[Colab FAQ「Which activities are restricted in Colab?」](https://research.google.com/colaboratory/faq.html#disallowed-activities) |
| Colab Enterprise / 専用 VM | Enterprise は Google Cloud の IAM・ネットワーク設定を使い、runtime template で GPU とディスクを選べる。専用 VM も選択肢にある。 | 管理者がこの用途と接続経路を許可し、対応 GPU を割り当てた環境が候補。今回、該当するプロジェクト・VM は未提供。[Enterprise のランタイム](https://docs.cloud.google.com/colab/docs/introduction)・[Colab FAQ の資源制限](https://research.google.com/colaboratory/faq.html#resource-limits) |
| Colab local runtime | Notebook のコードは利用者が管理するホストで実行される。 | 対応ホストを所有する場合は同一ホストで Gear を使える。専用 VM なら許可された SSH 経路も選べる。結果は各配置の欄に記録する。[Local runtimes](https://research.google.com/colaboratory/local-runtimes.html)・[Gear の local / SSH 検証](README.md) |

Cloudflare Quick Tunnel は公式に SSE 非対応で、起動のたびにホスト名が変わる。JSON が届いた場合でも、その結果が示すのは非ストリーム経路の到達性まで。今回は JSON も未測定。別のトンネルを選ぶ場合も、そのサービス・プランの SSE 仕様と Colab 側の利用条件を個別に確認する。[Quick Tunnels の制限](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations)

## 資源と検証記録

| 項目 | 候補・判定基準 | 今回の結果 |
| --- | --- | --- |
| Gear / Strata / モデル | Gear 0.1.0（基準 commit `dfbb1b6`）、Strata v0.2.7（`5e8a4cca`）、ISTA-DASLab Qwen3.8-Flash-Next IQ3_XXS | Gear のローカル試験のみ。engine 版・実ロード ID は未確認 |
| 割当資源と保存先 | GPU 型・総/空き VRAM・driver/CUDA は割当ホストの `nvidia-smi` で確認。Linux のホスト RAM は `free -h`、空きディスクはモデル保存先を指定した `df -h` で測定 | 作業端末は macOS arm64。対象ランタイムの全項目は未測定、保存先未選択 |
| 推論設定 | 候補は context 32768、予約 2048、effort `low`、cap 512、JSON と SSE を各 1 構成。ロード後に実 context と照合 | 未適用 |
| HTTPS・認証・推論 | キーなしの推論 POST が 401/403、正しい Bearer が完了。JSON と SSE は個別に判定 | 両形式とも未実施。許可を確認できた経路なし |
| tool 往復・reasoning 再送・同一モデル再開・切断 | クライアントの隔離ファイルを読み、結果を推論先に返して最終回答まで確認 | Colab 実機ではすべて未実施 |
| ローカル fixture | 401、HTTP 200 の HTML、通信切断、terminal 前の EOF、incomplete を失敗として診断。JSON/SSE の tool 往復と再開 | 成功。再現コマンドは下記 |

IQ3_XXS の公開値はモデル取得 **76 GB**、RAM は experts 約 43 GB と追加約 6 GB。GGUF から準備する場合は pack 約 40 GB と draft layer 約 6 GB も考慮する。利用条件を確認できたホストでは、GPU・driver が v0.2.7 の対応表に載るか確認し、公式のネイティブ Terminal の `strata launch --plan --family qwen --model iq3_xxs --context 32k` で不足ファイルと保存先を照合する。Docker の GPU 利用や Drive のマウント・読み出し性能は前提に含めない。[Strata の導入要件](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/INSTALL.md)・[モデル別の取得量と RAM](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/DETAILS.md#which-model)・[pack 準備時の追加容量](https://github.com/Niko1221/Strata/blob/v0.2.7/docs/DOCKER.md#files-you-already-have)

GPU の割当・モデル準備・推論には compute unit または VM の利用料がかかり得る。今回の消費率は未測定。利用者が対象環境の残高・時間単価（CU/時または VM 料金）と予定時間、ストレージ料金を確認し、取得・購入・再ロード・外部接続をそれぞれ明示的に実行する。

## 許容された実機で確認する項目

1. Strata を loopback に限定し、ローカルでロード状態・実モデル ID・context を確認する。外部経路は認証付き HTTPS とし、Gear の `model.url` には `https://<endpoint>/v1/responses` を prefix ごと指定する。Strata の Bearer と中継サービスの token は別々の Secrets に格納する。TLS 検証、Host/Origin 保護を維持する。Gear のサーバー間 HTTP に CORS 全許可は不要。
2. `gear model check --profile strata` で metadata を照合する。推論 POST の認証確認には、試験用設定の `api_key_env = ""` で `model check --infer` を実行し、401/403 を確認する。次に `api_key_env = "STRATA_API_KEY"` で `model check --infer` を実行する。ログイン画面や別の認証ヘッダーを要求するゲートウェイは現行 Bearer での接続失敗として記録する。
3. [共通の JSON/SSE・tool 往復・再開試験](README.md#各構成で-json-と-sse-を確認する)と同じ基準を使う。SSE はクライアントで長い prefill 中の `response.in_progress`、`response.output_text.delta`、`response.completed` の到着時刻を記録し、一括配送なら不合格、prefill が短ければ heartbeat は未検証とする。heartbeat は処理中のモデル応答に限り、Notebook の idle 回避には使用しない。
4. ランタイム停止・中継切断・URL 変更は個別に記録する。未完了出力を最終回答として扱わず、失敗種別と実行済み tool の履歴をクライアントの原 JSONL に残す。ターン全体を自動再実行しない。新 URL では接続先・実モデル ID を再確認し、opaque reasoning の scope 判定を維持する。別モデルでは新規セッションを使う。
5. 終了時は利用者が開いた中継を閉じ、専用の Strata を `strata launch --stop` で停止する。不要なランタイム・VM を終了して課金状態を確認する。原 JSONL やモデル・tool の本文は非公開の Git 管理外ディレクトリに置き、公開記録は版・設定・集計・失敗種別に限る。キーや Notebook の実行出力は保存しない。

Gear の共通診断・tool 往復は次のコマンドで再検証できる。fixture はローカル HTTP サーバーを使い、GPU・モデルダウンロード・外部サービスは不要。

```sh
uv run --locked pytest -q tests/test_model_check.py tests/test_stream_transport.py \
  tests/test_strata_compatibility.py tests/test_strata_probe.py
```
