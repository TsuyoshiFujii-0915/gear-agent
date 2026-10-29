# treg の Web 検索・取得

`provider = "treg"` は TinyFish の直接 endpoint を使います。検索と本文取得は独立に
選択でき、片方だけを Tavily にしても動作します。主モデルの設定は変更しません。
モデルの引数は `web_search(query)` / `web_fetch(url)` のままです。

## 設定と認証

既存の `[tool]` の `web_search` / `web_fetch` を必要に応じて `true` にし、該当する
Web 設定を次の内容で置き換えてください。Tavily 固有のキーは残さないでください。

```toml
[web_search]
provider = "treg"
api_key_env = "TREG_TOKEN"
max_results = 5
timeout_seconds = 20

[web_search.treg]
org = "your-org-slug"

[web_fetch]
provider = "treg"
api_key_env = "TREG_TOKEN"
max_content_chars = 20000
timeout_seconds = 30

[web_fetch.treg]
org = "your-org-slug"
ttl_seconds = 0
per_url_timeout_ms = 20000
```

**TinyFish 個別の provider キーは通常不要ですが、treg token は必要です。** treg に
登録・ログインして取得した REST 用 token を `TREG_TOKEN` 環境変数に設定します。
identity token では利用組織を `org` に明示します。組織専用の per-org token を使う
場合は `org` の行を省略してください。`[web_search.treg]` テーブルは残します。
REST 認証は `X-Treg-Token`、組織指定は `X-Treg-Org` です。`Authorization: Bearer`
は送信しません。無効な機能については token を要求せず、通信もしません。

Gear は `~/.treg/config.json` を読みません。treg CLI の導入・ログイン、BYOK キーの
アップロード、残高購入・自動 top-up も行いません。ユーザー自身が公式 CLI で
`treg login` した場合、その設定ファイルに保存される token と組織情報の取り扱いは
ユーザーが管理します。Gear の Docker shell に CLI やブラウザは不要です。

## direct と routed、費用

- 検索は `GET https://treg.to/call/tinyfish.web.search` に `query` と
  `domain_type=web` だけを送ります。
- 取得は `POST https://treg.to/call/tinyfish.web.fetch` に単一要素の `urls`、
  `format="markdown"`、`ttl`、`per_url_timeout_ms` を送ります。
- `treg.web.search` / `treg.web.extract` は別 vendor への有料 fallback があり得る
  routed endpoint です。この実装では利用できません。provider 選択や有料利用を
  モデル引数から変更することもできません。

2026-10-29 に公開 catalog の両 direct endpoint について `free` / USD 0 を確認
しました（catalog 内の価格確認日は 2026-09-23）。これは永続的な無料保証では
ありません。毎回、認証情報を送らず公開 catalog を確認し、価格が不明・非無料に
なった場合や overflow が広告される場合は実呼び出し前に停止します。

実呼び出しには `X-Treg-Route-Max-Cost: 0` も送ります。確認した公式サーバー実装では
direct の予約課金にもこの上限が適用されます。この header を送っても routed
endpoint を選択することにはなりません。catalog 確認と呼び出しは不可分の処理では
なく、上限の強制は稼働中のサービスに依存します。応答に非ゼロ課金や異なる経路が
報告されたら明示的エラーにしますが、事後検知は返金や課金の取消ではありません。

`provider_metadata.treg` に設定された endpoint と、報告された `served_by`、
`charged_micro`、`call_id` を区別して保存します。金額が報告されなければ `null`
です。`micro_USD` は 100 万分の 1 USD の単位で、Tavily credits とは合算しません。
実 upstream が報告されなければ、それも `null` とし TinyFish と推測して埋めません。
組織が独自に登録した BYOK キーを treg 側が使う場合、vendor 側の請求は treg の
残高課金報告とは別です。

401・402・429・timeout・不正 JSON・HTTP 200 内のエラーも失敗として返し、
自動 retry、別 vendor への切替、API の redirect 追従はしません。

## 結果と制限

検索の `max_results` は受信した順位付き結果に対するローカル上限です。
API に未定義の `count` / `limit` は送らず、不足件数を補う自動ページングも行いません。
検索 0 件は正常です。`content` は snippet、`position` は順位として保持し、確率
score に変換しません。検索時の本文取得は無効で、必要なら `web_fetch` を別に呼びます。

取得は `results[].url` が要求 URL に一致することと `errors[]` を検査します。
`text` を本文にし、`requested_url` と報告された `final_url` を区別します。
別 URL の結果で代用せず、空本文・`max_content_chars` 超過はエラーにします。
HTTP 応答は 8 MiB までで、圧縮応答も拒否します。無断の切り詰め・要約はしません。

`ttl_seconds` はキャッシュ有効期間の秒数で、0 は fresh fetch の要求です。
正の値ならキャッシュされた内容を含み得ます。取得日時や最新性は捏造しません。
`per_url_timeout_ms` は TinyFish 側の URL 単位のミリ秒、`timeout_seconds` は
Gear の HTTP timeout の秒数です。いずれも必須で、TTL 以外は正の整数です。
`max_results` / `max_content_chars` も正の整数です。

公開 HTTP(S) URL が対象です。ページ認証、cookie、任意ブラウザ操作には対応しません。
クエリ・対象 URL は treg と TinyFish へ送信されます。本文は外部由来のツールデータ
として扱い、system/developer 指示に昇格させません。Web 通信はホスト側で行い、
Docker shell 用の `runtime.network` とは独立です。

## 検証状況

公開 catalog と公式 source の確認、およびモック HTTP による契約テストを実施しました。
利用者の token による認証付き検索・取得は未実行で、**認証疎通は未確認**です。
token 準備後は小さな公開 query と公開 URL 1 件で試し、結果の endpoint、
`served_by`、`charged_micro` を記録してください。未報告値は未報告のまま扱います。

参照: [検索 catalog](https://treg.to/catalog/endpoints/tinyfish.web.search)、
[取得 catalog](https://treg.to/catalog/endpoints/tinyfish.web.fetch)、
[公式利用方法](https://github.com/superdesigndev/treg/blob/main/USAGE.md)、
[ADR](decisions/0033-use-treg-tinyfish-direct-endpoints.md)。
