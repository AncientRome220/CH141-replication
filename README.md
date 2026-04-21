# 「ルールベース手法とLLMによる古文書からの経済情報抽出の比較」再現パッケージ

## 論文情報

- **著者**: 矢野 駿介
- **学会**: 情報処理学会 人文科学とコンピュータ研究会（CH141）
- **年**: 2025
- **DOI**: （公開後に追記）

## 概要

本リポジトリは、古代エジプトのギリシア語パピルス文書（DDbDP EpiDoc XMLコーパス）から穀物価格情報を自動抽出するパイプラインの再現パッケージです。ルールベース手法（正規表現 + ヒューリスティクス）とLLM手法（Gemini 2.0 Flash）の2つのアプローチを比較評価し、それぞれの精度・再現率・F1スコアを報告します。評価にはゴールドスタンダードアノテーションおよびHarper (2016) の既存データセットとの再現率比較を用います。

## リポジトリ構成

```
CH141-replication/
├── README.md                         本ファイル
├── requirements.txt                  Python依存パッケージ
├── .env.example                      環境変数テンプレート（LLM抽出用）
├── .gitignore
├── LICENSE                           MITライセンス
├── src/                              パイプラインスクリプト
│   ├── pipeline_shared.py            共有定数・正規表現・正規化関数
│   ├── 1_harvest_candidates.py       第1段階：候補文書選定
│   ├── 2_extract_prices.py           第2段階A：ルールベース抽出
│   ├── 2b_llm_extract_prices.py      第2段階B：LLM抽出
│   ├── 3_clean_and_analyze.py        第3段階：クリーニング・分析
│   ├── 4_plot_robust_trends.py       第4段階：可視化
│   ├── 5_sample_for_annotation.py    評価セット層別抽出
│   ├── 6_build_gold_standard.py      ゴールドスタンダード構築
│   └── 7_harper_recall.py            Harper再現率評価
├── data/                             事前計算済みデータ
│   ├── candidate_documents.csv       候補文書一覧（4,042件）
│   ├── extracted_price_mentions_v12.csv   ルールベース抽出結果
│   ├── rejected_mentions_v12.csv     ルールベース除外メンション
│   ├── extracted_price_mentions_llm.csv   LLM抽出結果
│   ├── gold_standard_annotation.csv  ゴールドスタンダードアノテーション
│   ├── harper_2016_wheat_prices.xlsx Harper (2016) 小麦価格データ
│   └── harper_recall_comparison.csv  Harper再現率比較結果
└── examples/                         サンプルXML（小規模テスト用）
    ├── DDB_EpiDoc_XML/               DDbDPサンプル（14件）
    └── HGV_meta_EpiDoc/              対応するHGVメタデータ
```

## 動作環境

- Python 3.10 以上
- 依存パッケージは `requirements.txt` を参照

## 外部データ

パイプラインの完全な再実行には、以下の外部XMLコーパスが必要です。

- **DDbDP EpiDoc XML**: Duke Databank of Documentary Papyri
- **HGV メタデータ**: Heidelberger Gesamtverzeichnis der griechischen Papyrusurkunden Ägyptens

いずれも [papyri/idp.data](https://github.com/papyri/idp.data) から取得できます。

```bash
git clone https://github.com/papyri/idp.data.git
```

クローン後、`DDB_EpiDoc_XML/` と `HGV_meta_EpiDoc/` ディレクトリへのパスをスクリプト内の定数で指定してください。

> **注**: `examples/` ディレクトリに14件のサンプルXMLを同梱しています。外部データをダウンロードせずに、パイプラインの動作確認を小規模に実施できます。

## インストール

```bash
pip install -r requirements.txt
```

LLM抽出（第2段階B）を再実行する場合は、`.env.example` を `.env` にコピーし、Gemini APIキーを設定してください。

```bash
cp .env.example .env
# .env を編集して GEMINI_API_KEY を設定
```

## 結果の再現方法

`data/` ディレクトリに事前計算済みの結果が含まれています。スクリプトを再実行しなくても、論文中の数値を確認できます。

完全な再実行を行う場合は、以下の順序でスクリプトを実行してください。

| スクリプト | 論文との対応 | 入力 | 出力 |
|-----------|-------------|------|------|
| `1_harvest_candidates.py` | 第3節 候補選定 | DDbDP XML + HGV XML | `candidate_documents.csv` |
| `2_extract_prices.py` | 第3節 ルールベース抽出 | `candidate_documents.csv` + DDbDP XML | `extracted_price_mentions_v12.csv`, `rejected_mentions_v12.csv` |
| `2b_llm_extract_prices.py` | 第3節 LLM抽出 | `candidate_documents.csv` + DDbDP XML | `extracted_price_mentions_llm.csv` |
| `3_clean_and_analyze.py` | 補足分析 | 抽出結果CSV | 統計サマリー |
| `4_plot_robust_trends.py` | 補足可視化 | 抽出結果CSV | 図（`outputs/`） |
| `5_sample_for_annotation.py` | 第4節 評価セット | 抽出結果CSV | サンプリング済みリスト |
| `6_build_gold_standard.py` | 第4節 ゴールドスタンダード | アノテーション結果 | `gold_standard_annotation.csv` |
| `7_harper_recall.py` | 第4.4節 Harper再現率 | 抽出結果 + Harper Excel | `harper_recall_comparison.csv` |

## データ辞書

### `candidate_documents.csv`

候補文書のメタデータとスコアリング結果。

| カラム | 説明 |
|--------|------|
| `DDB_ID` | DDbDP文書識別子（例: `bgu;1;14`） |
| `XML_RelPath` | XMLファイルの相対パス |
| `Title` | 文書タイトル |
| `Place` | 出土地 |
| `Date_Text` | 日付テキスト（HGVメタデータ） |
| `Date_NotBefore` / `Date_NotAfter` | 年代範囲 |
| `Score` | 候補選定スコア |
| `Grain_Hits` / `Unit_Hits` / `Money_Hits` | 穀物・単位・貨幣の出現数 |

### `extracted_price_mentions_v12.csv`

ルールベース手法による抽出メンション。

| カラム | 説明 |
|--------|------|
| `DDB_ID` | 文書識別子 |
| `Mention_ID` | メンション識別子 |
| `Grain_Form` | 穀物の表記形 |
| `Qty_Value` / `Qty_Unit` | 数量値・単位 |
| `Price_Value` / `Price_Cur` | 価格値・通貨 |
| `Score` | 信頼度スコア |
| `Context_Type` | 文脈タイプ（price, tax, rent等） |
| `Signal_Strength` | シグナル強度 |
| `Rejection_Reason` | 除外理由（該当する場合） |

### `extracted_price_mentions_llm.csv`

LLM手法（Gemini 2.0 Flash）による抽出メンション。

| カラム | 説明 |
|--------|------|
| `DDB_ID` | 文書識別子 |
| `Mention_ID` | メンション識別子 |
| `Grain_Form` | 穀物の表記形 |
| `Qty_Value` / `Qty_Unit` | 数量値・単位 |
| `Price_Value` / `Price_Cur` | 価格値・通貨 |
| `Is_Price` | 価格言及かどうか（LLM判定） |
| `Confidence` | LLM信頼度 |
| `Transaction_Type` | 取引タイプ |
| `Reasoning` | LLMの判断根拠 |

### `gold_standard_annotation.csv`

人手アノテーションによるゴールドスタンダード。

| カラム | 説明 |
|--------|------|
| `Sample_ID` | サンプル識別子 |
| `Sample_Stratum` | 層別抽出の層 |
| `Human_Label` | 人手ラベル（TP, FP, ME等） |
| `Human_Commodity` / `Human_Qty` / `Human_Price` | 人手による正解値 |
| `RB_*` | ルールベース手法の出力値 |
| `LLM_*` | LLM手法の出力値 |

### `harper_recall_comparison.csv`

Harper (2016) の小麦価格データとの再現率比較。

| カラム | 説明 |
|--------|------|
| `Harper_Source` | Harper表の文書名 |
| `Period` | 時代区分 |
| `DDB_IDs_Matched` | マッチしたDDB ID |
| `In_Candidates` | 候補に含まれるか |
| `In_RuleBased` | ルールベースで抽出されたか |
| `In_LLM` | LLMで抽出されたか |

## 引用

```bibtex
@inproceedings{yano2025grain,
  author    = {矢野 駿介},
  title     = {ルールベース手法と{LLM}による古文書からの経済情報抽出の比較},
  booktitle = {情報処理学会研究報告人文科学とコンピュータ（CH）},
  volume    = {2025-CH-141},
  year      = {2025},
  note      = {DOI: 公開後に追記}
}
```

## ライセンス

本リポジトリのソースコードは [MIT License](LICENSE) の下で公開しています。

`data/harper_2016_wheat_prices.xlsx` は Harper (2016) に基づくデータです。`examples/` 内のXMLファイルは [papyri/idp.data](https://github.com/papyri/idp.data) からの抜粋であり、元データのライセンスに従います。
