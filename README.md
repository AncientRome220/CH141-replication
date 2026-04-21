# ルールベース手法とLLMによる古文書からの経済情報抽出の比較 ――ローマ帝政期エジプトの穀物価格を事例に―― 再現パッケージ

## 論文情報

- **タイトル**: ルールベース手法とLLMによる古文書からの経済情報抽出の比較 ――ローマ帝政期エジプトの穀物価格を事例に――
- **英題**: Comparing Rule-Based and LLM Methods for Extracting Economic Information from Ancient Documents: Grain Prices in Roman Imperial Egypt
- **著者**: 藤本 俊哉（京都大学）・阿部 忠道（千葉大学）・小川 潤（東京大学）
- **学会**: 情報処理学会 人文科学とコンピュータ研究会 第141回研究発表会（CH141）
- **年**: 2025
- **DOI**: （公開後に追記）

## 概要

本研究は、ローマ帝政期エジプト（概ね前30年〜後300年頃）のギリシア語パピルスから穀物価格の情報を機械的に抽出する手法を提案する。元データには、パピルスのオンラインデータベースPapyri.infoが公開するDDbDP（本文）およびHGV（メタデータ）のXMLファイルを用いる。

本研究では2つの手法による抽出を比較する。第一に、Pythonによるルールベースのテクスト処理により、穀物名・数量・通貨の共起パターンから価格の記述を同定する。否定文脈（税・運搬費・配給・種子貸与等）を事前に除外する文脈フィルタを備える。第二に、大規模言語モデル（Gemini 2.5 Flash）に指示を与え、同様の抽出を行わせる。

145件の人手評価（境界事例を除く）では、LLM手法の広義の適合率は62.2%（ルールベース39.3%）、Harper（2016）に対する文書再現率は60.3%（同42.6%）であった。ただし、人手データセット上の再現率ではルールベースが優位（78.6% vs 66.7%）であり、両手法は相補的である。

## 抽出パイプライン

パイプラインは3段階からなる。

1. **第1段階（候補選定）**: 全DDbDP文書（約82,000件）を走査し、穀物語彙・通貨語彙・数量単位語彙の共起および`<num>`タグの存在を正規表現で検査して候補文書を選定する。4,042件を選定。
2. **第2段階A（ルールベース抽出）**: 候補文書のXMLを解析し、文脈フィルタによる否定文脈の除外を行ったうえで、穀物語と近傍の数量・金額表現をペアリング・スコアリングする。550件の候補から419件を除外し、131件を保持。
3. **第2段階B（LLM抽出）**: 候補文書から穀物語の前後約100文字のテキストウィンドウ（13,479個）を生成し、Gemini 2.5 Flashに構造化JSON形式で抽出結果を返させる。1,263件を穀物価格と判定。

## 主要な結果

| 指標 | ルールベース | LLM |
|------|-------------|-----|
| 広義の適合率（TP+ME / 陽性判定数） | 39.3% | 62.2% |
| 人手データセット再現率 | 78.6% | 66.7% |
| Harper再現率（文書単位） | 42.6% | 60.3% |

文脈フィルタにより除外された419件の理由分布: 税30.1%、種子22.2%、運搬18.9%、貸付6.2%、配給6.2%、行政3.6%、複合12.9%。除外された件はすべてFP（偽陽性）であり、真の価格記述の誤除外は確認されなかった。

## リポジトリ構成

```
CH141-replication/
├── README.md                         本ファイル
├── requirements.txt                  Python依存パッケージ
├── .env.example                      環境変数テンプレート（LLM抽出用）
├── .gitignore
├── LICENSE                           MITライセンス
├── src/                              パイプラインスクリプト
│   ├── pipeline_shared.py            共有定数・正規表現・正規化関数（Table 2の換算値）
│   ├── 1_harvest_candidates.py       第1段階：候補文書選定（第3節）
│   ├── 2_extract_prices.py           第2段階A：ルールベース抽出（第3節）
│   ├── 2b_llm_extract_prices.py      第2段階B：LLM抽出（第3節）
│   ├── 3_clean_and_analyze.py        第3段階：クリーニング・分析
│   ├── 4_plot_robust_trends.py       第4段階：可視化
│   ├── 5_sample_for_annotation.py    評価セット層別抽出（第3.6節）
│   ├── 6_build_gold_standard.py      ゴールドスタンダード構築（第3.6節）
│   └── 7_harper_recall.py            Harper再現率評価（第4.4節）
├── data/                             事前計算済みデータ
│   ├── candidate_documents.csv       候補文書一覧（4,042件）
│   ├── extracted_price_mentions_v12.csv   ルールベース抽出結果（550件、うち131件保持）
│   ├── rejected_mentions_v12.csv     文脈フィルタによる除外メンション（419件）
│   ├── extracted_price_mentions_llm.csv   LLM抽出結果（13,479ウィンドウ、1,263件価格判定）
│   ├── gold_standard_annotation.csv  人手アノテーション（150件: TP=25, FP=103, ME=17, BORDER=5）
│   ├── harper_2016_wheat_prices.xlsx Harper (2016) 小麦価格データ（68件）
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

- **DDbDP EpiDoc XML**: Duke Databank of Documentary Papyri（パピルス本文、約82,000ファイル）
- **HGV メタデータ**: Heidelberger Gesamtverzeichnis der griechischen Papyrusurkunden Ägyptens（日付・出土地・タイトル等）

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
| `2_extract_prices.py` | 第3.4節 ルールベース抽出 | `candidate_documents.csv` + DDbDP XML | `extracted_price_mentions_v12.csv`, `rejected_mentions_v12.csv` |
| `2b_llm_extract_prices.py` | 第3.5節 LLM抽出 | `candidate_documents.csv` + DDbDP XML | `extracted_price_mentions_llm.csv` |
| `3_clean_and_analyze.py` | 補足分析 | 抽出結果CSV | 統計サマリー |
| `4_plot_robust_trends.py` | 補足可視化 | 抽出結果CSV | 図（`outputs/`） |
| `5_sample_for_annotation.py` | 第3.6節 評価セット | 抽出結果CSV | サンプリング済みリスト |
| `6_build_gold_standard.py` | 第3.6節 ゴールドスタンダード | アノテーション結果 | `gold_standard_annotation.csv` |
| `7_harper_recall.py` | 第4.4節 Harper再現率 | 抽出結果 + Harper Excel | `harper_recall_comparison.csv` |

> **注意**: 第2段階BのLLM抽出はGemini APIを使用するため、APIキーと実行費用が必要です。`data/extracted_price_mentions_llm.csv` に事前計算済みの結果を同梱しているため、APIなしでも評価結果の確認が可能です。

## データ辞書

### `candidate_documents.csv`

第1段階で選定された候補文書のメタデータとスコアリング結果（4,042件）。

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

ルールベース手法による抽出メンション（文脈フィルタ通過後の候補を含む）。

| カラム | 説明 |
|--------|------|
| `DDB_ID` | 文書識別子 |
| `Mention_ID` | メンション識別子 |
| `Grain_Form` | 穀物の表記形（7語幹: sit-, pyr-, krith-, zea-, olyr-, stachy-, aleur-） |
| `Qty_Value` / `Qty_Unit` | 数量値・単位（リットルに換算可能） |
| `Price_Value` / `Price_Cur` | 価格値・通貨（ドラクマに換算可能） |
| `Score` | 信頼度スコア（閾値25以上で保持） |
| `Context_Type` | 文脈分類（price / other） |
| `Signal_Type` | 取引シグナル種別（time, sale_verb, rate_construction等） |
| `Signal_Strength` | シグナル強度（0〜1） |
| `Neg_Signals` | 検出された否定シグナル |
| `Rejection_Reason` | 除外理由（該当する場合） |

### `rejected_mentions_v12.csv`

文脈フィルタにより除外されたメンション（419件）。

| カラム | 説明 |
|--------|------|
| `DDB_ID` | 文書識別子 |
| `Rejection_Reason` | 除外理由（TAX, SEED, TRANSPORT, LOAN, RATION, ADMIN, 複合） |
| `Context_Window` | 穀物語の前後テキスト |

### `extracted_price_mentions_llm.csv`

LLM手法（Gemini 2.5 Flash）による抽出メンション（13,479ウィンドウ）。

| カラム | 説明 |
|--------|------|
| `DDB_ID` | 文書識別子 |
| `Mention_ID` | メンション識別子 |
| `Grain_Form` | 穀物の表記形 |
| `Qty_Value` / `Qty_Unit` | 数量値・単位（LLM出力） |
| `Price_Value` / `Price_Cur` | 価格値・通貨（LLM出力） |
| `Is_Price` | 穀物価格かどうか（LLM判定: true/false） |
| `Confidence` | LLM信頼度（high / medium / low） |
| `Transaction_Type` | 取引タイプ（sale, evaluation, rate等） |
| `Reasoning` | LLMの判断根拠（自然言語） |

### `gold_standard_annotation.csv`

150件のテキストウィンドウに対する人手アノテーション。層A（ルールベース抽出ありの文書）87件、層B（抽出なしの文書）63件。

| カラム | 説明 |
|--------|------|
| `Sample_ID` | サンプル識別子 |
| `Sample_Stratum` | 層別抽出の層（A / B） |
| `Human_Label` | 人手ラベル（TP: 真陽性, FP: 偽陽性, ME: 部分正解, BORDER: 境界事例） |
| `Human_Commodity` / `Human_Qty` / `Human_Price` / `Human_Currency` | 人手による正解値 |
| `RB_*` | ルールベース手法の出力値（Score, Context_Type等） |
| `LLM_*` | LLM手法の出力値（Is_Price, Confidence, Reasoning等） |

### `harper_recall_comparison.csv`

Harper (2016) の小麦価格データ68件との文書単位の再現率比較。

| カラム | 説明 |
|--------|------|
| `Harper_Source` | Harper表の文書名 |
| `Period` | 時代区分 |
| `Date_Earliest` / `Date_Latest` | 年代範囲 |
| `DDB_IDs_Matched` | マッチしたDDB ID |
| `In_Candidates` | 第1段階で候補に含まれるか（75.0%） |
| `In_RuleBased` | ルールベースで抽出されたか（42.6%） |
| `In_LLM` | LLMで抽出されたか（60.3%） |

## 引用

```bibtex
@inproceedings{fujimoto2025grain,
  author    = {藤本 俊哉 and 阿部 忠道 and 小川 潤},
  title     = {ルールベース手法と{LLM}による古文書からの経済情報抽出の比較――ローマ帝政期エジプトの穀物価格を事例に――},
  booktitle = {情報処理学会研究報告人文科学とコンピュータ（CH）},
  volume    = {2025-CH-141},
  year      = {2025},
  note      = {DOI: 公開後に追記}
}
```

## ライセンス

本リポジトリのソースコードは [MIT License](LICENSE) の下で公開しています。

`data/harper_2016_wheat_prices.xlsx` は Harper (2016) に基づくデータです。`examples/` 内のXMLファイルは [papyri/idp.data](https://github.com/papyri/idp.data) からの抜粋であり、元データのライセンスに従います。
