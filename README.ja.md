<h1 align="center">KeepCount</h1>

<p align="center">
  行動のデータから作った画像が何を保つかを段ごとに数える監査の、条件・コード・記録・git の履歴。<br>
  打鍵のタイルを作例とする。
</p>

<p align="center">
  <a href="LICENSE"><img alt="Code: MIT" src="https://img.shields.io/badge/code-MIT-2f6f9f"></a>
  <a href="LICENSE-docs"><img alt="Documents and records: CC BY-NC-SA 4.0" src="https://img.shields.io/badge/documents%20%26%20records-CC%20BY--NC--SA%204.0-d66e28"></a>
  <img alt="Python 3.13" src="https://img.shields.io/badge/python-3.13-4b8bbe">
  <a href="https://docs.astral.sh/uv/"><img alt="Environment: uv.lock" src="https://img.shields.io/badge/environment-uv.lock-5c5c5c"></a>
  <img alt="Tests: 384 passed (659 with the rebuilt records)" src="https://img.shields.io/badge/tests-384%20passed%20%7C%20659%20with%20records-3f8f4f">
  <a href="https://doi.org/10.5281/zenodo.23858735"><img alt="DOI 10.5281/zenodo.23858735" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23858735-1682d4"></a>
</p>

<p align="center"><a href="README.md">English</a> | <b>日本語</b></p>

このリポジトリは Mikoto Miura・Yuusuke Harada の 2 つの論文のアーカイブである。*KeepCount: A Stage-by-Stage Audit of What Images of Behavioral Data Keep, with Keystroke Tiles as a Worked Example*（査読前のプレプリント [10.5281/zenodo.23858734](https://doi.org/10.5281/zenodo.23858734)。run v1〜v6j）と、版 1.1.0 からの *Can a pause summary show where one writer paused? An audit procedure for keystroke logs*（*Journal of Writing Research* に投稿した原稿。run v6k〜v6s）である。後者は同じ問いを、書く過程の研究が報告する休止の要約に向ける（下の「休止の要約の論文」）。

KeepCount は監査の手順である。データから画像への変換を段に切り、段ごとに、数える前に決めた比べの相手に対して、(a) その段が人をどれだけ見分けるか、(b) データへの操作がその段に映るかを数える。陽性対照を置き、人自身の揺れより細かい物差しには、間を入れ替えても同じとみなせる帰無（同じ間を無作為な順に並べたもの）を添える。作例は、打鍵の時刻を速さと休みの 2 つの量を通して、1 文 1 枚の陰影のタイルに変える変換である。

<p align="center">
  <img alt="短い合成の文章の 1 文ごとの灰色のタイル 11 枚を 5 行。行の名は before・after・none・inside・shuffled。before と inside の行は同じ"
       src="docs/media/atlas.png" width="100%">
</p>
<p align="center"><sub>この目的のために書いた 1 つの文章の合成の打鍵（run v4、プレプリントの図 3）。1 文 1 枚のタイルを、文の前の間をその文に数える（<b>before</b>）、前の文に数える（<b>after</b>）、どれにも数えない（<b>none</b>）、休みを同じ文の中へ動かす（<b>inside</b>）、すべての間を並べ替える（<b>shuffled</b>）で描いた。<i>before</i> と <i>inside</i> は同じである。タイルは間の順を持たないからである。この README のどこにも人の記録は載せていない。</sub></p>

## 打鍵がタイルになるまで

<p align="center">
  <img alt="アニメーション。合成の文章の 1 文ごとに 1 コマ: 時間軸の上の打鍵（休みは橙）、2 秒の閾値に対する文の間（対数）、
       速さ p と休み q、規則の細かさ f・深さ s・粗さ γ、タイル、そこまでのタイルの列" src="docs/media/process.gif" width="900">
</p>

1 つの文について、速さ *p* は 2 秒より短い間の中央値を、速い打ち手と遅い打ち手の平均の間隔のあいだに対数で置いたもの、休み *q* は 2 秒以上の間に費やした時間の割合を 1 から引いたものである。感情から触覚のレリーフへの対応から形を借りた規則が、*p* と *q* からノイズの場の細かさ *f*・深さ *s*・粗さ γ を決め、場を灰色で描く。アニメーションは上の図と同じ文章の 11 文を v4 の系列でたどる。作るツール `tools/readme_media.py` は v4 の関数でタイルを作り直し、*p*・*q*・速さ・休みの割合とすべての画素が v4 の記録と図に一致しなければ止まる。アニメーションはその後で各コマを 128 色に減らしている。説明のための図であり、論文の数はここから取っていない。

<p align="center">
  <img alt="図: 打鍵から S0（時刻）、S1（2 つの量）、S2（形）、SEQ（タイルの順）、IMG（タイルの列）と参照の観測量 R へ。
       各段で行う 2 つの数えを箱に記す" src="docs/media/pipeline.svg" width="900">
</p>

監査した段（プレプリントの図 1）。どの段も記録の部分ごとの元の系列の要約なので、段のあいだの差には、失われたものと、距離が特徴をどう重みづけるかの違いが混ざる。参照の観測量 R は、各間がどの種類の境界の前にあるかを読む。画像の段はこれを読まない。操作で画像を判定するのは、その操作が R を動かす場合に限る。

## プレプリントの図と表の出どころ

計算はすべて、スクリプトを初めて走らせる前に git にコミットした条件の上で行った。各記録には、条件・スクリプト・入力の SHA-256 と git のコミットが入っている。run `vX` の条件は `works/conditions/prototype_vX.json`、スクリプトは `works/prototype_vX.py`（`works/_common.py` と `src/keystroke.py` を使う）、記録とノートは `results/prototype_vX/` にある。

| プレプリント | run |
|---|---|
| 図 1（段） | 3 節から手で描いた図。数は無い |
| 図 2（操作、すべてのセル） | v6b・v6e。手順 4 で読めるセルかは v6f |
| 図 3（合成のアトラス） | v4（`results/prototype_v4/seasons_sentence_external_f42.png`） |
| 表 1（人） | v6・v6c・v6i・v6j |
| 表 2（操作） | v6b・v6d・v6e |
| 表 3（タイルの列） | v6f・v6g・v6h |
| 表 4（記録と、その入れ替えた間の系列との比べ） | v6i・v6j |

run の順と、条件と記録のコミット。プレプリント（補足 A）が名指すものと、この履歴でのものを並べる。

| run | 位置づけ | 問うたこと | プレプリント: 条件・記録 | ここ: 条件・記録 |
|---|---|---|---|---|
| v1 | b | 4 つの文章の合成の打鍵、30 秒の窓 | c22cd74, 2d033fd | 同じ |
| v2 | b | 10 秒の窓。文章の間に合わせた打鍵の型 | 5502d13, 80adbc2 | 同じ |
| v3 | b | 1 文 1 枚 | acd4f6c, 173753c | 同じ |
| v4 | b | アトラス: 休みの場所、文の前の間の数え方、場の種 | 4cc995d, 7afc35b | 同じ |
| v5 | b | v4 のタイルを深さと細かさで | 8fe045c, 6fd5d61 | 同じ |
| v6 | p | 人のデータでの最初の数え: 人と 3 つの操作（KUPA、136M Keystrokes） | 06b46c2, e3a7fc3 | 06b46c2, 2296b67 |
| v6b | c | 2 つ目の分け方と参照の R。KUPA の新しい 200 人で確かめた | 8bec59a, 071087b | e62dd13, afcfbbc |
| v6c | d | rank-1・rank-20・EER。人数への依存 | c0927ce, e436f43 | d5c5446, bc63d93 |
| v6d | d | R を位置と lag-1 に分ける | 1b70dbe, 17a3ddc | ac659e1, 241abae |
| v6e | c | v6b と v6d を 2 つ目のデータ（LWP）で。確かめの規則は KUPA から先に決めた | cbffc5d, 8d38d89 | aabe8ee, 6efd81e |
| v6f | d | タイルの列と、画像の段の陽性対照 | 3ef9868, 2abf5af | cb787a6, 72d0ed0 |
| v6g | d | 細かい物差し: タイルを自分の間から引き直す | 652ad2e, be5f062 | 1fd4441, 34a3a39 |
| v6h | d | v6g の物差しのための、入れ替えた間の帰無 | 713cdf7, fabcdae | 084006e, 442365a |
| v6i | d | 各記録を自分の入れ替えた系列と比べる。画像に近い段での見分け | 94ceb8c, 3f1202d | 1bc3a87, 70089ac |
| v6j | d | ゆっくりした流れの下の陽性対照。タイルを無作為な順にした DTW。登録側の尺度 | d46e4b4, ab5ab80 | 3605277, 796727a |

位置づけはプレプリントと同じである: **p** 事前に指定、**c** 確かめ、**d** 同じ人での分解、**b** 作りから決まる。v1〜v5 は合成の打鍵だけを使い、`works/texts/` の文章（著者が書いたもの）から作った。v6〜v6j は公開されている 3 つの打鍵のデータを数える。記録にあるのは人をまたぐ数と要約（要約の最小と最大は 1 人の値）で、人の ID は数と SHA-256 だけである。条件・ノート・コード・コミットのメッセージにある `.steering/` は作業のノートを指し、アーカイブには入れていない。

## 休止の要約の論文（版 1.1.0）

2 つ目の論文は、書く過程の研究が打鍵の記録から報告する要約に監査を当てる。外の目盛りを使わない文の要約（U）、P-burst の平均の長さとその散らばり（B・B_sd）、KLiCKe の休止の平均（I）を、KUPA-KEYS の作文と LWP で数える。要約を 1 人について読むのは、事前に決めた陽性対照が同じ人で同じ操作に届くときに限る。run は v6k〜v6s で、条件・コード・テストは `works/` と `tests/` にあり、作業用のリポジトリの順に、条件をコードより先にコミットしている。コードを走らせたコミットを、論文（付録 A）が名指すものと、ここでのものとで並べる。

| run | 論文: コミット | ここ: コミット | 条件（SHA-256） |
|---|---|---|---|
| v6k | f940b81 | cfa00cf | febecc6c |
| v6l | 0289799 | d646fb4 | 96c9563b |
| v6m | bf5f640 | 12584ea | 43eb39e3 |
| v6n | 0c22760 | ca76249 | b10f9f12 |
| v6o | b636d51 | ab30e35 | 05d21d59 |
| v6p | b97b9db | ce2f57d | 0f328d36 |
| v6q | fa25ee6 | 8024c0c | 9a268caf |
| v6r | 3a89682 | 0bc91f6 | 4c8bfb87 |
| v6s | 1975fcd | edfcfdf | b03bc1e2 |

この論文のためにここにあるもの:

- `article/jowr/values/`: 論文が印字する値のすべてを、印字する形のまま置いた（本文は `numbers.json`、`fig2.tex` と `fig2-halves.tex`、付録の表は `app-*.tex`）。記録からそれを作るスクリプト（`paper_numbers.py`・`appendix_tables.py`）と、出力ごとにそれを作るために読んだ記録のセル（キーの道筋だけで値は無い）を並べた `sources.json` もある。このアーカイブにある値は論文自身の文字列であり、記録のほかの値は公開していない（下）。
- 開示の台帳（`ledger_inventory.py`・`ledger_classify.py`・`ledger_known_unexplained.json`）: 論文・付録・ここにある条件が書く記録の値を、それぞれを数えた書き手の集合とともに並べ、書いた値から 5 人未満の和が線形に導けると止まる（付録 A）。v6l・v6m・v6o・v6q の条件は、理由として前の記録の値を引いている。台帳はその引用も書いた値として数える。
- `results/record-digests.json`・`tools/record_digest.py`・`tools/rebuild_records.py`: 元の各記録の SHA-256 と、来歴を除いた中身の SHA-256。v6k〜v6s を順に作り直し、作り直した記録が元の中身と同じでなければ止まるツール（下の「再現」）。
- `docs/procedure.md`: コードの動かし方、KLiCKe のコーパスの欄と手順が要る入力との対応、ランダムな 2 分割で判定を数えるやり方（付録 H）。
- `docs/literature.md`: 引いた文献ごとにどの本文をどこまで読んだかと、検索の記録（付録 H）。

## このアーカイブに入れていないものと理由

| 入れていないもの | 理由 | 代わりにここにあるもの |
|---|---|---|
| 打鍵のデータ（KUPA-KEYS、136M Keystrokes、LWP） | それぞれの条件のもとで入手元に置かれている | v6 と v6e の条件にある入手元・パス・SHA-256（下の「再現」） |
| run v6 の図 6 枚（`results/prototype_v6/{kupa_copy,kupa_essay,aalto_session}_{tiles,scatter}.png`） | 1 人ずつの記録を描いている。論文は人をまたぐ数と要約だけを公開する | `results/prototype_v6/record.json` にある SHA-256。すべてのコミットから除いた（下の「履歴と来歴」） |
| 原稿・補足と、その数・表・図 2 と図 3 を書くスクリプト | プレプリントの側に置いている。読むのはここにある記録だけである | プレプリントと補足（[10.5281/zenodo.23858734](https://doi.org/10.5281/zenodo.23858734)） |
| run v6k〜v6s の記録とノート（`results/prototype_v6k/`〜`results/prototype_v6s/`） | 論文は自分が書く値だけを共有する（開示の規則: 共有した値から 5 人未満の和が導けるセルを残さない。付録 A）。記録はそれよりずっと多くの値を持ち、ノートはそれを引く | 条件・コード・テスト。論文が書く値と、それが来た記録のセル（`article/jowr/values/`）。記録の SHA-256 と中身の SHA-256（`results/record-digests.json`）と、公開のデータから作り直すツール（`tools/rebuild_records.py`）。これらの記録やノートだけを足したコミットはこの履歴に無い |
| 2 つ目の論文の原稿の元 | 論文の側に置いている | 印字した値（`article/jowr/values/`） |
| 後の別の研究のコード・条件・テスト（run v7 と `capture/`） | これらの論文の一部ではない | 無い。この履歴にも入れていない |
| 作業のノート（`advice/`・`handoff/`・`.steering/`） | 作業の記録で、計算の入力ではない | スクリプトを走らせるのには要らない |

## 再現

必要なものは [uv](https://docs.astral.sh/uv/) と git である。テストと実行はすべて Apple silicon の macOS で行った。生成器 `atact` は公開リポジトリから固定したコミットで入る（`pyproject.toml`。保管先 [10.5281/zenodo.23681773](https://doi.org/10.5281/zenodo.23681773)）。

```bash
uv sync --frozen                                 # Python 3.13 と uv.lock の固定版
uv run --frozen pytest                           # 384 passed、5 skipped。v6k〜v6s の記録があれば 659 passed
uv run --frozen python tools/readme_media.py     # この README の画像。v4 の記録と照合する
```

v1〜v5 はほかに何も要らない。run は結果を上書きせず、条件・コード・入力が最後のコミットと同じかを確かめるので、手元のブランチで結果を退けてから走らせる。

```bash
git switch -c rerun
git mv results/prototype_v4 results/prototype_v4.archived && git commit -m "move v4 aside"
uv run --frozen python works/prototype_v4.py
```

（`tools/readme_media.py` は `results/prototype_v4/` を読むので、このブランチでは走らない。）

この版で v4 をこうして走らせ直すと、記録は来歴（コミットと、後の run が関数を足した `src/keystroke.py` の SHA-256）を除いて同じになり、図はバイト単位で同じになった（Apple silicon の macOS）。

v6〜v6s は、条件に書いたパスの打鍵のデータを読み、ファイルの SHA-256 が違えば止まる（v6k〜v6s が読むのは KUPA-KEYS と LWP）。

| データ | 入手元 | 条件 | パス |
|---|---|---|---|
| KUPA-KEYS（Velentzas ほか、LREC-COLING 2024） | https://huggingface.co/datasets/ALTACambridge/KUPA-KEYS | CC BY-NC-SA 4.0 | `external/kupa-keys/KUPA-KEYS-TASK-1.csv`、`KUPA-KEYS-TASK-2.csv` |
| 136M Keystrokes（Dhakal ほか、CHI 2018） | http://userinterfaces.aalto.fi/136Mkeystrokes | 非営利・著者の表示 | `external/aalto-136m/Keystrokes.zip` |
| Linking Writing Processes to Writing Quality（Kaggle、2023） | https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality | コンペティションの規則 | `external/kaggle-lwp/data/train_logs.csv` |

v6k〜v6s の記録は、データを置いた上で手元のブランチで作り直す。

```bash
git switch -c rebuild
uv run --frozen python tools/rebuild_records.py
```

ツールは v6k〜v6s を順に走らせ、各記録が元の中身（来歴を除いた SHA-256）と同じでなければ止まる。作り直した記録は来歴（時刻・コミット）が元のファイルと違うので、後の run の条件が名指す SHA-256 と合わなくなる。ツールは手元のブランチで、後の条件にあるその SHA-256 と、それで変わった条件のファイルの SHA-256（後の条件は前の条件も名指す）だけを置き換え、記録とともにコミットする（`docs/procedure.md`）。この版を保管する前に、新しい clone から macOS 27.0.1（Apple silicon）・uv 0.11.1・Python 3.13.12 で 9 本の記録をすべて元の中身どおりに作り直せることを確かめた。そのうえでテストは 659 passed、`article/jowr/values/` の scripts は本文の 135 の値と付録の表をすべて論文どおりに出した。違ったのは run の表（`app-runs.tex`）だけで、これは各記録の時刻・コミット・SHA-256 を印字する表である。

## 履歴と来歴

この履歴は、計算を行った作業用のリポジトリから、run v6 の図 6 枚をすべてのコミットから除いて作った。日付・順序・メッセージ・作者は変えていない。そのため、除いた最初の図より後のコミットは、論文と記録が名指すものと違う。`commit-map.tsv` が元のコミットとここのコミットを対応させる。対応表の最後のコミットより後は、README・使用許諾・対応表を足すコミット（版 1.0.0）と、版 1.0.1 でこの README の画像・それを作るツール・日本語の README・`CITATION.cff` を足すコミットである。

版 1.1.0 は、v6j の後の履歴を続ける。作業用のリポジトリのコミットのうち、v6k〜v6s の条件・コード・テストと、前のスクリプトの一時ファイルの直しを持つものを同じやり方で（日付・メッセージ・作者を変えずに）写し、版 1.0.1 のコミットの後に置いた。v6k〜v6s の記録やノートだけを足したコミットと、後の研究のコミットは入れていない。`commit-map.tsv` はこれらのコミットも対応させる。最後の 2 つのコミットがこの版のファイルとその DOI を足す。

版: 1.0.0（[10.5281/zenodo.23858736](https://doi.org/10.5281/zenodo.23858736)。1 つ目の論文のプレプリントが引く版）、1.0.1（[10.5281/zenodo.23862942](https://doi.org/10.5281/zenodo.23862942)。README の画像とそのツール、`README.ja.md`、`CITATION.cff` を足し、README を書き直した版）。1.1.0（[10.5281/zenodo.24095177](https://doi.org/10.5281/zenodo.24095177)。v6k〜v6s の条件・コード・テストと 2 つ目の論文のファイルを足した版。2 つ目の論文が引く）。[10.5281/zenodo.23858735](https://doi.org/10.5281/zenodo.23858735) は最新の版を指す。

## 引用

使う論文とこのアーカイブを引くこと（`CITATION.cff` にもある）。

> Miura, M. and Harada, Y. Can a pause summary show where one writer paused? An audit procedure for keystroke logs.
> Manuscript submitted to the *Journal of Writing Research*, 2026.

> Miura, M. and Harada, Y. KeepCount: A Stage-by-Stage Audit of What Images of Behavioral Data Keep, with Keystroke
> Tiles as a Worked Example. Preprint, Zenodo, 2026. https://doi.org/10.5281/zenodo.23858734

> Miura, M. and Harada, Y. (2026). KeepCount: conditions, code, records and history. Zenodo.
> https://doi.org/10.5281/zenodo.23858735

この DOI は最新の版を指す。論文が使った版を引くときは、「履歴と来歴」の節の版の一覧にあるその版の DOI を引く。1 つ目の論文のプレプリントは 1.0.0、2 つ目の論文は 1.1.0 である。

規則の形は第 2 著者のプレプリントによる。

> Harada, Y. (2026). Affective Tactility: Cross-Modal Translation from Natural Language to Procedural Height Maps.
> Preprint, Research Square. https://doi.org/10.21203/rs.3.rs-8696852/v1

## 使用許諾

コードは MIT License（`LICENSE`）、文書・条件・記録・文章・画像は CC BY-NC-SA 4.0（`LICENSE-docs`）である。`LICENSE-docs` には、v6〜v6s の記録が数えた打鍵のデータの条件も書いてある。

## 著者

- Mikoto Miura、Independent Researcher（[ORCID 0009-0000-4196-0508](https://orcid.org/0009-0000-4196-0508)）
- Yuusuke Harada、Graduate School of Humanities and Social Sciences, Hiroshima University, Japan（[ORCID 0009-0006-7252-5115](https://orcid.org/0009-0006-7252-5115)）
