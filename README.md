# Beyond static snapshots: capturing aging momentum for enhanced lifespan prediction in mice

本リポジトリは、論文にて提案したデータセット $F_2^{\mathrm{cdrts}}$ の予測性能を、$F_1$ と比較・評価するためのソースコード一式を格納しています。

## 1. 動作確認済み環境 (Environment)
以下の環境において、正常な動作を確認しています。

* **OS**: Ubuntu 24.04.2 LTS (on Windows WSL2)
* **Platform**: VS Code (Jupyter Extension)
* **GPU**: NVIDIA TITAN RTX
* **Python**: 3.10.18
* **CUDA (PyTorch)**: 12.6
* **主要ライブラリ (Software Versions)**:
    * `torch`: 2.7.1+cu126
    * `scikit-learn`: 1.6.1
    * `pandas`: 2.2.3
    * `numpy`: 2.0.1
    * `xgboost`: 3.0.5

## 2. データの取得方法 (Data Availability)
本研究で使用したデータセットは、以下の Mendeley Data より公開されているものを利用しています。

* **データ名**: `Healthspan.data.paired.csv`
* **取得元**: [Mendeley Data (https://data.mendeley.com/datasets/kfkhgw359t/1)](https://data.mendeley.com/datasets/kfkhgw359t/1)
* **配置**: 上記URLよりデータをダウンロードし、ファイル名を **`physiological_lifespan_dataset.xlsx`** に変更した上で、`.ipynb` ファイルと同じディレクトリに配置してください。

## 3. ファイル構成 (Project Structure)
* `death_pred_func_mine_update2.py`:
    * ニューラルネットワーク（NN）モデルの定義。
    * 活性化関数として **GELU** を採用し、過学習抑制のために **Dropout層** を追加した改良版モデルを含みます。
    * クロスバリデーションを実行するための共通関数群が記述されています。
* `death_prediction_for_comparison2.ipynb`:
    * 解析のメイン実行ノートブックです。
    * 学習および評価を行い、予測値の Mean Absolute Error (MAE) を算出して **Excelファイル (.xlsx)** として保存します。

## 4. 実行手順 (How to Run)
1.  `death_pred_func_mine_update2.py` と `death_prediction_for_comparison2.ipynb` が同一階層にあることを確認します。
2.  `physiological_lifespan_dataset.xlsx` を同ディレクトリに配置します。
3.  VS Code の Jupyter 拡張機能を用いて `death_prediction_for_comparison2.ipynb` を開き、順にセルを実行してください。