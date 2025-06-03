## 引用論文リスト

以下は、これまで議論された機械学習アルゴリズムに関連する主要な論文のリストです。

### 1. 決定木 (Decision Trees)

*   **CART (Classification and Regression Trees)**:
    *   Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth & Brooks/Cole Advanced Books & Software.
    *(決定木の基本的なアルゴリズムであるCARTを提案した古典的な書籍)*

*   **C4.5 (ID3の改良版)**:
    *   Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers Inc.
    *(情報利得率や枝刈りなどの概念を導入した影響力のある決定木アルゴリズム)*

*   **GUIDE (Generalized, Unbiased, Interaction Detection and Estimation)**:
    *   Loh, W. Y. (2002). Regression trees with unbiased variable selection and interaction detection. *Statistica Sinica*, 12(2), 361-386.
    *(変数選択バイアスのない回帰木を提案。分類木に関する論文も別途存在します)*
    *   Loh, W. Y. (2009). Improving the precision of classification trees. *Annals of Applied Statistics*, 3(4), 1710-1737.
    *(分類木におけるGUIDEの改良など)*
    *   Loh, W.-Y. (2011). Classification and regression trees. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 1(1), 14-23.
    *(今回OCRで参照した、分類木と回帰木のレビュー論文)*

### 2. ランダムフォレスト (Random Forests)

*   **Random Forests (提案論文)**:
    *   Breiman, L. (2001). Random forests. *Machine learning*, 45(1), 5-32.
    *(ランダムフォレストのアルゴリズムを提案し、その理論と性能を示した独創的な論文)*

*   **Bagging (ランダムフォレストの基礎技術の一つ)**:
    *   Breiman, L. (1996). Bagging predictors. *Machine learning*, 24(2), 123-140.
    *(ブートストラップ集約（バギング）を提案した論文)*

### 3. 勾配ブースティング決定木 (Gradient Boosting Decision Trees - GBDT)

*   **Greedy Function Approximation (GBDTの基礎理論)**:
    *   Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of statistics*, 29(5), 1189-1232.
    *(勾配ブースティングの一般的なフレームワークを提案し、GBDTの基礎を築いた重要な論文。今回OCRで参照した論文)*

*   **Stochastic Gradient Boosting (GBDTの改良)**:
    *   Friedman, J. H. (2002). Stochastic gradient boosting. *Computational statistics & data analysis*, 38(4), 367-378.
    *(GBDTにランダムサンプリングを導入し、性能と計算効率を向上させた論文)*

### 4. XGBoost (Extreme Gradient Boosting)

*   **XGBoost (提案論文)**:
    *   Chen, T., & Guestrin, C. (2016, August). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining* (pp. 785-794).
    *(XGBoostのアルゴリズム、正則化、スケーラビリティのためのシステム設計について詳細に説明した論文。今回OCRで参照した論文)*

### 5. LightGBM (Light Gradient Boosting Machine)

*   **LightGBM (提案論文)**:
    *   Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. In *Advances in neural information processing systems* (Vol. 30).
    *(GOSSとEFBという新しい技術を導入し、GBDTの効率とスケーラビリティを大幅に向上させたLightGBMを提案した論文。今回OCRで参照した論文)*