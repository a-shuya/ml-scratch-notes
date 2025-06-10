<!-- ===================================================== -->
<!-- 🇯🇵 Japanese                                            -->
<!-- ===================================================== -->

<p align="right">
  <a href="#-english">🇺🇸 English version below</a>
</p>

# MLスクラッチ実装コレクション

---

機械学習およびそれに付随するトピックのアルゴリズムを **NumpyやPyTorchのみで実装** し、仕組みを掘り下げる写経用リポジトリです。 

---

## 凡例

| マーク | 意味 |
| :----: | ---- |
| ✅ | 実装＆動作確認済み |
| ⏳ | 実装中 |

---

## ノートブック一覧

### 線形モデル
- ✅[重回帰](notebooks/Linear_Regression/01_multiple_regression.ipynb)  
- ✅[Ridge 回帰](notebooks/Linear_Regression/02_ridge.ipynb)  
- ✅[Lasso 回帰](notebooks/Linear_Regression/03_lasso.ipynb)  
- ✅[Elastic Net](notebooks/Linear_Regression/04_elastic_net.ipynb)  

### ツリーベース
- ✅[CART (Decision Tree)](notebooks/Tree/01_CART.ipynb)  
- ✅[Random Forest](notebooks/Tree/02_random_forest.ipynb)  
- ✅[Gradient Boosting (GBDT)](notebooks/Tree/03_gbdt.ipynb)  
- ✅[XGBoost](notebooks/Tree/04_xgboost.ipynb)  
- ⏳[LightGBM](notebooks/Tree/05_lightgbm.ipynb)  

### クラスタリング
- ✅[K-means](notebooks/clustering/01_kmeans.ipynb)  
- ✅[階層クラスタリング](notebooks/clustering/02_hierarchical.ipynb)  
- ✅[DBSCAN & GMM](notebooks/clustering/03_dense.ipynb)  
- ✅[スペクトラルクラスタリング](notebooks/clustering/04_spectral.ipynb)

### ニューラルネット
- ✅[MLP](notebooks/DL_fundamental/01_mlp.ipynb)  
- ✅[活性化関数](notebooks/DL_fundamental/02_activation_function.ipynb)
- ✅[損失関数](notebooks/DL_fundamental/03_loss_function.ipynb)
- ✅[重み初期化](notebooks/DL_fundamental/04_weight_initialization.ipynb)
- ✅[Optimizer](notebooks/DL_fundamental/05_optimizer.ipynb)
- ✅[Learning Rate](notebooks/DL_fundamental/06_learning_rate.ipynb)
- ✅[Normalization](notebooks/DL_fundamental/07_normalization.ipynb)

### CNNs
- ✅[CNN (LeNet-5)](notebooks/CNNs/01_CNN_lenet5.ipynb)  
- ✅[AlexNet](notebooks/CNNs/02_AlexNet.ipynb)  
- ✅[VGG](notebooks/CNNs/03_VGG.ipynb)  
- ⏳[GoogleNet](notebooks/CNNs/04_GoogleNet.ipynb)  
- ⏳[ResNet](notebooks/CNNs/05_ResNet.ipynb)  

### NLP Basic
- ✅[BoW & TF-IDF](notebooks/basic_NLP/01_text_preprocessing.ipynb) 
- ✅[Language Modeling](notebooks/basic_NLP/02_language_modeling.ipynb) 
- ✅[Word2Vec](notebooks/basic_NLP/03_word_embeddings.ipynb) 
- ✅[Sub-word Tokenization](notebooks/basic_NLP/04_subword_tokenization.ipynb) 
- ✅[HuggingFace Tokenizer](notebooks/basic_NLP/05_huggingface_tokenizer.ipynb) 


### RNNs
- ⏳[SRN](notebooks/RNNs/01_SRN.ipynb)  
- ⏳[BPTT](notebooks/RNNs/02_BPTT.ipynb)  
- ✅[LSTM](notebooks/RNNs/03_LSTM.ipynb)  
- ✅[GRU](notebooks/RNNs/04_GRU.ipynb)  

---

<!-- ===================================================== -->
<!-- 🇺🇸 English                                            -->
<!-- ===================================================== -->

<a id="-english"></a>
<p align="right">
  <a href="#mlスクラッチ実装コレクション">🇯🇵 日本語はこちら</a>
</p>

# ML Scratch-Implementation Collection

A hands-on repository for **implementing machine-learning algorithms from scratch** to truly understand how they work.  
Everything is written in **pure Python** (only NumPy and/or PyTorch as helpers).

---

## Legend

| Mark | Meaning |
| :--: | ------- |
| ✅ | Implemented & verified |
| ⏳ | Work in progress |

---

## Notebook List

### Linear Models
- ✅[Multiple Regression](notebooks/Linear_Regression/01_multiple_regression.ipynb)  
- ✅[Ridge Regression](notebooks/Linear_Regression/02_ridge.ipynb)  
- ✅[Lasso Regression](notebooks/Linear_Regression/03_lasso.ipynb)  
- ✅[Elastic Net](notebooks/Linear_Regression/04_elastic_net.ipynb)  

### Tree-Based Methods
- ✅[CART (Decision Tree)](notebooks/Tree/01_CART.ipynb)  
- ✅[Random Forest](notebooks/Tree/02_random_forest.ipynb)  
- ✅[Gradient Boosting (GBDT)](notebooks/Tree/03_gbdt.ipynb)  
- ✅[XGBoost](notebooks/Tree/04_xgboost.ipynb)  
- ⏳[LightGBM](notebooks/Tree/05_lightgbm.ipynb)  

### Clustering
- ✅[K-means](notebooks/clustering/01_kmeans.ipynb)  
- ✅[Hierarchical Clustering](notebooks/clustering/02_hierarchical.ipynb)  
- ✅[DBSCAN & GMM](notebooks/clustering/03_dense.ipynb)  
- ✅[Spectral Clustering](notebooks/clustering/04_spectral.ipynb)

### Neural Networks
- ✅[MLP](notebooks/DL_fundamental/01_mlp.ipynb)  
- ✅[Activation Functions](notebooks/DL_fundamental/02_activation_function.ipynb)
- ✅[Loss Functions](notebooks/DL_fundamental/03_loss_function.ipynb)
- ✅[Weight Initialization](notebooks/DL_fundamental/04_weight_initialization.ipynb)
- ✅[Optimizer](notebooks/DL_fundamental/05_optimizer.ipynb)
- ✅[Learning Rate](notebooks/DL_fundamental/06_learning_rate.ipynb)
- ✅[Normalization](notebooks/DL_fundamental/07_normalization.ipynb)
- ✅[Regularization](notebooks/DL_fundamental/08_other_techniques.ipynb)

### CNNs
- ✅[CNN (LeNet-5)](notebooks/CNNs/01_CNN_lenet5.ipynb)  
- ✅[AlexNet](notebooks/CNNs/02_AlexNet.ipynb)  
- ✅[VGG](notebooks/CNNs/03_VGG.ipynb)  
- ⏳[GoogleNet](notebooks/CNNs/04_GoogleNet.ipynb)  
- ⏳[ResNet](notebooks/CNNs/05_ResNet.ipynb)  

### NLP Basic
- ✅[BoW & TF-IDF](notebooks/basic_NLP/01_text_preprocessing.ipynb) 
- ✅[Language Modeling](notebooks/basic_NLP/02_language_modeling.ipynb) 
- ✅[Word2Vec](notebooks/basic_NLP/03_word_embeddings.ipynb) 
- ✅[Sub-word Tokenization](notebooks/basic_NLP/04_subword_tokenization.ipynb) #
- ✅[HuggingFace Tokenizer](notebooks/basic_NLP/05_huggingface_tokenizer.ipynb) 

### RNNs
- ⏳[SRN](notebooks/RNNs/01_SRN.ipynb)  
- ⏳[BPTT](notebooks/RNNs/02_BPTT.ipynb)  
- ✅[LSTM](notebooks/RNNs/03_LSTM.ipynb)  
- ✅[GRU](notebooks/RNNs/04_GRU.ipynb)



---

## How This Repository Evolves

- Whenever I finish an implementation and confirm it works, I add it above with a ✅  
- Items under development keep the ⏳ mark  
