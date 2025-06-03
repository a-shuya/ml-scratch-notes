# MLスクラッチ実装コレクション

手元でアルゴリズムを**一から実装**して仕組みを掘り下げる――そんな “写経 & 再発明” 的リポジトリです。  
以下の一覧にあるモデルを **Pure Python（＋NumPy, PyTorch）** でスクラッチ実装または概念の理解を目標に、完了したものから ✅ マークを付けていきます。

---

## 実装ステータス

> **凡例**  
> ✅ = 実装＆動作確認済み  
> ⏳ = 実装中  
> ☐ = 未着手

### 線形モデル
- ✅ **線形回帰**
- ✅ **リッジ回帰**
- ✅ **ラッソ回帰**
- ✅ **Elastic Net**
- ☐ **ロジスティック回帰**
- ☐ **線形SVM**
- ☐ **線形判別分析 (LDA)**
- ☐ **二次判別分析 (QDA)**
- ☐ **ポアソン回帰**
- ☐ **Bayesian Ridge**
- ☐ **ガウス過程回帰**

### ツリーベース
- ✅ **決定木**
- ✅ **ランダムフォレスト**
- ☐ **Extra Trees**
- ✅ **Gradient Boosting**
- ✅ **XGBoost**
- ⏳ **LightGBM**
- ☐ **CatBoost**
- ☐ **AdaBoost**

### カーネル／距離ベース
- ☐ **RBF・多項式SVM**
- ☐ **k近傍法 (k-NN)**
- ☐ **Gaussian Process (分類／回帰)**
- ☐ **OPTICS**
- ☐ **DBSCAN**

### 確率モデル・ベイズ推論
- ☐ **ネイーブベイズ**（Gaussian／Multinomial／Bernoulli）
- ☐ **Hidden Markov Model (HMM)**
- ☐ **Variational Inference (VI)**
- ☐ **Noise Contrastive Estimation (NCE)**
- ☐ **Monte Carlo Dropout**

### ニューラルネット／深層学習
- ✅ **多層パーセプトロン (MLP)**
- ✅ **畳み込みニューラルネット (CNN, LeNet-5)**
- ✅ **ResNet**
- ✅ **VGG**
- ⏳ **GoogleNet**
- ⏳ **ResNet**
- ☐ **DenseNet／EfficientNet／MobileNet／Inception**
- ☐ **Vision Transformer (ViT)／Swin Transformer**
- ☐ **U-Net／Mask R-CNN／YOLOv8**
- ☐ **リカレントネット (RNN)**
- ☐ **LSTM／GRU**
- ☐ **Transformer**
- ☐ **BERT／RoBERTa／DeBERTa／T5／BART／XLNet**
- ☐ **GPT／Llama／Mistral**
- ☐ **VAE**
- ☐ **GAN／CycleGAN／StyleGAN3**
- ☐ **Stable Diffusion**

### 次元削減・表現学習
- ☐ **主成分分析 (PCA)**
- ☐ **t-SNE**
- ☐ **UMAP**
- ☐ **独立成分分析 (ICA)**
- ☐ **非負値行列因子分解 (NMF)**
- ☐ **オートエンコーダ**

### クラスタリング
- ☐ **k-means**
- ☐ **階層クラスタリング**
- ☐ **Gaussian Mixture Model (GMM)**
- ☐ **Birch**
- ☐ **Self-Organizing Map (SOM)**

### 強化学習
- ☐ **Q-learning**
- ☐ **SARSA**
- ☐ **DQN／Double DQN／Dueling DQN**
- ☐ **Policy Gradient／REINFORCE**
- ☐ **Actor-Critic／A2C**
- ☐ **PPO**
- ☐ **TRPO**
- ☐ **DDPG**
- ☐ **TD3**
- ☐ **SAC**
- ☐ **IMPALA／R2D2**
- ☐ **AlphaZero／MuZero**

### 時系列解析
- ☐ **AR／MA／ARMA／ARIMA／SARIMA**
- ☐ **GARCH**
- ☐ **Exponential Smoothing**
- ☐ **Prophet**
- ☐ **Hawkes Process**

### 異常検知
- ☐ **Isolation Forest**
- ☐ **One-Class SVM**
- ☐ **Local Outlier Factor (LOF)**
- ☐ **Elliptic Envelope**

---

## 進め方

実装したいモデルがあれば逐次追加していきます。

