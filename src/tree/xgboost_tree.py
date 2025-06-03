import numpy as np

class XGBoostNode:
    def __init__(
            self,
            feature_index=None,
            threshold=None,
            category=None,
            feature_type=None,
            value=None,
            left_child=None,
            right_child=None,
            is_leaf=False,
            # デバッグや可視化用
            sum_g=None,
            sum_h=None,
            num_samples=None,
            depth=None,
            gain=None,
    ):
        self.feature_index = feature_index  # 分割に使用する特徴量のインデックス
        self.threshold = threshold          # 分割の閾値
        self.category = category            # カテゴリ変数の値（もしあれば）
        self.feature_type = feature_type    # 特徴量の型（numeric, categorical）
        self.value = value                  # 葉ノードの場合の最適重み（w_j^*）
        self.left_child = left_child        # 左の子ノード
        self.right_child = right_child      # 右の子ノード
        self.is_leaf = is_leaf              # 葉ノードかどうか

        # デバッグや可視化用の情報
        self.sum_g = sum_g                  # このノードに属するサンプルのg_iの合計
        self.sum_h = sum_h                  # このノードに属するサンプルのh_iの合計
        self.num_samples = num_samples      # このノードに属するサンプル数
        self.depth = depth                  # ノードの深さ
        self.gain = gain                    # このノードでの分割によって得られたGain（中間ノードの場合）

    def __str__(self): # デバッグ表示用
        if self.is_leaf:
            return (f"Leaf(value={self.value:.4f}, samples={self.num_samples}, "
                    f"sum_g={self.sum_g:.2f}, sum_h={self.sum_h:.2f}, depth={self.depth})")
        else:
            if self.feature_type == 'numeric':
                condition = f"Feat_{self.feature_index} <= {self.threshold:.2f}"
            else:
                condition = f"Feat_{self.feature_index} == {self.category}"
            return (f"Node({condition}, samples={self.num_samples}, gain={self.gain:.4f}, "
                    f"sum_g={self.sum_g:.2f}, sum_h={self.sum_h:.2f}, depth={self.depth})")
        

class XGBoostTree:
    def __init__(
            self,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            reg_lambda=1.0,
            gamma=0.0,
            max_features=None,
    ):
        self.root = None                            # 木のルートノード
        self.max_depth = max_depth                  # 木の最大深さ
        self.min_samples_split = min_samples_split  # ノードを分割するための最小サンプル数
        self.min_samples_leaf = min_samples_leaf    # 葉ノードに残すための最小サンプル数
        self.reg_lambda = reg_lambda                # 正則化パラメータ λ
        self.gamma = gamma                          # 木の複雑度ペナルティ
        self.max_features = max_features            # 分割に使用する特徴量の最大数（Noneの場合は全ての特徴量を使用）
        self.feature_types = None                   # fit時に設定される特徴量の型

    def _calculate_leaf_weight(self, G_node, H_node):
        """
        葉ノードの重みを計算する関数
        parameters:
            G_node: ノードに属するサンプルのg_iの合計
            H_node: ノードに属するサンプルのh_iの合計
        """
        # w_j^* = - G_j / (H_j + λ)
        return -G_node / (H_node + self.reg_lambda)
    
    def _calculate_split_gain(self, G_parent, H_parent, G_left, H_left, G_right, H_right):
        """
        分割によるGainを計算する関数
        parameters:
            G_parent: 親ノードに属するサンプルのg_iの合計
            H_parent: 親ノードに属するサンプルのh_iの合計
            G_left: 左の子ノードに属するサンプルのg_iの合計
            H_left: 左の子ノードに属するサンプルのh_iの合計
            G_right: 右の子ノードに属するサンプルのg_iの合計
            H_right: 右の子ノードに属するサンプルのh_iの合計
        """
        # Gain = 0.5 * [G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - (G_L+G_R)^2/(H_L+H_R+λ)] - γ
        term_left = G_left**2 / (H_left + self.reg_lambda)
        term_right = G_right**2 / (H_right + self.reg_lambda)
        term_parent = (G_parent**2) / (H_parent + self.reg_lambda)
        gain = 0.5 * (term_left + term_right - term_parent) - self.gamma

        return gain
    
    def _find_best_split(self, X_node, g_node, h_node):
        '''
        最適な分割を見つける関数
        parameters:
            X_node: ノードに属するサンプルの特徴量
            g_node: ノードに属するサンプルのg_iの合計
            h_node: ノードに属するサンプルのh_iの合計
        '''
        best_split = {'gain': -np.inf} # gainを最大化するので負の無限大で初期化
        n_samples_node, n_total_features = X_node.shape

        # 現在のノードのGとH
        G_parent_node = np.sum(g_node)
        H_parent_node = np.sum(h_node)

        # min_samples_splitは分割のための最小サンプル数
        if n_samples_node < self.min_samples_split:
            return None
        
        # 特徴量のランダムサブセットを選択（max_featuresの処理）
        if self.max_features is not None:
            num_features_to_consider = n_total_features
        elif isinstance(self.max_features, int):
            num_features_to_consider = min(self.max_features, n_total_features)
        elif isinstance(self.max_features, float):
            num_features_to_consider = int(self.max_features * n_total_features)
            num_features_to_consider = max(1, num_features_to_consider)
        elif self.max_features == 'sqrt':
            num_features_to_consider = int(np.sqrt(n_total_features))
            num_features_to_consider = max(1, num_features_to_consider)
        elif self.max_features == 'log2':
            num_features_to_consider = int(np.log2(n_total_features))
            num_features_to_consider = max(1, num_features_to_consider)
        else:
            num_features_to_consider = n_total_features
        
        if num_features_to_consider < n_total_features:
            feature_indices_to_consider = np.random.choice(n_total_features, num_features_to_consider, replace=False)
        else:
            feature_indices_to_consider = np.arange(n_total_features)
        
        for feature_idx in feature_indices_to_consider:
            feature_values_at_node = X_node[:, feature_idx] # 現在のノードの当該特徴量の値
            unique_sorted_values = np.unique(feature_values_at_node)

            if self.feature_types[feature_idx] == 'numeric':
                if len(unique_sorted_values) > 1:
                    split_candidates = (unique_sorted_values[:-1] + unique_sorted_values[1:]) / 2
                else:
                    split_candidates = []

                for threshold_candidate in split_candidates:
                    # この閾値で分割した場合の左右の子ノードのインデックス
                    left_child_indices_bool = feature_values_at_node <= threshold_candidate
                    right_child_indices_bool = feature_values_at_node > threshold_candidate

                    # 葉の最小サンプル数を満たすかチェック
                    if np.sum(left_child_indices_bool) < self.min_samples_leaf or np.sum(right_child_indices_bool) < self.min_samples_leaf:
                        continue

                    # 左右の子ノードのgとhを計算
                    G_left_child = np.sum(g_node[left_child_indices_bool])
                    H_left_child = np.sum(h_node[left_child_indices_bool])
                    G_right_child = np.sum(g_node[right_child_indices_bool])
                    H_right_child = np.sum(h_node[right_child_indices_bool])

                    current_split_gain = self._calculate_split_gain(
                        G_parent_node, H_parent_node,
                        G_left_child, H_left_child,
                        G_right_child, H_right_child
                    )

                    if current_split_gain > best_split['gain']:
                        best_split = {
                            'gain': current_split_gain,
                            'feature_index': feature_idx,
                            'threshold': threshold_candidate,
                            'type': 'numeric',
                            'left_indices_bool': left_child_indices_bool,
                            'right_indices_bool': right_child_indices_bool,
                        }
            
            elif self.feature_types[feature_idx] == 'categorical':
                # XGBoostの標準ライブラリはカテゴリ特徴量をより効率的に扱うが、
                # ここでは単純に全てのカテゴリ値で分割を試みる

                for category_split_value in unique_sorted_values:
                    left_child_indices_bool = feature_values_at_node == category_split_value
                    right_child_indices_bool = feature_values_at_node != category_split_value

                    if np.sum(left_child_indices_bool) < self.min_samples_leaf or np.sum(right_child_indices_bool) < self.min_samples_leaf:
                        continue

                    G_left_child = np.sum(g_node[left_child_indices_bool])
                    H_left_child = np.sum(h_node[left_child_indices_bool])
                    G_right_child = np.sum(g_node[right_child_indices_bool])
                    H_right_child = np.sum(h_node[right_child_indices_bool])

                    current_split_gain = self._calculate_split_gain(
                        G_parent_node, H_parent_node,
                        G_left_child, H_left_child,
                        G_right_child, H_right_child
                    )

                    if current_split_gain > best_split['gain']:
                        best_split = {
                            'gain': current_split_gain,
                            'feature_index': feature_idx,
                            'category': category_split_value,
                            'type': 'categorical',
                            'left_indices_bool': left_child_indices_bool,
                            'right_indices_bool': right_child_indices_bool,
                        }
            
        if best_split['gain'] <= 0:
            return None
        return best_split
    
    def _build_tree_recursive(self, X_node, g_node, h_node, current_depth):
        '''
        再帰的に木を構築する
        parameters:
            X_node: ノードに属するサンプルの特徴量
            g_node: ノードに属するサンプルのg_iの合計
            h_node: ノードに属するサンプルのh_iの合計
            current_depth: 現在のノードの深さ 
        '''
        n_samples_node = X_node.shape[0]
        sum_g_at_node = np.sum(g_node)
        sum_h_at_node = np.sum(h_node)

        # 停止条件1: 最大深さに達した場合
        if current_depth >= self.max_depth:
            leaf_weight = self._calculate_leaf_weight(sum_g_at_node, sum_h_at_node)
            return XGBoostNode(
                value=leaf_weight,
                is_leaf=True,
                sum_g=sum_g_at_node,
                sum_h=sum_h_at_node,
                num_samples=n_samples_node,
                depth=current_depth
            )
    
        # 最適な分割を見つける
        best_split = self._find_best_split(X_node, g_node, h_node)

        # 停止条件2: 分割が見つからない場合
        if best_split is None:
            leaf_weight = self._calculate_leaf_weight(sum_g_at_node, sum_h_at_node)
            return XGBoostNode(
                value=leaf_weight,
                is_leaf=True,
                sum_g=sum_g_at_node,
                sum_h=sum_h_at_node,
                num_samples=n_samples_node,
                depth=current_depth
            )
        
        # 分割を実行して子ノードを再帰的に構築
        left_indices_bool = best_split['left_indices_bool']
        right_indices_bool = best_split['right_indices_bool']

        lef_child_node = self._build_tree_recursive(
            X_node[left_indices_bool],
            g_node[left_indices_bool],
            h_node[left_indices_bool],
            current_depth + 1
        )
        right_child_node = self._build_tree_recursive(
            X_node[right_indices_bool],
            g_node[right_indices_bool],
            h_node[right_indices_bool],
            current_depth + 1
        )

        # 現在のノードを作成
        node_params = {
            'feature_index': best_split['feature_index'],
            'feature_type': best_split['type'],
            'left_child': lef_child_node,
            'right_child': right_child_node,
            'is_leaf': False,
            'sum_g': sum_g_at_node,
            'sum_h': sum_h_at_node,
            'num_samples': n_samples_node,
            'depth': current_depth,
            'gain': best_split['gain'],
        }
        if best_split['type'] == 'numeric':
            node_params['threshold'] = best_split['threshold']
        else:
            node_params['category'] = best_split['category']

        return XGBoostNode(**node_params)
    
    def fit(self, X_train, g_train, h_train, feature_type_list):
        '''
        与えられた一階微分g_iと二階微分h_iを用いて木を学習する
        parameters:
            X_train: 学習データの特徴量
            g_train: 学習データの一階微分
            h_train: 学習データの二階微分
            feature_type_list: 特徴量の型リスト（'numeric' or 'categorical'）
        '''
        self.feature_types = feature_type_list
        self.root = self._build_tree_recursive(X_train, g_train, h_train, current_depth=0)

    def predict(self, X_test):
        '''
        学習済みの木を用いて予測を行い葉ノードの重みを返す
        '''
        if self.root is None:
            raise ValueError("The model has not been trained yet. Please call fit() first.")
        
        predictions = np.zeros(X_test.shape[0])

        for i in range(X_test.shape[0]):
            current_node = self.root
            while not current_node.is_leaf:
                sample_feature_value = X_test[i, current_node.feature_index]
                if current_node.feature_type == 'numeric':
                    if sample_feature_value <= current_node.threshold:
                        current_node = current_node.left_child
                    else:
                        current_node = current_node.right_child
                else:
                    if sample_feature_value == current_node.category:
                        current_node = current_node.left_child
                    else:
                        current_node = current_node.right_child
            predictions[i] = current_node.value
        return predictions

class XGBoostRegressor:
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=3,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 reg_lambda=1.0,
                 gamma=0.0,
                 max_features=None,
                 random_state=None,
                 #初期予測値
                 base_score=0.0
        ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda        # L2正則化
        self.gamma = gamma              # 木の複雑度ペナルティ
        self.max_features = max_features    # 分割に使用する特徴量の最大数
        self.random_state = random_state
        
        self.base_score = base_score  # 初期予測値 F₀(x)
        self.trees_ = []
        self.feature_types = None

    def calculate_gradients_mse(self, y_true, y_pred):
        '''
        二乗誤差損失 L = 0.5 * (y_true - y_pred) ** 2 の場合
        '''
        # g_i = dL/dy_pred = y_pred - y_true
        g = y_pred - y_true
        # h_i = d^2L/dy_pred^2 = 1
        h = np.ones_like(y_true)
        return g, h
    
    def fit(self, X_train, y_train, feature_types_ls):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.trees_ = []
        self.feature_types = feature_types_ls
        n_samples, n_features = X_train.shape

        # 1. 初期モデル F₀(x) の決定
        current_predictions = np.full(shape=n_samples, fill_value=self.base_score)
        
        for i in range(self.n_estimators):
            # 一階微分 g と 二階微分 h を計算
            g, h = self.calculate_gradients_mse(y_train, current_predictions)

            # gとhを用いて新しい木を学習
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                max_features=self.max_features
            )
            tree.fit(X_train, g, h, self.feature_types)
            self.trees_.append(tree)

            # モデルの更新
            tree_predictions_for_update = tree.predict(X_train)
            current_predictions += self.learning_rate * tree_predictions_for_update

            # 学習の進捗を表示
            progress_interval = self.n_estimators // 10 if self.n_estimators >= 10 else 1
            if (i + 1) % progress_interval == 0 or i == self.n_estimators - 1:
                mse = np.mean((y_train - current_predictions) ** 2)
                print(f"Iteration {i + 1}/{self.n_estimators}, MSE: {mse:.4f}")
        print("Training completed.")

    def predict(self, X_test):
        if not self.trees_:
            if X_test.shape[0] > 0:
                return np.full(shape=X_test.shape[0], fill_value=self.base_score)
            else:
                np.array([])

        # 初期予測値 F₀(x) を設定
        current_predictions = np.full(shape=X_test.shape[0], fill_value=self.base_score)

        # 各木の葉の重みを学習率をかけて加算
        for tree in self.trees_:
            current_predictions += self.learning_rate * tree.predict(X_test)

        return current_predictions