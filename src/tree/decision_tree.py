# DecisionTree.py

import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_index=None, threshold=None, category=None, feature_type=None,
                 left_child=None, right_child=None, value=None, impurity=None, num_samples=None):
        """
        決定木のノード
        Args:
            feature_index (int): 分割に使われた特徴量のインデックス
            threshold (float): 数値特徴量の分割閾値
            category (any): カテゴリ特徴量の分割カテゴリ
            feature_type (str): 'numeric' or 'categorical'
            left_child (Node): 左の子ノード
            right_child (Node): 右の子ノード
            value (any): 葉ノードの場合の予測値 (分類ならクラス、回帰なら平均値)
            impurity (float): このノードの不純度 (分類ならGini、回帰ならMSEなど)
            num_samples (int): このノードのサンプル数
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.category = category
        self.feature_type = feature_type
        self.left_child = left_child
        self.right_child = right_child
        self.value = value
        self.impurity = impurity # GiniまたはMSEなど
        self.num_samples = num_samples

    def is_leaf_node(self):
        return self.value is not None

# --- 分類用ヘルパー関数 ---
def _calculate_gini(y_subset):
    if len(y_subset) == 0:
        return 0
    counts = Counter(y_subset)
    impurity = 1.0
    for key in counts:
        prob_of_key = counts[key] / len(y_subset)
        impurity -= prob_of_key**2
    return impurity

# --- 回帰用ヘルパー関数 ---
def _calculate_mse(y_subset):
    if len(y_subset) == 0:
        return 0
    mean_y = np.mean(y_subset)
    return np.mean((y_subset - mean_y)**2)

def _calculate_variance(y_subset): # MSEと同じだが、明示的に
    if len(y_subset) == 0:
        return 0
    return np.var(y_subset) * len(y_subset) # 重み付け分散を返す (親の分散 - Σ子の分散の減少を最大化)
                                         # または単純にnp.var(y_subset)でも良い (比較のため)
                                         # ここでは単純な分散 (MSEと同じ指標) を使う

class DecisionTreeBase:
    """決定木の共通基盤クラス"""
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, criterion_func=None, leaf_value_func=None):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_types = None
        self.max_features = max_features
        self._criterion_func = criterion_func # 不純度計算関数 (Gini or MSE)
        self._leaf_value_func = leaf_value_func # 葉ノードの値計算関数 (多数決 or 平均)

    def _find_best_split(self, X_subset, y_subset):
        best_split_info = None
        # Gini Reduction や Variance Reduction (MSE Reduction) を最大化
        max_reduction = -1.0 
        
        parent_impurity = self._criterion_func(y_subset)
        n_samples, n_total_features = X_subset.shape

        if n_samples <= 1:
            return None

        # 特徴量のランダムサブセットを選択 (max_features)
        if self.max_features is None:
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

        if num_features_to_consider < n_total_features :
            feature_indices_to_consider = np.random.choice(n_total_features, num_features_to_consider, replace=False)
        else:
            feature_indices_to_consider = np.arange(n_total_features)

        for feature_index in feature_indices_to_consider:
            feature_values = X_subset[:, feature_index]
            unique_values = np.unique(feature_values)

            if self.feature_types[feature_index] == 'numeric':
                sorted_unique_values = np.sort(unique_values)
                threshold_candidates = []
                if len(sorted_unique_values) > 1:
                    threshold_candidates = (sorted_unique_values[:-1] + sorted_unique_values[1:]) / 2
                
                for threshold in threshold_candidates:
                    left_bool_indices = feature_values <= threshold
                    right_bool_indices = feature_values > threshold
                    y_left = y_subset[left_bool_indices]
                    y_right = y_subset[right_bool_indices]

                    if len(y_left) == 0 or len(y_right) == 0: continue
                    
                    impurity_left = self._criterion_func(y_left)
                    impurity_right = self._criterion_func(y_right)
                    
                    weighted_impurity_children = (len(y_left) / n_samples) * impurity_left + \
                                                 (len(y_right) / n_samples) * impurity_right
                    
                    current_reduction = parent_impurity - weighted_impurity_children

                    if current_reduction > max_reduction:
                        max_reduction = current_reduction
                        best_split_info = {'feature_index': feature_index, 'threshold': threshold, 'type': 'numeric',
                                           'reduction': current_reduction, 
                                           'left_indices_bool': left_bool_indices, 'right_indices_bool': right_bool_indices}
            
            elif self.feature_types[feature_index] == 'categorical':
                for category_value in unique_values:
                    left_bool_indices = feature_values == category_value
                    right_bool_indices = feature_values != category_value
                    y_left = y_subset[left_bool_indices]; y_right = y_subset[right_bool_indices]

                    if len(y_left) == 0 or len(y_right) == 0: continue

                    impurity_left = self._criterion_func(y_left)
                    impurity_right = self._criterion_func(y_right)
                    weighted_impurity_children = (len(y_left) / n_samples) * impurity_left + (len(y_right) / n_samples) * impurity_right
                    current_reduction = parent_impurity - weighted_impurity_children

                    if current_reduction > max_reduction:
                        max_reduction = current_reduction
                        best_split_info = {'feature_index': feature_index, 'category': category_value, 'type': 'categorical',
                                           'reduction': current_reduction,
                                           'left_indices_bool': left_bool_indices, 'right_indices_bool': right_bool_indices}
        
        if max_reduction <= 0: return None # 不純度が減少しない場合は分割しない
        return best_split_info

    def _build_tree(self, X_data, y_data, depth=0):
        n_samples, n_features = X_data.shape
        current_impurity = self._criterion_func(y_data) # 現在のノードの不純度
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y_data)) == 1: # 純粋なノード (分類の場合) または全て同じ値 (回帰の場合、MSE=0)
            leaf_value = self._leaf_value_func(y_data)
            return Node(value=leaf_value, impurity=current_impurity, num_samples=n_samples)

        best_split_info = self._find_best_split(X_data, y_data)

        if best_split_info is None: # 有益な分割が見つからない
            leaf_value = self._leaf_value_func(y_data)
            return Node(value=leaf_value, impurity=current_impurity, num_samples=n_samples)
        
        left_indices_bool = best_split_info['left_indices_bool']
        right_indices_bool = best_split_info['right_indices_bool']
        
        # 分割後の葉ノードの最小サンプル数チェック
        if np.sum(left_indices_bool) < self.min_samples_leaf or \
           np.sum(right_indices_bool) < self.min_samples_leaf:
            leaf_value = self._leaf_value_func(y_data)
            return Node(value=leaf_value, impurity=current_impurity, num_samples=n_samples)

        left_node = self._build_tree(
            X_data[left_indices_bool],
            y_data[left_indices_bool],
            depth + 1
        )
        right_node = self._build_tree(
            X_data[right_indices_bool],
            y_data[right_indices_bool],
            depth + 1
        )

        node_params = {
            'feature_index': best_split_info['feature_index'],
            'feature_type': best_split_info['type'],
            'left_child': left_node,
            'right_child': right_node,
            'impurity': current_impurity, # 分割前の不純度
            'num_samples': n_samples
        }
        if best_split_info['type'] == 'numeric':
            node_params['threshold'] = best_split_info['threshold']
        else:
            node_params['category'] = best_split_info['category']
            
        return Node(**node_params)

    def fit(self, X_train, y_train, feature_types_list):
        self.feature_types = feature_types_list
        self.root = self._build_tree(X_train, y_train)

    def _traverse_tree(self, x_sample_row, node):
        if node.is_leaf_node():
            return node.value
        feature_val = x_sample_row[node.feature_index]
        if node.feature_type == 'numeric':
            if feature_val <= node.threshold:
                return self._traverse_tree(x_sample_row, node.left_child)
            else:
                return self._traverse_tree(x_sample_row, node.right_child)
        elif node.feature_type == 'categorical':
            if feature_val == node.category:
                return self._traverse_tree(x_sample_row, node.left_child)
            else:
                return self._traverse_tree(x_sample_row, node.right_child)
        else:
            raise ValueError(f"Unknown feature type '{node.feature_type}' in node.")

    def predict_single_tree(self, X_test):
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        predictions = []
        for i in range(X_test.shape[0]):
            row_sample = X_test[i, :]
            predictions.append(self._traverse_tree(row_sample, self.root))
        return np.array(predictions)


class DecisionTreeClassification(DecisionTreeBase):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        # 分類用の不純度関数と葉の値計算関数を渡す
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features,
                         criterion_func=_calculate_gini, 
                         leaf_value_func=self._majority_vote_leaf)

    def _majority_vote_leaf(self, y_data):
        """分類木の葉ノードの値を決定するための多数決"""
        if len(y_data) == 0: return None
        counts = Counter(y_data)
        return counts.most_common(1)[0][0]

class DecisionTreeRegression(DecisionTreeBase):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        # 回帰用の不純度関数 (MSE) と葉の値計算関数 (平均値) を渡す
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features,
                         criterion_func=_calculate_mse, 
                         leaf_value_func=self._mean_value_leaf)

    def _mean_value_leaf(self, y_data):
        """回帰木の葉ノードの値を決定するための平均値"""
        if len(y_data) == 0: return None
        return np.mean(y_data)


# --- 表示用関数 (オプション) ---
def print_tree(node, depth=0, feature_names=None, class_names_map=None, 
               category_names_map=None, indent_char="  "):
    if node is None:
        print(f"{indent_char * depth}Empty Node")
        return
            
    indent = indent_char * depth
    if node.is_leaf_node():
        # valueがクラスラベルか数値かで表示を少し変えても良い
        value_str = f"{node.value:.4f}" if isinstance(node.value, float) else str(node.value)
        if class_names_map and not isinstance(node.value, float) and node.value in class_names_map :
            value_str = class_names_map[node.value]
        
        print(f"{indent}Predict: {value_str} (Impurity: {node.impurity:.3f}, Samples: {node.num_samples})")
        return

    feature_name = f"Feature_{node.feature_index}"
    if feature_names and node.feature_index < len(feature_names):
        feature_name = feature_names[node.feature_index]

    if node.feature_type == 'numeric':
        condition = f"{feature_name} <= {node.threshold:.2f}"
        condition_else = f"{feature_name} > {node.threshold:.2f}"
    else: 
        cat_code = node.category
        cat_display_name = str(cat_code)
        if category_names_map and node.feature_index in category_names_map and \
           cat_code in category_names_map[node.feature_index]:
            cat_display_name = category_names_map[node.feature_index][cat_code]
        condition = f"{feature_name} == {cat_display_name}"
        condition_else = f"{feature_name} != {cat_display_name}"
    
    # 不純度の種類も表示すると良いかもしれない (Gini or MSE)
    print(f"{indent}If {condition} (Impurity: {node.impurity:.3f}, Samples: {node.num_samples}):")
    print_tree(node.left_child, depth + 1, feature_names, class_names_map, category_names_map, indent_char)
    print(f"{indent}Else ({condition_else}):")
    print_tree(node.right_child, depth + 1, feature_names, class_names_map, category_names_map, indent_char)