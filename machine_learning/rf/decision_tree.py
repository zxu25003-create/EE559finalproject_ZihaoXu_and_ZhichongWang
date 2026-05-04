# decision_tree.py

import numpy as np


class TreeNode:
    """
    决策树中的一个节点
    """
    def __init__(self, feature_index=None, threshold=None,
                 left=None, right=None, value=None):
        # 用于内部节点
        self.feature_index = feature_index   # 当前按哪个特征分裂
        self.threshold = threshold           # 分裂阈值
        self.left = left                     # 左子树
        self.right = right                   # 右子树

        # 用于叶节点
        self.value = value                   # 叶节点类别（0 或 1）


class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        """
        参数:
            max_depth: 树的最大深度
            min_samples_split: 一个节点至少有多少样本才允许继续分裂
            max_features: 每次分裂时考虑的特征数
                          None 表示使用全部特征
                          随机森林时这里会用到
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        """
        训练决策树
        """
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        对多个样本进行预测
        """
        predictions = [self._predict_one(x, self.root) for x in X]
        return np.array(predictions, dtype=np.int32)

    def _predict_one(self, x, node):
        """
        预测单个样本
        """
        # 如果到达叶节点，直接返回类别
        if node.value is not None:
            return node.value

        # 根据当前节点的分裂规则走左还是右
        if x[node.feature_index] < node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def _build_tree(self, X, y, depth):
        """
        递归建树
        """
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # ---------- 停止条件 ----------
        # 1. 到达最大深度
        # 2. 当前节点样本数太少，不能再分裂
        # 3. 当前节点已经纯了（只有一个类别）
        if (depth >= self.max_depth or
            num_samples < self.min_samples_split or
            num_labels == 1):
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value)

        # 选择当前节点可用的特征
        feature_indices = self._get_feature_indices(num_features)

        # 找最优分裂
        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        # 如果找不到有效分裂，直接变成叶节点
        if best_feature is None:
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value)

        # 按最优分裂划分数据
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # 再保险：防止分裂后某一边为空
        if len(y_left) == 0 or len(y_right) == 0:
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value)

        # 递归构建左右子树
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def _get_feature_indices(self, num_features):
        """
        决定当前节点要考虑哪些特征
        """
        if self.max_features is None or self.max_features >= num_features:
            return np.arange(num_features)

        return np.random.choice(
            num_features,
            self.max_features,
            replace=False
        )

    def _best_split(self, X, y, feature_indices):
        """
        在给定特征集合中寻找最优分裂
        """
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]

            # 候选阈值：直接使用当前特征出现过的唯一值
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_indices = feature_values < threshold
                right_indices = feature_values >= threshold

                # 跳过无效分裂
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                y_left = y[left_indices]
                y_right = y[right_indices]

                gini = self._weighted_gini(y_left, y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini(self, y):
        """
        计算一个节点的 Gini impurity
        Gini = 1 - sum(p_k^2)
        """
        num_samples = len(y)

        if num_samples == 0:
            return 0.0

        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / num_samples

        gini = 1.0 - np.sum(probabilities ** 2)
        return gini

    def _weighted_gini(self, y_left, y_right):
        """
        计算一次分裂后的加权 Gini
        """
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)

        weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        return weighted_gini

    def _majority_class(self, y):
        """
        返回当前节点中样本数最多的类别
        """
        classes, counts = np.unique(y, return_counts=True)
        majority_index = np.argmax(counts)
        return classes[majority_index]


if __name__ == "__main__":
    # 一个非常小的测试样例
    X = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.8, 0.7],
        [0.9, 0.6]
    ], dtype=np.float32)

    y = np.array([0, 0, 1, 1], dtype=np.int32)

    tree = DecisionTreeClassifierScratch(max_depth=3)
    tree.fit(X, y)
    preds = tree.predict(X)

    print("Predictions:", preds)
    print("Ground truth:", y)