"""
Application Streamlit pour la Comparaison d'Algorithmes de Clustering
=====================================================================
Auteur: Expert en Data Mining
Version: 1.1
Python: 3.10+

Dépendances requises:
pip install streamlit scikit-learn pandas plotly seaborn scipy openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn import datasets
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler

# === SUPERVISED LEARNING ADDITION ===
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.tree import plot_tree

# Configuration de la page
st.set_page_config(
    page_title="Comparaison Clustering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de session_state
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if 'scaling_applied' not in st.session_state:
    st.session_state.scaling_applied = False
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False

if 'unscaled_data' not in st.session_state:
    st.session_state.unscaled_data = None

if 'data_before_scaling' not in st.session_state:
    st.session_state.data_before_scaling = None

# === SUPERVISED LEARNING ADDITION ===
# Hold supervised-specific session data
if 'task_type' not in st.session_state:
    st.session_state.task_type = "Clustering (Unsupervised)"
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'X_data' not in st.session_state:
    st.session_state.X_data = None
if 'y_data' not in st.session_state:
    st.session_state.y_data = None

# ============================================================================
# FONCTIONS UTILITAIRES - CLUSTERING EXISTANT
# ============================================================================

class KMedoids:
    """K-Medoids clustering algorithm"""
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoids = None
        self.labels_ = None
    
    def fit_predict(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Initialize medoids randomly
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[medoid_indices]
        
        # Compute distance matrix
        distances = squareform(pdist(X, metric='euclidean'))
        
        for iteration in range(self.max_iter):
            # Assign points to nearest medoid
            medoid_distances = distances[medoid_indices]
            self.labels_ = np.argmin(medoid_distances, axis=0)
            
            # Update medoids
            new_medoids = medoid_indices.copy()
            for k in range(self.n_clusters):
                cluster_mask = self.labels_ == k
                if cluster_mask.sum() > 0:
                    cluster_indices = np.where(cluster_mask)[0]
                    # Find point with minimum average distance to others in cluster
                    cluster_distances = distances[cluster_indices][:, cluster_indices]
                    avg_distances = cluster_distances.sum(axis=1)
                    new_medoid = cluster_indices[np.argmin(avg_distances)]
                    new_medoids[k] = new_medoid
            
            # Check for convergence
            if np.array_equal(new_medoids, medoid_indices):
                break
            
            medoid_indices = new_medoids
        
        self.medoid_indices = medoid_indices
        self.medoids = X[medoid_indices]
        return self.labels_

class KMeans:
    """K-Means clustering algorithm"""
    def __init__(self, n_clusters=3, init='k-means++', max_iter=300, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
    
    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        if self.init == 'random':
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices]
        else:  # k-means++
            centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
                probabilities = distances / distances.sum()
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()
                
                for idx, prob in enumerate(cumulative_probs):
                    if r < prob:
                        centroids.append(X[idx])
                        break
            
            centroids = np.array(centroids)
        
        return centroids
    
    def fit(self, X):
        self.cluster_centers_ = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)
            
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0) 
                                      for k in range(self.n_clusters)])
            
            if np.allclose(self.cluster_centers_, new_centroids):
                break
            
            self.cluster_centers_ = new_centroids
        
        self.inertia_ = sum([np.linalg.norm(X[i] - self.cluster_centers_[self.labels_[i]])**2 
                            for i in range(len(X))])
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
class DBSCAN:
    """DBSCAN clustering algorithm"""
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def _get_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)
        cluster_id = 0
        
        for point_idx in range(n_samples):
            if self.labels_[point_idx] != -1:
                continue
            
            neighbors = self._get_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                continue
            
            self.labels_[point_idx] = cluster_id
            
            seed_set = list(neighbors)
            i = 0
            while i < len(seed_set):
                current_point = seed_set[i]
                
                if self.labels_[current_point] == -1:
                    self.labels_[current_point] = cluster_id
                
                if self.labels_[current_point] == -1 or self.labels_[current_point] == cluster_id:
                    self.labels_[current_point] = cluster_id
                    current_neighbors = self._get_neighbors(X, current_point)
                    
                    if len(current_neighbors) >= self.min_samples:
                        for neighbor in current_neighbors:
                            if neighbor not in seed_set:
                                seed_set.append(neighbor)
                
                i += 1
            
            cluster_id += 1
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
class DIANA:
    """Divisive Analysis (DIANA) - top-down hierarchical clustering"""
    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
    
    def _compute_distance_matrix(self, X):
        return squareform(pdist(X, metric='euclidean'))
    
    def _find_splinter(self, cluster_indices, X, dist_matrix):
        if len(cluster_indices) <= 1:
            return None
        
        avg_distances = []
        for idx in cluster_indices:
            other_indices = [i for i in cluster_indices if i != idx]
            avg_dist = np.mean([dist_matrix[idx, j] for j in other_indices])
            avg_distances.append((avg_dist, idx))
        
        return max(avg_distances)[1]
    
    def _split_cluster(self, cluster_indices, X, dist_matrix):
        if len(cluster_indices) <= 1:
            return [cluster_indices], []
        
        splinter_group = [self._find_splinter(cluster_indices, X, dist_matrix)]
        old_group = [idx for idx in cluster_indices if idx not in splinter_group]
        
        changed = True
        while changed and len(old_group) > 0:
            changed = False
            for idx in old_group[:]:
                dist_to_splinter = np.mean([dist_matrix[idx, j] for j in splinter_group])
                other_old = [i for i in old_group if i != idx]
                if len(other_old) == 0:
                    break
                dist_to_old = np.mean([dist_matrix[idx, j] for j in other_old])
                
                if dist_to_splinter < dist_to_old:
                    splinter_group.append(idx)
                    old_group.remove(idx)
                    changed = True
        
        return old_group, splinter_group
    
    def fit(self, X):
        n_samples = X.shape[0]
        dist_matrix = self._compute_distance_matrix(X)
        
        clusters = [list(range(n_samples))]
        
        while len(clusters) < self.n_clusters:
            max_diameter = -1
            split_idx = 0
            
            for i, cluster in enumerate(clusters):
                if len(cluster) <= 1:
                    continue
                diameter = max([dist_matrix[p1, p2] for p1 in cluster for p2 in cluster])
                if diameter > max_diameter:
                    max_diameter = diameter
                    split_idx = i
            
            cluster_to_split = clusters[split_idx]
            group1, group2 = self._split_cluster(cluster_to_split, X, dist_matrix)
            
            clusters[split_idx] = group1
            if len(group2) > 0:
                clusters.append(group2)
        
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                self.labels_[idx] = cluster_id
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_    

class AgglomerativeClustering:
    """Agglomerative (AGNES) hierarchical clustering"""
    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
    
    def _compute_distance_matrix(self, X):
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def _cluster_distance(self, X, cluster1_indices, cluster2_indices, dist_matrix):
        if self.linkage == 'single':
            distances = [dist_matrix[i, j] for i in cluster1_indices for j in cluster2_indices]
            return min(distances)
        elif self.linkage == 'complete':
            distances = [dist_matrix[i, j] for i in cluster1_indices for j in cluster2_indices]
            return max(distances)
        elif self.linkage == 'average':
            distances = [dist_matrix[i, j] for i in cluster1_indices for j in cluster2_indices]
            return np.mean(distances)
        elif self.linkage == 'ward':
            cluster1_center = X[cluster1_indices].mean(axis=0)
            cluster2_center = X[cluster2_indices].mean(axis=0)
            n1 = len(cluster1_indices)
            n2 = len(cluster2_indices)
            return np.sqrt((n1 * n2) / (n1 + n2)) * np.linalg.norm(cluster1_center - cluster2_center)
    
    def fit(self, X):
        n_samples = X.shape[0]
        clusters = [[i] for i in range(n_samples)]
        dist_matrix = self._compute_distance_matrix(X)
        
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    dist = self._cluster_distance(X, clusters[i], clusters[j], dist_matrix)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                self.labels_[idx] = cluster_id
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

@st.cache_data
def load_preset_dataset(dataset_name):
    """Charge un dataset prédéfini depuis sklearn"""
    if dataset_name == "Iris":
        data = datasets.load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == "Wine":
        data = datasets.load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
    return df

def check_missing_values(df):
    """Vérifie les valeurs manquantes dans le DataFrame"""
    missing = df.isnull().sum()
    if missing.any():
        return True, missing[missing > 0]
    return False, None

# === SUPERVISED LEARNING ADDITION ===
# Complete implementations of supervised algorithms from scratch

class DecisionTree:
    """Decision Tree implementation from scratch supporting both C4.5 (entropy) and CART (gini)"""
    
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature      # Feature index for splitting
            self.threshold = threshold  # Threshold value for splitting
            self.left = left           # Left subtree
            self.right = right         # Right subtree
            self.value = value         # Class value for leaf nodes
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, random_state=42):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None
        np.random.seed(random_state)
    
    def _calculate_impurity(self, y):
        """Calculate impurity using specified criterion"""
        if len(y) == 0:
            return 0
        
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        
        if self.criterion == 'gini':
            # Gini impurity
            return 1 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            # Entropy
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")
    
    def _calculate_information_gain(self, y, y_left, y_right):
        """Calculate information gain for a split"""
        parent_impurity = self._calculate_impurity(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        weighted_impurity = (n_left / n) * self._calculate_impurity(y_left) + \
                           (n_right / n) * self._calculate_impurity(y_right)
        
        return parent_impurity - weighted_impurity
    
    def _find_best_split(self, X, y):
        """Find the best split for a node"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        n_samples = len(y)
        
        for feature in range(n_features):
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)
            
            # Try different thresholds
            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                if len(y_left) < 1 or len(y_right) < 1:
                    continue
                
                gain = self._calculate_information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            # Create leaf node
            unique, counts = np.unique(y, return_counts=True)
            return self.Node(value=unique[np.argmax(counts)])
        
        # Find best split
        feature, threshold, gain = self._find_best_split(X, y)
        
        if feature is None:
            # No good split found, create leaf
            unique, counts = np.unique(y, return_counts=True)
            return self.Node(value=unique[np.argmax(counts)])
        
        # Split the data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Recursively build subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return self.Node(feature=feature, threshold=threshold, 
                        left=left_child, right=right_child)
    
    def _calculate_feature_importances(self, node, n_features):
        """Calculate feature importances by traversing the tree"""
        importances = np.zeros(n_features)
        
        def traverse(node, n_samples_total):
            if node.value is not None:
                return
            
            # Calculate weighted importance
            left_mask = self.X_train[:, node.feature] <= node.threshold
            right_mask = ~left_mask
            
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            if n_left > 0 and n_right > 0:
                # Information gain weighted by proportion of samples
                gain = self._calculate_information_gain(
                    self.y_train, 
                    self.y_train[left_mask], 
                    self.y_train[right_mask]
                )
                
                importances[node.feature] += (n_samples_total / len(self.X_train)) * gain
            
            # Recurse
            traverse(node.left, n_left)
            traverse(node.right, n_right)
        
        traverse(self.root, len(self.X_train))
        return importances
    
    def fit(self, X, y):
        """Fit the decision tree"""
        self.X_train = X
        self.y_train = y
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances
        n_features = X.shape[1]
        self.feature_importances_ = self._calculate_feature_importances(self.root, n_features)
        
        return self
    
    def _predict_single(self, node, x):
        """Predict for a single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(node.left, x)
        else:
            return self._predict_single(node.right, x)
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        return np.array([self._predict_single(self.root, x) for x in X])
    
    def _predict_single_feature(self, feature_idx, X):
        """Helper for feature importance calculation"""
        return X[:, feature_idx]

class KNN:
    """K-Nearest Neighbors implementation from scratch"""
    
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _get_neighbors(self, x):
        """Get k nearest neighbors for a sample"""
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        return k_indices
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = []
        
        for x in X:
            k_indices = self._get_neighbors(x)
            k_labels = self.y_train[k_indices]
            
            if self.weights == 'uniform':
                # Majority vote
                unique, counts = np.unique(k_labels, return_counts=True)
                prediction = unique[np.argmax(counts)]
            else:  # 'distance'
                # Weighted by inverse distance
                distances = [self._euclidean_distance(x, self.X_train[i]) for i in k_indices]
                weights = 1 / (np.array(distances) + 1e-10)  # Avoid division by zero
                
                weighted_votes = {}
                for i, label in enumerate(k_labels):
                    if label not in weighted_votes:
                        weighted_votes[label] = 0
                    weighted_votes[label] += weights[i]
                
                prediction = max(weighted_votes, key=weighted_votes.get)
            
            predictions.append(prediction)
        
        return np.array(predictions)

class NaiveBayes:
    """Naive Bayes (Gaussian) implementation from scratch"""
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}
    
    def fit(self, X, y):
        """Fit the Naive Bayes model"""
        self.classes = np.unique(y)
        
        for cls in self.classes:
            X_cls = X[y == cls]
            
            # Calculate prior probability
            self.class_priors[cls] = len(X_cls) / len(X)
            
            # Calculate mean and variance for each feature
            self.class_means[cls] = np.mean(X_cls, axis=0)
            self.class_variances[cls] = np.var(X_cls, axis=0)
        
        return self
    
    def _gaussian_probability(self, x, mean, var):
        """Calculate Gaussian probability density"""
        eps = 1e-10  # Avoid division by zero
        coeff = 1 / np.sqrt(2 * np.pi * (var + eps))
        exponent = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        return coeff * exponent
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = []
        
        for x in X:
            posteriors = {}
            
            for cls in self.classes:
                # Calculate posterior = prior * product of likelihoods
                posterior = np.log(self.class_priors[cls])  # Use log to avoid underflow
                
                for i, feature_val in enumerate(x):
                    likelihood = self._gaussian_probability(
                        feature_val, 
                        self.class_means[cls][i], 
                        self.class_variances[cls][i]
                    )
                    posterior += np.log(likelihood + 1e-10)
                
                posteriors[cls] = posterior
            
            # Choose class with highest posterior
            prediction = max(posteriors, key=posteriors.get)
            predictions.append(prediction)
        
        return np.array(predictions)

class SVM:
    """Simple SVM implementation from scratch using SMO algorithm (binary classification)"""
    
    def __init__(self, C=1.0, gamma='scale', max_iter=1000, random_state=42):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.kernel = None
        np.random.seed(random_state)
    
    def _rbf_kernel(self, x1, x2):
        """RBF kernel function"""
        if self.gamma == 'scale':
            gamma_val = 1 / (x1.shape[1] * np.var(self.X_train))
        else:
            gamma_val = self.gamma
        
        return np.exp(-gamma_val * np.sum((x1 - x2) ** 2))
    
    def _kernel(self, x1, x2):
        """Kernel function"""
        if self.kernel_type == 'rbf':
            return self._rbf_kernel(x1, x2)
        else:
            # Linear kernel
            return np.dot(x1, x2)
    
    def fit(self, X, y):
        """Fit the SVM model using simplified SMO"""
        self.X_train = X
        # Convert to binary classification (-1, 1)
        unique_classes = np.unique(y)
        if len(unique_classes) > 2:
            raise ValueError("SVM implementation currently supports binary classification only")
        
        self.y_train = np.where(y == unique_classes[0], -1, 1)
        self.classes_ = unique_classes
        self.kernel_type = 'rbf' if self.gamma != 'linear' else 'linear'
        
        n_samples = len(X)
        self.alpha = np.zeros(n_samples)
        
        # Simplified SMO implementation
        for _ in range(self.max_iter):
            alpha_changed = 0
            
            for i in range(n_samples):
                # Calculate prediction for i
                prediction_i = self.b
                for j in range(n_samples):
                    prediction_i += self.alpha[j] * self.y_train[j] * self._kernel(X[i], X[j])
                
                error_i = prediction_i - self.y_train[i]
                
                if (self.y_train[i] * error_i < -0.001 and self.alpha[i] < self.C) or \
                   (self.y_train[i] * error_i > 0.001 and self.alpha[i] > 0):
                    
                    # Select random j != i
                    j = np.random.randint(n_samples)
                    while j == i:
                        j = np.random.randint(n_samples)
                    
                    # Calculate prediction for j
                    prediction_j = self.b
                    for k in range(n_samples):
                        prediction_j += self.alpha[k] * self.y_train[k] * self._kernel(X[j], X[k])
                    
                    error_j = prediction_j - self.y_train[j]
                    
                    # Calculate bounds
                    if self.y_train[i] != self.y_train[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Calculate eta
                    eta = 2 * self._kernel(X[i], X[j]) - self._kernel(X[i], X[i]) - self._kernel(X[j], X[j])
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    alpha_j_old = self.alpha[j]
                    self.alpha[j] -= self.y_train[j] * (error_i - error_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] += self.y_train[i] * self.y_train[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update b
                    b1 = self.b - error_i - self.y_train[i] * (self.alpha[i] - alpha_j_old) * self._kernel(X[i], X[i]) - \
                         self.y_train[j] * (self.alpha[j] - alpha_j_old) * self._kernel(X[i], X[j])
                    b2 = self.b - error_j - self.y_train[i] * (self.alpha[i] - alpha_j_old) * self._kernel(X[i], X[j]) - \
                         self.y_train[j] * (self.alpha[j] - alpha_j_old) * self._kernel(X[j], X[j])
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    alpha_changed += 1
            
            if alpha_changed == 0:
                break
        
        return self
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = []
        
        for x in X:
            prediction = self.b
            for i in range(len(self.X_train)):
                prediction += self.alpha[i] * self.y_train[i] * self._kernel(x, self.X_train[i])
            
            # Convert back to original class labels
            class_idx = 0 if prediction < 0 else 1
            predictions.append(self.classes_[class_idx])
        
        return np.array(predictions)

# Wrapper classes for supervised algorithms with fit_predict(X, y) and .labels_/.predictions_

class KNNWrapper:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNN(n_neighbors=n_neighbors, weights=weights)
        self.labels_ = None
        self.predictions_ = None
    def fit_predict(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.labels_ = preds
        self.predictions_ = preds
        return preds

class GaussianNBWrapper:
    def __init__(self):
        self.model = NaiveBayes()
        self.labels_ = None
        self.predictions_ = None
    def fit_predict(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.labels_ = preds
        self.predictions_ = preds
        return preds

class C45Wrapper:
    # DecisionTreeClassifier with entropy (C4.5-like)
    def __init__(self, max_depth=None, min_samples_split=2, random_state=42):
        self.model = DecisionTree(
            criterion='entropy',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.labels_ = None
        self.predictions_ = None
    def fit_predict(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.labels_ = preds
        self.predictions_ = preds
        return preds

class CARTWrapper:
    # DecisionTreeClassifier with gini
    def __init__(self, max_depth=None, min_samples_split=2, random_state=42):
        self.model = DecisionTree(
            criterion='gini',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.labels_ = None
        self.predictions_ = None
    def fit_predict(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.labels_ = preds
        self.predictions_ = preds
        return preds

class SVMWrapper:
    # SVM RBF
    def __init__(self, C=1.0, gamma='scale', random_state=42):
        # random_state not used in SVC, kept for signature consistency
        self.model = SVM(C=C, gamma=gamma, random_state=random_state)
        self.labels_ = None
        self.predictions_ = None
    def fit_predict(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.labels_ = preds
        self.predictions_ = preds
        return preds

# === END SUPERVISED LEARNING ADDITION ===

# === MODIFICATION: run_clustering → run_model (supports clustering and classification) ===
def run_model(task_type, algo_name, params, data, y=None):
    """
    Exécute l'algorithme choisi selon le type de tâche.
    Args:
        task_type: "Clustering (Unsupervised)" ou "Classification (Supervised)"
        algo_name: Nom de l'algorithme
        params: Dictionnaire des paramètres
        data: DataFrame features
        y: Série target (pour supervision)
    Returns:
        predictions_or_labels, model, y_true (None pour clustering)
    """
    X = data.values

    if task_type == "Clustering (Unsupervised)":
        if algo_name == "K-Means":
            model = KMeans(
                n_clusters=params['n_clusters'],
                init=params['init'],
                random_state=42,
                n_init=10
            )
            labels = model.fit_predict(X)
        elif algo_name == "DBSCAN":
            model = DBSCAN(
                eps=params['eps'],
                min_samples=params['min_samples']
            )
            labels = model.fit_predict(X)
        elif algo_name == "K-Medoids":
            model = KMedoids(
                n_clusters=params['n_clusters'],
                max_iter=100,
                random_state=42
            )
            labels = model.fit_predict(X)
        elif algo_name == "AGNES":
            model = AgglomerativeClustering(
                n_clusters=params['n_clusters'],
                linkage=params['linkage']
            )
            labels = model.fit_predict(X)
        elif algo_name == "DIANA":
            model = DIANA(
                n_clusters=params['n_clusters'],
                linkage=params['linkage']
            )
            labels = model.fit_predict(X)
            # Compute linkage matrix for dendrogram
            Z = linkage(X, method=params['linkage'])
            model.linkage_matrix = Z
        return labels, model, None

    else:  # Classification (Supervised)
        if algo_name == "KNN":
            model = KNNWrapper(n_neighbors=params['n_neighbors'], weights=params['weights'])
        elif algo_name == "Naive Bayes (Gaussian)":
            model = GaussianNBWrapper()
        elif algo_name == "C4.5 (Entropy)":
            model = C45Wrapper(max_depth=params.get('max_depth'), min_samples_split=params.get('min_samples_split', 2), random_state=42)
        elif algo_name == "CART (Gini)":
            model = CARTWrapper(max_depth=params.get('max_depth'), min_samples_split=params.get('min_samples_split', 2), random_state=42)
        elif algo_name == "SVM (RBF)":
            model = SVMWrapper(C=params.get('C', 1.0), gamma=params.get('gamma', 'scale'), random_state=42)
        else:
            raise ValueError("Algorithme de classification inconnu.")

        preds = model.fit_predict(X, y.values if isinstance(y, pd.Series) else y)
        return preds, model, y

def calculate_metrics(X, labels):
    """Calcule les métriques de qualité du clustering"""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {
            'Silhouette Score': None,
            'Calinski-Harabasz': None,
            'Davies-Bouldin': None,
            'Note': 'Métriques non calculables (< 2 clusters)'
        }
    try:
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        if len(set(labels_filtered)) < 2 or len(labels_filtered) < 2:
            return {
                'Silhouette Score': None,
                'Calinski-Harabasz': None,
                'Davies-Bouldin': None,
                'Note': 'Pas assez de points valides'
            }
        silhouette = silhouette_score(X_filtered, labels_filtered)
        calinski = calinski_harabasz_score(X_filtered, labels_filtered)
        davies = davies_bouldin_score(X_filtered, labels_filtered)
        return {
            'Silhouette Score': round(silhouette, 4),
            'Calinski-Harabasz': round(calinski, 2),
            'Davies-Bouldin': round(davies, 4)
        }
    except Exception as e:
        return {
            'Silhouette Score': None,
            'Calinski-Harabasz': None,
            'Davies-Bouldin': None,
            'Note': f'Erreur: {str(e)}'
        }

# === SUPERVISED LEARNING ADDITION ===
def calculate_classification_metrics(y_true, y_pred):
    """Calcule les métriques de classification"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'Accuracy': round(acc, 4),
        'Precision (macro)': round(precision, 4),
        'Recall (macro)': round(recall, 4),
        'F1-score (macro)': round(f1, 4),
        'Confusion Matrix': cm
    }

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("📊 Comparaison d'Algorithmes de Clustering")
# if st.session_state.current_data is not None:
#     st.write("🔍 **Debug Info:**")
#     st.write(f"Scaling applied: {st.session_state.get('scaling_applied', False)}")
#     st.write(f"Preprocessing done: {st.session_state.get('preprocessing_done', False)}")
#     st.write(f"Data shape: {st.session_state.current_data.shape}")
#     st.write(f"Sample values from first numeric column:")
#     numeric_cols_dbg = st.session_state.current_data.select_dtypes(include=[np.number]).columns
#     if len(numeric_cols_dbg) > 0:
#         st.write(st.session_state.current_data[numeric_cols_dbg[0]].head())
# st.markdown("### Application pédagogique pour l'analyse et la visualisation de clustering")

# ============================================================================
# SIDEBAR - CHARGEMENT DES DONNÉES
# ============================================================================

st.sidebar.header("🔧 Configuration")

st.sidebar.subheader("1️⃣ Chargement des données")
data_source = st.sidebar.radio(
    "Source des données:",
    ["Ensembles prédéfinis", "Import personnalisé"]
)

df = None
selected_features = []

if data_source == "Ensembles prédéfinis":
    dataset_name = st.sidebar.selectbox(
        "Choisir un dataset:",
        ["Iris", "Wine", "Breast Cancer"]
    )
    
    if st.sidebar.button("📥 Charger le dataset"):
        with st.spinner("Chargement en cours..."):
            df = load_preset_dataset(dataset_name)
            st.session_state.current_data = df
            st.session_state.data_before_scaling = df.copy()
            st.sidebar.success(f"✅ Dataset {dataset_name} chargé ({len(df)} lignes)")

else:  # Import personnalisé
    uploaded_file = st.sidebar.file_uploader(
        "Uploader un fichier CSV/Excel:",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.current_data = df.copy()
            st.session_state.data_before_scaling = df.copy()
            st.session_state.unscaled_data = df.copy()
            st.session_state.scaling_applied = False
            st.session_state.preprocessing_done = False
            
            st.sidebar.success(f"✅ Fichier chargé ({len(df)} lignes × {len(df.columns)} colonnes)")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if len(non_numeric_cols) > 0:
                st.sidebar.info(f"ℹ️ {len(non_numeric_cols)} colonnes non-numériques détectées")
            
            has_missing, missing_cols = check_missing_values(df)
            if has_missing:
                st.sidebar.warning(f"⚠️ Valeurs manquantes dans {len(missing_cols)} colonnes")
                
        except Exception as e:
            st.sidebar.error(f"⚠️ Erreur de chargement: {str(e)}")

# ============================================================================
# SIDEBAR - DATA PREPROCESSING (FIXED VERSION)
# ============================================================================
if st.session_state.current_data is not None:
    df = st.session_state.current_data.copy()
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧰 Prétraitement des Données")

    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = False

    has_missing, missing_cols = check_missing_values(df)
    if has_missing:
        st.sidebar.warning(f"⚠️ Valeurs manquantes détectées")
        with st.sidebar.expander("📋 Détails des valeurs manquantes"):
            for col, count in missing_cols.items():
                st.write(f"- **{col}**: {count} valeurs manquantes")
        
        missing_action = st.sidebar.selectbox(
            "Gérer les valeurs manquantes:",
            ["Ne rien faire", "Supprimer les lignes", "Remplir par la moyenne", "Remplir par la médiane", "Remplir par 0"]
        )
    else:
        missing_action = "Ne rien faire"
        st.sidebar.success("✅ Aucune valeur manquante")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) > 0:
        st.sidebar.warning(f"⚠️ {len(catégoriales := categorical_cols)} colonnes catégorielles détectées")
        with st.sidebar.expander("📋 Colonnes catégorielles"):
            for col in categorical_cols:
                unique_count = df[col].nunique()
                st.write(f"- **{col}**: {unique_count} valeurs uniques")
        
        encoding_method = st.sidebar.selectbox(
            "Encodage des variables catégorielles:",
            ["Ne rien faire", "One-Hot Encoding", "Label Encoding", "Supprimer ces colonnes"]
        )
    else:
        encoding_method = "Ne rien faire"
        st.sidebar.success("✅ Aucune variable catégorielle")

    st.sidebar.markdown("**Normalisation des features:**")
    scaling_method = st.sidebar.selectbox(
        "Méthode de normalisation:",
        ["Aucune", "StandardScaler (Z-score)", "MinMaxScaler (0-1)"]
    )

    if st.sidebar.button("⚙️ Appliquer le prétraitement", type="primary"):
        with st.spinner("Application du prétraitement..."):
            df_processed = st.session_state.data_before_scaling.copy()
            
            if missing_action != "Ne rien faire":
                if missing_action == "Supprimer les lignes":
                    df_processed = df_processed.dropna()
                    st.sidebar.success(f"✅ Lignes avec valeurs manquantes supprimées")
                else:
                    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                    if missing_action == "Remplir par la moyenne":
                        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
                    elif missing_action == "Remplir par la médiane":
                        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
                    elif missing_action == "Remplir par 0":
                        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
                    st.sidebar.success(f"✅ Valeurs manquantes traitées")
            
            if encoding_method != "Ne rien faire":
                categorical_cols_current = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
                if len(categorical_cols_current) > 0:
                    if encoding_method == "Supprimer ces colonnes":
                        df_processed = df_processed.drop(columns=categorical_cols_current)
                        st.sidebar.success(f"✅ {len(categorical_cols_current)} colonnes catégorielles supprimées")
                    elif encoding_method == "One-Hot Encoding":
                        df_processed = pd.get_dummies(df_processed, columns=categorical_cols_current, drop_first=True)
                        st.sidebar.success(f"✅ One-Hot Encoding appliqué")
                    elif encoding_method == "Label Encoding":
                        le = LabelEncoder()
                        for col in categorical_cols_current:
                            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        st.sidebar.success(f"✅ Label Encoding appliqué")
            
            if scaling_method != "Aucune":
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                if 'target' in numeric_cols:
                    numeric_cols.remove('target')
                if len(numeric_cols) > 0:
                    if scaling_method == "StandardScaler (Z-score)":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                    st.sidebar.success(f"✅ {scaling_method} appliqué sur {len(numeric_cols)} colonnes")
                    st.session_state.scaling_applied = True
                else:
                    st.sidebar.warning("⚠️ Aucune colonne numérique à normaliser")
                    st.session_state.scaling_applied = False
            else:
                st.session_state.scaling_applied = False
            
            st.session_state.current_data = df_processed
            st.session_state.preprocessing_done = True
            st.sidebar.success("✅ Prétraitement terminé!")
            st.sidebar.info(f"📊 Dataset final: {len(df_processed)} lignes × {len(df_processed.columns)} colonnes")
            st.cache_data.clear()
            st.rerun()

# ============================================================================
# SIDEBAR - DATA PREVIEW AND FEATURE SELECTION (FIXED) + TASK TYPE TOGGLE
# ============================================================================
if st.session_state.current_data is not None:
    df = st.session_state.current_data.copy()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Aperçu des données")
    
    if st.session_state.get('preprocessing_done', False):
        st.sidebar.success("✅ Données prétraitées")
    if st.session_state.get('scaling_applied', False):
        st.sidebar.info("📏 Normalisation appliquée")
    st.sidebar.caption(f"📊 {len(df)} lignes × {len(df.columns)} colonnes")
    
    preview_key = f"preview_{st.session_state.get('preprocessing_done', False)}_{st.session_state.get('scaling_applied', False)}"
    st.sidebar.dataframe(df.head(10), height=200, use_container_width=True, key=preview_key)
    
    if st.sidebar.button("📈 Voir les statistiques"):
        st.sidebar.dataframe(df.describe(), use_container_width=True)

    # === SUPERVISED LEARNING ADDITION ===
    # Step 1: Task type toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Type de tâche")
    task_type = st.sidebar.radio("Task Type", ["Clustering (Unsupervised)", "Classification (Supervised)"])
    st.session_state.task_type = task_type

    # Feature selection (always available; for classification these are X)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # We'll not auto-remove 'target' here; we filter later based on selected target
    if len(numeric_cols) >= 2:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Sélection des Features")
        selected_features = st.sidebar.multiselect(
            "Sélectionner les features (min. 2):",
            [col for col in numeric_cols],  # keep numeric
            default=[col for col in numeric_cols if col != 'target'][:min(4, len(numeric_cols))],
            help="Choisissez au moins 2 features numériques"
        )
    else:
        selected_features = []
        st.sidebar.error("❌ Au moins 2 features numériques requises")
        st.sidebar.info("💡 Appliquez le prétraitement pour encoder les variables catégorielles")

    # === SUPERVISED LEARNING ADDITION ===
    # Step 3: In supervised mode, target selection and X/y preparation
    target_column = None
    if task_type == "Classification (Supervised)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🏷️ Colonne cible (label)")
        # Prefer existing 'target' if present; else let user pick any column
        available_target_cols = df.columns.tolist()
        # Help user by placing 'target' first if exists
        default_target_index = available_target_cols.index('target') if 'target' in available_target_cols else 0
        target_column = st.sidebar.selectbox("Sélectionner la colonne cible:", available_target_cols, index=default_target_index)
        st.session_state.target_column = target_column

        # Ensure y is label-encoded if categorical
        y_series = df[target_column].copy()
        if y_series.dtype == 'object' or str(y_series.dtype) == 'category':
            y_le = LabelEncoder().fit_transform(y_series.astype(str))
        else:
            # If numeric but not integer classes, try casting
            y_le = y_series.values
        # Prepare X by excluding the target column from selected features if present
        selected_features_cls = [f for f in selected_features if f != target_column]
        if len(selected_features_cls) < 1:
            st.sidebar.warning("⚠️ Sélectionnez des features (excluez la colonne cible des features).")
        X_cls_df = df[selected_features_cls].copy()
        st.session_state.X_data = X_cls_df
        st.session_state.y_data = pd.Series(y_le, name=target_column)

# ============================================================================
# SIDEBAR - PARAMÉTRAGE DES ALGORITHMES
# ============================================================================
if len(selected_features) >= 2:
    if st.session_state.scaling_applied:
        st.sidebar.markdown("📌 **Données** : ✅ Normalisées")
    else:
        st.sidebar.markdown("📌 **Données** : ⚠️ Non normalisées (seront normalisées avant clustering)")

if len(selected_features) >= 2 and st.session_state.current_data is not None:
    st.sidebar.markdown("---")
    # === SUPERVISED LEARNING ADDITION ===
    # Switch configuration based on task type
    if st.session_state.task_type == "Clustering (Unsupervised)":
        st.sidebar.subheader("2️⃣ Configuration du clustering")
        algo_name = st.sidebar.selectbox(
            "Choisir l'algorithme:",
            ["K-Means", "DBSCAN", "K-Medoids", "AGNES", "DIANA"]
        )
        params = {}
        if algo_name == "K-Means":
            params['n_clusters'] = st.sidebar.slider("Nombre de clusters:", 2, 10, 3)
            params['init'] = st.sidebar.selectbox("Méthode d'initialisation:", ["k-means++", "random"])
        elif algo_name == "DBSCAN":
            params['eps'] = st.sidebar.slider("Epsilon (eps):", 0.1, 5.0, 0.5, 0.1)
            params['min_samples'] = st.sidebar.slider("Min samples:", 1, 20, 5)
        elif algo_name == "K-Medoids":
            params['n_clusters'] = st.sidebar.slider("Nombre de clusters:", 2, 10, 3)
        elif algo_name == "AGNES":
            params['n_clusters'] = st.sidebar.slider("Nombre de clusters:", 2, 10, 3)
            params['linkage'] = st.sidebar.selectbox("Méthode de liaison:", ["ward", "complete", "average", "single"])
        elif algo_name == "DIANA":
            params['n_clusters'] = st.sidebar.slider("Nombre de clusters:", 2, 10, 3)
            params['linkage'] = st.sidebar.selectbox("Méthode de liaison:", ["ward", "complete", "average", "single"])

        if st.sidebar.button("🚀 Exécuter le clustering", type="primary"):
            with st.spinner("Clustering en cours..."):
                df_run = st.session_state.current_data
                X_df = df_run[selected_features].copy()
                labels, model, _ = run_model("Clustering (Unsupervised)", algo_name, params, X_df, None)
                X_scaled = X_df.values
                metrics = calculate_metrics(X_scaled, labels)
                result = {
                    'task_type': "Clustering (Unsupervised)",
                    'algorithm': algo_name,
                    'params': params.copy(),
                    'labels': labels,
                    'model': model,
                    'metrics': metrics,
                    'features': selected_features,
                    'X_scaled': X_df,
                    'X_original': X_df
                }
                st.session_state.results_history.append(result)
                st.sidebar.success("✅ Clustering terminé!")
    else:
        st.sidebar.subheader("2️⃣ Configuration de la classification")
        algo_name = st.sidebar.selectbox(
            "Choisir l'algorithme:",
            ["KNN", "Naive Bayes (Gaussian)", "C4.5 (Entropy)", "CART (Gini)", "SVM (RBF)"]
        )
        params = {}
        if algo_name == "KNN":
            params['n_neighbors'] = st.sidebar.slider("Nombre de voisins (k):", 1, 25, 5)
            params['weights'] = st.sidebar.selectbox("Pondération:", ["uniform", "distance"])
        elif algo_name == "Naive Bayes (Gaussian)":
            pass  # Aucun paramètre nécessaire
        elif algo_name in ["C4.5 (Entropy)", "CART (Gini)"]:
            use_max_depth = st.sidebar.checkbox("Limiter la profondeur", value=False)
            if use_max_depth:
                params['max_depth'] = st.sidebar.slider("Profondeur max:", 1, 50, 10)
            else:
                params['max_depth'] = None
            params['min_samples_split'] = st.sidebar.slider("Min samples split:", 2, 20, 2)
        elif algo_name == "SVM (RBF)":
            params['C'] = st.sidebar.slider("C (Regularization):", 0.1, 10.0, 1.0, 0.1)
            params['gamma'] = st.sidebar.selectbox("Gamma:", ["scale", "auto"])

        # Ensure target available
        if st.session_state.target_column is None or st.session_state.X_data is None or st.session_state.y_data is None:
            st.sidebar.error("❌ Sélectionnez une colonne cible et des features valides.")
        else:
            if st.sidebar.button("🚀 Exécuter la classification", type="primary"):
                with st.spinner("Classification en cours..."):
                    X_df = st.session_state.X_data.copy()
                    y_series = st.session_state.y_data.copy()
                    y_pred, model, y_true = run_model("Classification (Supervised)", algo_name, params, X_df, y_series)
                    metrics_cls = calculate_classification_metrics(y_true.values if isinstance(y_true, pd.Series) else y_true, y_pred)
                    result = {
                        'task_type': "Classification (Supervised)",
                        'algorithm': algo_name,
                        'params': params.copy(),
                        'labels': y_pred,
                        'model': model,
                        'metrics': metrics_cls,
                        'features': X_df.columns.tolist(),
                        'X_scaled': X_df,
                        'X_original': X_df,
                        'y_true': y_true.values if isinstance(y_true, pd.Series) else y_true,
                        'target_name': st.session_state.target_column
                    }
                    st.session_state.results_history.append(result)
                    st.sidebar.success("✅ Classification terminée!")

# ============================================================================
# ZONE CENTRALE - AFFICHAGE DES RÉSULTATS
# ============================================================================
if len(st.session_state.results_history) > 0:
    latest_result = st.session_state.results_history[-1]
    task_latest = latest_result.get('task_type', "Clustering (Unsupervised)")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Visualisation 2D/3D",
        "📊 Métriques",
        "📉 Graphiques complémentaires",
        "📄 Comparaisons"
    ])
    
    # ========================================================================
    # TAB 1: VISUALISATION 2D/3D
    # ========================================================================
    with tab1:
        if task_latest == "Classification (Supervised)":
            st.subheader("Visualisation des Classes (Vrai vs Prédit)")
            col1, col2, col3 = st.columns(3)
            features = latest_result['features']
            X_orig = latest_result['X_original']
            y_true = latest_result['y_true']
            y_pred = latest_result['labels']
            
            with col1:
                x_axis = st.selectbox("Axe X:", features, index=0, key="x_axis_tab1")
            with col2:
                y_axis = st.selectbox("Axe Y:", features, index=min(1, len(features)-1), key="y_axis_tab1")
            with col3:
                viz_type = st.radio("Type:", ["2D", "3D"], key="viz_type_tab1")
            
            viz_df = X_orig.copy()
            viz_df['True'] = y_true
            viz_df['Pred'] = y_pred
            view_mode = st.radio("Affichage:", ["Vérités", "Prédictions"], horizontal=True, key="view_mode_tab1")
            color_col = 'True' if view_mode == "Vérités" else 'Pred'
            
            if viz_type == "2D":
                fig = px.scatter(
                    viz_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_col,
                    title=f"{latest_result['algorithm']} - {view_mode} (Vue 2D)",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data=features
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                if len(features) >= 3:
                    z_axis = st.selectbox("Axe Z:", features, index=min(2, len(features)-1), key="z_axis_tab1")
                    fig = px.scatter_3d(
                        viz_df,
                        x=x_axis, y=y_axis, z=z_axis,
                        color=color_col,
                        title=f"{latest_result['algorithm']} - {view_mode} (Vue 3D)",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        hover_data=features
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Au moins 3 features sont nécessaires pour la visualisation 3D")
        
        else:  # Clustering
            st.subheader("Visualisation des Clusters")
            col1, col2, col3 = st.columns(3)
            features = latest_result['features']
            X_orig = latest_result['X_original']
            labels = latest_result['labels']
            
            with col1:
                x_axis = st.selectbox("Axe X:", features, index=0, key="x_axis_clust_tab1")
            with col2:
                y_axis = st.selectbox("Axe Y:", features, index=min(1, len(features)-1), key="y_axis_clust_tab1")
            with col3:
                viz_type = st.radio("Type:", ["2D", "3D"], key="viz_type_clust_tab1")
            
            viz_df = X_orig.copy()
            viz_df['Cluster'] = labels
            viz_df['Cluster'] = viz_df['Cluster'].astype(str)
            viz_df['Cluster'] = viz_df['Cluster'].replace('-1', 'Bruit')
            
            if viz_type == "2D":
                fig = px.scatter(
                    viz_df,
                    x=x_axis, y=y_axis,
                    color='Cluster',
                    title=f"Clustering {latest_result['algorithm']} - Vue 2D",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data=features
                )
                
                # Add centroids for K-Means and medoids for K-Medoids
                if latest_result['algorithm'] in ['K-Means']:
                    model = latest_result['model']
                    if hasattr(model, 'cluster_centers_'):
                        centers = model.cluster_centers_
                        x_idx = features.index(x_axis)
                        y_idx = features.index(y_axis)
                        fig.add_trace(go.Scatter(
                            x=centers[:, x_idx],
                            y=centers[:, y_idx],
                            mode='markers',
                            marker=dict(symbol='x', size=15, color='black', line=dict(width=2, color='white')),
                            name='Centroïdes',
                            showlegend=True
                        ))
                elif latest_result['algorithm'] in ['K-Medoids']:
                    model = latest_result['model']
                    if hasattr(model, 'medoids'):
                        medoids = model.medoids
                        x_idx = features.index(x_axis)
                        y_idx = features.index(y_axis)
                        fig.add_trace(go.Scatter(
                            x=medoids[:, x_idx],
                            y=medoids[:, y_idx],
                            mode='markers',
                            marker=dict(symbol='diamond', size=15, color='red', line=dict(width=2, color='white')),
                            name='Médoïdes',
                            showlegend=True
                        ))
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                if len(features) >= 3:
                    z_axis = st.selectbox("Axe Z:", features, index=min(2, len(features)-1), key="z_axis_clust_tab1")
                    fig = px.scatter_3d(
                        viz_df,
                        x=x_axis, y=y_axis, z=z_axis,
                        color='Cluster',
                        title=f"Clustering {latest_result['algorithm']} - Vue 3D",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        hover_data=features
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Au moins 3 features sont nécessaires pour la visualisation 3D")
    
    # ========================================================================
    # TAB 2: MÉTRIQUES
    # ========================================================================
    with tab2:
        if task_latest == "Classification (Supervised)":
            st.subheader("Évaluation de la Classification")
            metrics = latest_result['metrics']
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Accuracy", value=f"{metrics.get('Accuracy', 0):.4f}")
            with col2:
                st.metric(label="Precision (macro)", value=f"{metrics.get('Precision (macro)', 0):.4f}")
            with col3:
                st.metric(label="Recall (macro)", value=f"{metrics.get('Recall (macro)', 0):.4f}")
            with col4:
                st.metric(label="F1-score (macro)", value=f"{metrics.get('F1-score (macro)', 0):.4f}")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.subheader("Matrice de Confusion")
            cm = metrics.get('Confusion Matrix')
            if cm is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
                    ax.set_title("Matrice de Confusion")
                    ax.set_xlabel("Classe Prédite")
                    ax.set_ylabel("Classe Vraie")
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("##### Interprétation")
                    total = cm.sum()
                    correct = np.trace(cm)
                    incorrect = total - correct
                    
                    st.metric("Prédictions Correctes", f"{correct}/{total}")
                    st.metric("Prédictions Incorrectes", f"{incorrect}/{total}")
                    st.metric("Taux de Réussite", f"{100*correct/total:.1f}%")
            
            st.markdown("---")
            
            # Per-class statistics
            st.subheader("Statistiques par Classe")
            y_true = latest_result['y_true']
            y_pred = latest_result['labels']
            unique_classes = np.unique(y_true)
            
            class_stats = []
            for cls in unique_classes:
                mask_true = y_true == cls
                mask_pred = y_pred == cls
                
                n_true = np.sum(mask_true)
                n_pred = np.sum(mask_pred)
                n_correct = np.sum((y_true == cls) & (y_pred == cls))
                
                class_accuracy = n_correct / n_true if n_true > 0 else 0
                
                class_stats.append({
                    'Classe': f'Classe {cls}',
                    'Instances Vraies': n_true,
                    'Instances Prédites': n_pred,
                    'Correctes': n_correct,
                    'Accuracy': f"{class_accuracy:.3f}"
                })
            
            stats_df = pd.DataFrame(class_stats)
            st.dataframe(stats_df, use_container_width=True)
        
        else:  # Clustering
            st.subheader("Évaluation Quantitative")
            metrics = latest_result['metrics']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Silhouette Score", 
                    value=metrics.get('Silhouette Score', 'N/A'),
                    help="Plus proche de 1 = meilleur clustering (de -1 à 1)"
                )
            with col2:
                st.metric(
                    label="Calinski-Harabasz Index", 
                    value=metrics.get('Calinski-Harabasz', 'N/A'),
                    help="Plus élevé = meilleur (ratio variance inter/intra-cluster)"
                )
            with col3:
                st.metric(
                    label="Davies-Bouldin Index", 
                    value=metrics.get('Davies-Bouldin', 'N/A'),
                    help="Plus proche de 0 = meilleur (similarité moyenne des clusters)"
                )
            
            if 'Note' in metrics:
                st.info(f"ℹ️ {metrics['Note']}")
            
            st.markdown("---")
            
            # Cluster statistics
            st.subheader("Statistiques des Clusters")
            labels = latest_result['labels']
            unique_labels = sorted(set(labels))
            
            cluster_stats = []
            for label in unique_labels:
                count = np.sum(labels == label)
                percentage = (count / len(labels)) * 100
                cluster_stats.append({
                    'Cluster': 'Bruit' if label == -1 else f'Cluster {label}',
                    'Nombre de points': count,
                    'Pourcentage': f"{percentage:.1f}%"
                })
            
            stats_df = pd.DataFrame(cluster_stats)
            st.dataframe(stats_df, use_container_width=True)
            
            # Visualization of cluster sizes
            st.markdown("---")
            st.subheader("Distribution des Tailles de Clusters")
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=[s['Cluster'] for s in cluster_stats],
                    values=[s['Nombre de points'] for s in cluster_stats],
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set2)
                )
            ])
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 3: GRAPHIQUES COMPLÉMENTAIRES
    # ========================================================================
    with tab3:
        if task_latest == "Classification (Supervised)":
            st.subheader("Analyses Complémentaires (Classification)")
            algo = latest_result['algorithm']
            
            # Feature Importance for Tree-based algorithms
            if algo in ["C4.5 (Entropy)", "CART (Gini)"]:
                st.markdown("#### 🌲 Importance des Features")
                model = latest_result['model'].model
                if hasattr(model, 'feature_importances_'):
                    features = latest_result['features']
                    importances = model.feature_importances_
                    
                    # Create dataframe and sort
                    feat_imp_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)
                    
                    # Plot horizontal bar chart
                    fig = go.Figure(go.Bar(
                        x=feat_imp_df['Importance'],
                        y=feat_imp_df['Feature'],
                        orientation='h',
                        marker_color='#66BB6A',
                        text=[f"{v:.3f}" for v in feat_imp_df['Importance']],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="Importance des Features (Gain d'Information)",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 Plus la valeur est élevée, plus la feature est importante pour la classification")
                
                # Decision Tree Plot
                st.markdown("#### 🌳 Visualisation de l'Arbre de Décision")
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Create a dummy plot_tree call to get the tree structure
                    # Since our custom DecisionTree doesn't have plot_tree compatibility,
                    # we'll create a simple text-based visualization
                    def plot_custom_tree(node, x=0.5, y=1.0, width=1.0, depth=0, ax=ax, features=None):
                        if node.value is not None:
                            # Leaf node
                            ax.text(x, y, f'Classe {node.value}', 
                                  ha='center', va='center', 
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                                  fontsize=10)
                            return
                        
                        # Internal node
                        feature_name = features[node.feature] if features else f'Feature {node.feature}'
                        ax.text(x, y, f'{feature_name}\n≤ {node.threshold:.3f}', 
                              ha='center', va='center', 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
                              fontsize=9)
                        
                        # Draw edges
                        left_x = x - width/4
                        right_x = x + width/4
                        child_y = y - 0.1
                        
                        ax.plot([x, left_x], [y-0.03, child_y+0.03], 'k-', linewidth=1)
                        ax.plot([x, right_x], [y-0.03, child_y+0.03], 'k-', linewidth=1)
                        
                        # Recurse
                        plot_custom_tree(node.left, left_x, child_y, width/2, depth+1, ax, features)
                        plot_custom_tree(node.right, right_x, child_y, width/2, depth+1, ax, features)
                    
                    plot_custom_tree(model.root, features=latest_result['features'])
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1.1)
                    ax.axis('off')
                    ax.set_title(f"Arbre de Décision - {algo}")
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    st.info("💡 L'arbre montre les décisions prises à chaque nœud basé sur les features")
                    
                except Exception as e:
                    st.warning(f"⚠️ Impossible de visualiser l'arbre: {str(e)}")
            
            st.markdown("---")
            
            # Class Distribution
            st.markdown("#### 📊 Distribution des Classes")
            y_true = latest_result['y_true']
            y_pred = latest_result['labels']
            unique_classes = np.unique(y_true)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # True class distribution
                class_counts = [np.sum(y_true == cls) for cls in unique_classes]
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Classe {c}" for c in unique_classes],
                        y=class_counts,
                        marker_color='#42A5F5',
                        text=class_counts,
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Distribution des Classes (Vraies)",
                    xaxis_title="Classe",
                    yaxis_title="Nombre d'instances",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Prediction distribution
                pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Classe {c}" for c in pred_unique],
                        y=pred_counts,
                        marker_color='#66BB6A',
                        text=pred_counts,
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Distribution des Classes (Prédites)",
                    xaxis_title="Classe",
                    yaxis_title="Nombre d'instances",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve for binary classification
            if len(unique_classes) == 2:
                st.markdown("---")
                st.markdown("#### 📈 Courbe ROC (Classification Binaire)")
                
                try:
                    from sklearn.metrics import roc_curve, auc
                    
                    # Get probability predictions if available
                    model = latest_result['model'].model
                    if hasattr(model, 'predict_proba'):
                        X_df = latest_result['X_scaled']
                        y_score = model.predict_proba(X_df.values)[:, 1]
                        
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        roc_auc = auc(fpr, tpr)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC curve (AUC = {roc_auc:.3f})',
                            line=dict(color='#2196F3', width=3)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Classificateur Aléatoire',
                            line=dict(color='gray', width=2, dash='dash')
                        ))
                        fig.update_layout(
                            title=f'Courbe ROC - {algo}',
                            xaxis_title='Taux de Faux Positifs (FPR)',
                            yaxis_title='Taux de Vrais Positifs (TPR)',
                            height=500,
                            xaxis=dict(range=[0, 1]),
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("AUC Score", f"{roc_auc:.3f}")
                        with col2:
                            interpretation = "Excellent" if roc_auc > 0.9 else "Bon" if roc_auc > 0.8 else "Acceptable" if roc_auc > 0.7 else "Faible"
                            st.metric("Interprétation", interpretation)
                        with col3:
                            st.metric("Random Baseline", "0.500")
                        
                        st.info("💡 AUC (Area Under Curve) mesure la capacité du modèle à distinguer entre les classes. Plus proche de 1.0 = meilleur")
                    else:
                        st.warning("⚠️ Cet algorithme ne supporte pas les probabilités de prédiction")
                except Exception as e:
                    st.error(f"Erreur lors du calcul de la courbe ROC: {str(e)}")
            
            # Misclassification Analysis
            st.markdown("---")
            st.markdown("#### ❌ Analyse des Erreurs de Classification")
            
            errors = y_true != y_pred
            n_errors = np.sum(errors)
            error_rate = n_errors / len(y_true)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre d'erreurs", n_errors)
            with col2:
                st.metric("Taux d'erreur", f"{error_rate*100:.2f}%")
            
            if n_errors > 0:
                # Show most common misclassifications
                error_pairs = list(zip(y_true[errors], y_pred[errors]))
                from collections import Counter
                error_counts = Counter(error_pairs)
                most_common = error_counts.most_common(5)
                
                st.markdown("##### Erreurs les plus fréquentes:")
                error_data = []
                for (true_cls, pred_cls), count in most_common:
                    error_data.append({
                        'Vraie Classe': f'Classe {true_cls}',
                        'Prédite Comme': f'Classe {pred_cls}',
                        'Nombre': count,
                        'Pourcentage': f"{100*count/n_errors:.1f}%"
                    })
                error_df = pd.DataFrame(error_data)
                st.dataframe(error_df, use_container_width=True)
        
        else:  # Clustering
            st.subheader("Analyses Complémentaires (Clustering)")
            algo = latest_result['algorithm']
            
            # Elbow Method for K-Means
            if algo == "K-Means":
                st.markdown("#### 📉 Méthode du coude (Elbow Method)")
                with st.spinner("Calcul de l'inertie..."):
                    X = latest_result['X_scaled'].values
                    inertias = []
                    silhouettes = []
                    K_range = range(2, 11)
                    
                    for k in K_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(X)
                        inertias.append(kmeans.inertia_)
                        
                        # Calculate silhouette for this k
                        labels_k = kmeans.labels_
                        if len(set(labels_k)) > 1:
                            sil = silhouette_score(X, labels_k)
                            silhouettes.append(sil)
                        else:
                            silhouettes.append(0)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Inertia plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(K_range),
                            y=inertias,
                            mode='lines+markers',
                            marker=dict(size=10, color='#FF6B6B'),
                            line=dict(width=2, color='#4ECDC4')
                        ))
                        fig.update_layout(
                            title="Inertie vs Nombre de Clusters",
                            xaxis_title="Nombre de clusters (k)",
                            yaxis_title="Inertie",
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Silhouette plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(K_range),
                            y=silhouettes,
                            mode='lines+markers',
                            marker=dict(size=10, color='#95E1D3'),
                            line=dict(width=2, color='#38ADA9')
                        ))
                        fig.update_layout(
                            title="Silhouette Score vs Nombre de Clusters",
                            xaxis_title="Nombre de clusters (k)",
                            yaxis_title="Silhouette Score",
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 Le 'coude' dans l'inertie et le maximum du Silhouette Score indiquent le nombre optimal de clusters")
            
            # Dendrogram for hierarchical methods
            elif algo in ["AGNES", "DIANA"]:
                st.markdown("#### 🌳 Dendrogramme")
                with st.spinner("Génération du dendrogramme..."):
                    X = latest_result['X_scaled'].values
                    linkage_method = latest_result['params']['linkage']
                    
                    # Sample if too many points
                    if len(X) > 100:
                        indices = np.random.choice(len(X), 100, replace=False)
                        X_sample = X[indices]
                        st.warning("⚠️ Affichage limité à 100 points pour la lisibilité")
                    else:
                        X_sample = X
                    
                    Z = linkage(X_sample, method=linkage_method)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    dendrogram(Z, ax=ax, color_threshold=0)
                    ax.set_title(f"Dendrogramme (linkage: {linkage_method})")
                    ax.set_xlabel("Index des échantillons")
                    ax.set_ylabel("Distance")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    
                    st.info("💡 La hauteur des fusions indique la dissimilarité entre les clusters")
            
            st.markdown("---")
            
            # Cluster Distribution
            st.markdown("#### 📊 Distribution des Clusters")
            labels = latest_result['labels']
            unique_labels = sorted(set(labels))
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            cluster_names = ['Bruit' if label == -1 else f'Cluster {label}' for label in unique_labels]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=cluster_names,
                        y=cluster_sizes,
                        marker_color='#95E1D3',
                        text=cluster_sizes,
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Taille des Clusters",
                    xaxis_title="Cluster",
                    yaxis_title="Nombre de points",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = go.Figure(data=[
                    go.Pie(
                        labels=cluster_names,
                        values=cluster_sizes,
                        hole=0.3,
                        marker=dict(colors=px.colors.qualitative.Set2)
                    )
                ])
                fig.update_layout(
                    title="Proportion des Clusters",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions per cluster
            st.markdown("---")
            st.markdown("#### 📈 Distribution des Features par Cluster")
            
            X_orig = latest_result['X_original']
            features = latest_result['features']
            
            selected_feature = st.selectbox(
                "Sélectionner une feature à analyser:",
                features,
                key="feature_dist_tab3"
            )
            
            # Create box plot
            plot_df = X_orig.copy()
            plot_df['Cluster'] = labels
            plot_df['Cluster'] = plot_df['Cluster'].astype(str).replace('-1', 'Bruit')
            
            fig = go.Figure()
            for cluster in sorted(plot_df['Cluster'].unique()):
                cluster_data = plot_df[plot_df['Cluster'] == cluster][selected_feature]
                fig.add_trace(go.Box(
                    y=cluster_data,
                    name=f'Cluster {cluster}',
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=f"Distribution de '{selected_feature}' par Cluster",
                yaxis_title=selected_feature,
                xaxis_title="Cluster",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    # TAB 4: COMPARAISONS
    with tab4:
        st.subheader("Comparaison des Exécutions")
        if len(st.session_state.results_history) > 1:
            comparison_data = []
            for i, result in enumerate(st.session_state.results_history):
                row = {
                    'Exécution': f"#{i+1}",
                    'Tâche': result.get('task_type', 'Clustering'),
                    'Algorithme': result['algorithm'],
                    'Paramètres': str(result['params']),
                    'Features': ', '.join(result['features'][:3]) + '...' if len(result['features']) > 3 else ', '.join(result['features']),
                }
                metrics = result['metrics']
                if result.get('task_type') == "Classification (Supervised)":
                    row['Accuracy'] = metrics.get('Accuracy', 'N/A')
                    row['F1 (macro)'] = metrics.get('F1-score (macro)', 'N/A')
                else:
                    row['Silhouette'] = metrics.get('Silhouette Score', 'N/A')
                    row['Davies-Bouldin'] = metrics.get('Davies-Bouldin', 'N/A')
                    labels = result['labels']
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    row['Clusters trouvés'] = n_clusters
                comparison_data.append(row)
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📊 Comparaison des Métriques")
            # Build separate visualizations for clustering/classification
            col1, col2 = st.columns(2)
            valid_clust = [r for r in st.session_state.results_history if r.get('task_type') == "Clustering (Unsupervised)" and r['metrics'].get('Silhouette Score') is not None]
            if valid_clust:
                values = [r['metrics']['Silhouette Score'] for r in valid_clust]
                labels_bar = [f"{r['algorithm']}" for r in valid_clust]
                fig_c = go.Figure(data=[
                    go.Bar(x=labels_bar, y=values, marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(values)],
                           text=[f"{v:.3f}" for v in values], textposition='auto')
                ])
                fig_c.update_layout(title="Silhouette (Clustering)", xaxis_title="Algorithme", yaxis_title="Score", height=400)
                col1.plotly_chart(fig_c, use_container_width=True)
            valid_cls = [r for r in st.session_state.results_history if r.get('task_type') == "Classification (Supervised)" and r['metrics'].get('Accuracy') is not None]
            if valid_cls:
                values = [r['metrics']['Accuracy'] for r in valid_cls]
                labels_bar = [f"{r['algorithm']}" for r in valid_cls]
                fig_s = go.Figure(data=[
                    go.Bar(x=labels_bar, y=values, marker_color=['#7E57C2', '#26A69A', '#EF5350', '#42A5F5'][:len(values)],
                           text=[f"{v:.3f}" for v in values], textposition='auto')
                ])
                fig_s.update_layout(title="Accuracy (Classification)", xaxis_title="Algorithme", yaxis_title="Score", height=400)
                col2.plotly_chart(fig_s, use_container_width=True)

            if st.button("🗑️ Réinitialiser l'historique"):
                st.session_state.results_history = []
                st.rerun()
        else:
            st.info("ℹ️ Exécutez plusieurs algorithmes pour comparer les résultats")

else:
    st.info("""
    ### 👋 Bienvenue dans l'application de comparaison de clustering!

    **Pour commencer:**
    1. Chargez un dataset dans la barre latérale (prédéfini ou personnalisé)
    2. Sélectionnez au moins 2 features numériques
    3. Configurez les paramètres de l'algorithme souhaité
    4. Cliquez sur "Exécuter le clustering"

    **Algorithmes disponibles:**
    - 🔵 **K-Means**: Partitionnement en k clusters sphériques
    - 🟢 **DBSCAN**: Détection de clusters de densité variable
    - 🟡 **K-Medoids**: Partitionnement autour de médoides (robuste aux outliers)
    - 🟠 **AGNES**: Clustering hiérarchique ascendant (agglomératif)
    - 🔴 **DIANA**: Clustering hiérarchique descendant (divisif)

    **Algorithmes de classification (nouveau):**
    - 🔷 **KNN**
    - 🟣 **Naive Bayes (Gaussian)**
    - 🟤 **C4.5 (Entropy)**
    - 🟠 **CART (Gini)**
    - ⚫ **SVM (RBF)**

    **Métriques d'évaluation:**
    - Clustering: Silhouette, Calinski-Harabasz, Davies-Bouldin
    - Classification: Accuracy, Precision (macro), Recall (macro), F1-score (macro), Confusion Matrix
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>📚 Application Pédagogique</p>
    <p>Data Mining & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
