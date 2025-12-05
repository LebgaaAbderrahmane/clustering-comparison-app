"""
Application Streamlit pour la Comparaison d'Algorithmes de Clustering
=====================================================================
Auteur: Expert en Data Mining
Version: 1.0
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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

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
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False

if 'unscaled_data' not in st.session_state:
    st.session_state.unscaled_data = None

if 'data_before_scaling' not in st.session_state:
    st.session_state.data_before_scaling = None
    
# ============================================================================
# FONCTIONS UTILITAIRES
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
        # Limiter à 500 points pour la performance
        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
    return df

def check_missing_values(df):
    """Vérifie les valeurs manquantes dans le DataFrame"""
    missing = df.isnull().sum()
    if missing.any():
        return True, missing[missing > 0]
    return False, None

def run_clustering(algo_name, params, data):
    """
    Exécute l'algorithme de clustering sélectionné
    
    Args:
        algo_name: Nom de l'algorithme
        params: Dictionnaire des paramètres
        data: DataFrame avec les features
    
    Returns:
        labels: Labels des clusters
        model: Modèle entraîné
    """
    X = data.values
    
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
        # Agglomerative Nesting (bottom-up hierarchical clustering)
        model = AgglomerativeClustering(
            n_clusters=params['n_clusters'],
            linkage=params['linkage']
        )
        labels = model.fit_predict(X)
        
    elif algo_name == "DIANA":
        # Divisive Analysis (top-down hierarchical clustering)
        # Using divisive approach with scipy linkage
        Z = linkage(X, method=params['linkage'])
        model = AgglomerativeClustering(
            n_clusters=params['n_clusters'],
            linkage=params['linkage']
        )
        labels = model.fit_predict(X)
        model.linkage_matrix = Z
    
    return labels, model

def calculate_metrics(X, labels):
    """Calcule les métriques de qualité du clustering"""
    # Vérifier qu'il y a au moins 2 clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters < 2:
        return {
            'Silhouette Score': None,
            'Calinski-Harabasz': None,
            'Davies-Bouldin': None,
            'Note': 'Métriques non calculables (< 2 clusters)'
        }
    
    try:
        # Filtrer les points de bruit pour DBSCAN
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

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("📊 Comparaison d'Algorithmes de Clustering")
st.markdown("### Application pédagogique pour l'analyse et la visualisation de clustering")

# ============================================================================
# SIDEBAR - CHARGEMENT DES DONNÉES
# ============================================================================

st.sidebar.header("🔧 Configuration")

# Section 1: Chargement des données
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

else:
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
            
            # Vérifier les valeurs manquantes
            has_missing, missing_cols = check_missing_values(df)
            if has_missing:
                st.sidebar.error("❌ Valeurs manquantes détectées:")
                st.sidebar.write(missing_cols)
                df = None
            else:
                st.session_state.current_data = df
                st.session_state.data_before_scaling = df.copy()
                st.sidebar.success(f"✅ Fichier chargé ({len(df)} lignes)")
                
        except Exception as e:
            st.sidebar.error(f"❌ Erreur de chargement: {str(e)}")

# ============================================================================
# SIDEBAR - DATA PREPROCESSING (FIXED FOR SCALING REAPPLICATION)
# ============================================================================
if st.session_state.current_data is not None:
    df = st.session_state.current_data.copy()
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧰 Prétraitement des Données")

    # Check for missing values
    has_missing, missing_cols = check_missing_values(df)
    if has_missing:
        st.sidebar.warning(f"⚠️ {len(missing_cols)} colonnes contiennent des valeurs manquantes.")
        missing_action = st.sidebar.selectbox(
            "Gérer les valeurs manquantes:",
            ["Supprimer les lignes", "Remplir par la moyenne", "Remplir par la médiane", "Remplir par 0"]
        )
        if st.sidebar.button("🔧 Appliquer le traitement"):
            if missing_action == "Supprimer les lignes":
                df = df.dropna()
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                non_numeric_cols = df.columns.difference(numeric_cols)

                if missing_action == "Remplir par la moyenne":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif missing_action == "Remplir par la médiane":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                else:  # Remplir par 0
                    df[numeric_cols] = df[numeric_cols].fillna(0)

                for col in non_numeric_cols:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()
                        fill_val = mode_val[0] if not mode_val.empty else "missing"
                        df[col] = df[col].fillna(fill_val)

            st.session_state.current_data = df
            st.session_state.data_before_scaling = df.copy()  # ← ADD THIS LINE
            st.session_state.unscaled_data = None
            st.session_state.scaling_applied = False
            st.session_state.data_preprocessed = False
            st.sidebar.success(f"✅ Valeurs manquantes traitées ({len(df)} lignes restantes)")
            st.rerun()

    # Categorical Encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) > 0:
        st.sidebar.markdown("**Encodage des variables catégorielles:**")
        encoding_method = st.sidebar.selectbox(
            "Choisir la méthode d'encodage:",
            ["Aucun", "One-Hot Encoding", "Label Encoding"]
        )
        if encoding_method != "Aucun":
            st.sidebar.info("⚠️ L'encodage sera appliqué sur toutes les colonnes catégorielles.")
    else:
        encoding_method = "Aucun"

    # Feature Scaling
    st.sidebar.markdown("**Normalisation des features:**")
    scaling_method = st.sidebar.selectbox(
        "Choisir la méthode de normalisation:",
        ["Aucune", "StandardScaler (Z-score)", "MinMaxScaler"]
    )

    # Apply preprocessing (missing + encoding → store as unscaled; then scale if needed)
    if st.sidebar.button("⚙️ Appliquer le prétraitement", type="secondary"):
        with st.spinner("Application du prétraitement..."):
            # Always start from clean data (after missing-value handling, before scaling/encoding)
            if st.session_state.data_before_scaling is not None:
                df_base = st.session_state.data_before_scaling.copy()
            else:
                df_base = df.copy()  # fallback

            df_preprocessed = df_base.copy()

            scaling_done = False
            encoding_applied = False

            # Handle encoding FIRST (affects column structure)
            categorical_cols_base = df_base.select_dtypes(include=['object', 'category']).columns.tolist()
            if encoding_method != "Aucun" and len(categorical_cols_base) > 0:
                if encoding_method == "One-Hot Encoding":
                    df_preprocessed = pd.get_dummies(df_preprocessed, columns=categorical_cols_base, drop_first=True)
                else:  # Label Encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    for col in categorical_cols_base:
                        df_preprocessed[col] = le.fit_transform(df_preprocessed[col].astype(str))
                encoding_applied = True
                st.sidebar.success(f"✅ Variables catégorielles encodées avec {encoding_method}")

            # Handle scaling on the (possibly encoded) data
            if scaling_method != "Aucune":
                numeric_cols = df_preprocessed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    scaler = StandardScaler() if scaling_method == "StandardScaler (Z-score)" else MinMaxScaler()
                    df_preprocessed[numeric_cols] = scaler.fit_transform(df_preprocessed[numeric_cols])
                    scaling_done = True
                    st.sidebar.success(f"✅ Features normalisées avec {scaling_method}")
                else:
                    st.sidebar.warning("⚠️ Aucune colonne numérique à normaliser.")

            # Save final result
            st.session_state.current_data = df_preprocessed
            st.session_state.scaling_applied = scaling_done
            st.session_state.data_preprocessed = True
            st.sidebar.success("✅ Prétraitement terminé!")
            st.rerun()

# Afficher aperçu et sélection des colonnes
if st.session_state.current_data is not None:
    df = st.session_state.current_data
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Aperçu des données")
    st.sidebar.dataframe(df.head(), height=150)
    
    # Sélection des features numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    if len(numeric_cols) >= 2:
        selected_features = st.sidebar.multiselect(
            "Sélectionner les features (min. 2):",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))]
        )
    else:
        st.sidebar.error("❌ Au moins 2 features numériques requises")

# ============================================================================
# SIDEBAR - PARAMÉTRAGE DES ALGORITHMES
# ============================================================================

if len(selected_features) >= 2:
    # Show data state
    if st.session_state.scaling_applied:
        st.sidebar.markdown("📌 **Données** : ✅ Normalisées")
    else:
        st.sidebar.markdown("📌 **Données** : ⚠️ Non normalisées (seront normalisées avant clustering)")

    
if len(selected_features) >= 2:
    st.sidebar.markdown("---")
    st.sidebar.subheader("2️⃣ Configuration du clustering")
    
    algo_name = st.sidebar.selectbox(
        "Choisir l'algorithme:",
        ["K-Means", "DBSCAN", "K-Medoids", "AGNES", "DIANA"]
    )
    
    params = {}
    
    # Paramètres spécifiques à chaque algorithme
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
    
        # Bouton d'exécution
    if st.sidebar.button("🚀 Exécuter le clustering", type="primary"):
        with st.spinner("Clustering en cours..."):
            df = st.session_state.current_data
            X_df = df[selected_features].copy()  # Already preprocessed (encoded + scaled if chosen)

            # Run clustering directly on preprocessed data
            labels, model = run_clustering(algo_name, params, X_df)

            # Use raw preprocessed array for metrics
            X_scaled = X_df.values
            metrics = calculate_metrics(X_scaled, labels)

            # Stocker les résultats
            result = {
                'algorithm': algo_name,
                'params': params.copy(),
                'labels': labels,
                'model': model,
                'metrics': metrics,
                'features': selected_features,
                'X_scaled': X_df,          # Now already scaled if chosen
                'X_original': X_df         # Same as scaled if preprocessed, else raw
            }
            st.session_state.results_history.append(result)
            st.sidebar.success("✅ Clustering terminé!")

# ============================================================================
# ZONE CENTRALE - AFFICHAGE DES RÉSULTATS
# ============================================================================

if len(st.session_state.results_history) > 0:
    latest_result = st.session_state.results_history[-1]
    
    # Créer des onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Visualisation 2D/3D",
        "📊 Métriques",
        "📉 Graphiques complémentaires",
        "🔄 Comparaisons"
    ])
    
    # ========================================================================
    # TAB 1: VISUALISATION 2D/3D
    # ========================================================================
    with tab1:
        st.subheader("Visualisation des Clusters")
        
        col1, col2, col3 = st.columns(3)
        
        features = latest_result['features']
        X_orig = latest_result['X_original']
        labels = latest_result['labels']
        
        with col1:
            x_axis = st.selectbox("Axe X:", features, index=0)
        with col2:
            y_axis = st.selectbox("Axe Y:", features, index=min(1, len(features)-1))
        with col3:
            viz_type = st.radio("Type:", ["2D", "3D"])
        
        # Préparer les données pour la visualisation
        viz_df = X_orig.copy()
        viz_df['Cluster'] = labels
        viz_df['Cluster'] = viz_df['Cluster'].astype(str)
        
        # Remplacer -1 (bruit DBSCAN) par "Bruit"
        viz_df['Cluster'] = viz_df['Cluster'].replace('-1', 'Bruit')
        
        if viz_type == "2D":
            fig = px.scatter(
                viz_df,
                x=x_axis,
                y=y_axis,
                color='Cluster',
                title=f"Clustering {latest_result['algorithm']} - Vue 2D",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hover_data=features
            )
            
            # Ajouter les centroïdes pour K-Means et GMM
            if latest_result['algorithm'] in ['K-Means', 'GMM']:
                model = latest_result['model']
                if hasattr(model, 'cluster_centers_'):
                    centers = model.cluster_centers_
                elif hasattr(model, 'means_'):
                    centers = model.means_
                else:
                    centers = None
                
                if centers is not None:
                    # Récupérer les indices des features sélectionnées
                    x_idx = features.index(x_axis)
                    y_idx = features.index(y_axis)
                    
                    fig.add_trace(go.Scatter(
                        x=centers[:, x_idx],
                        y=centers[:, y_idx],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=15,
                            color='black',
                            line=dict(width=2, color='white')
                        ),
                        name='Centroïdes',
                        showlegend=True
                    ))
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # 3D
            if len(features) >= 3:
                z_axis = st.selectbox("Axe Z:", features, index=min(2, len(features)-1))
                
                fig = px.scatter_3d(
                    viz_df,
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
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
        st.subheader("Évaluation Quantitative")
        
        metrics = latest_result['metrics']
        
        # Afficher les métriques avec explications
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
        
        # Informations sur les clusters
        st.markdown("---")
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
    
    # ========================================================================
    # TAB 3: GRAPHIQUES COMPLÉMENTAIRES
    # ========================================================================
    with tab3:
        st.subheader("Analyses Complémentaires")
        
        algo = latest_result['algorithm']
        
        if algo == "K-Means":
            st.markdown("#### 📉 Méthode du coude (Elbow Method)")
            
            with st.spinner("Calcul de l'inertie..."):
                X = latest_result['X_scaled'].values
                inertias = []
                K_range = range(2, 11)
                
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X)
                    inertias.append(kmeans.inertia_)
                
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
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 Le 'coude' indique le nombre optimal de clusters")
        
        elif algo in ["AGNES", "DIANA"]:
            st.markdown("#### 🌳 Dendrogramme")
            
            with st.spinner("Génération du dendrogramme..."):
                X = latest_result['X_scaled'].values
                linkage_method = latest_result['params']['linkage']
                
                # Limiter à 100 points pour la lisibilité
                if len(X) > 100:
                    indices = np.random.choice(len(X), 100, replace=False)
                    X_sample = X[indices]
                    st.warning("⚠️ Affichage limité à 100 points pour la lisibilité")
                else:
                    X_sample = X
                
                Z = linkage(X_sample, method=linkage_method)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                dendrogram(Z, ax=ax)
                ax.set_title(f"Dendrogramme (linkage: {linkage_method})")
                ax.set_xlabel("Index des échantillons")
                ax.set_ylabel("Distance")
                
                st.pyplot(fig)
                plt.close()
        
        # Histogramme des tailles de clusters (pour tous les algos)
        st.markdown("---")
        st.markdown("#### 📊 Distribution des Clusters")
        
        labels = latest_result['labels']
        unique_labels = sorted(set(labels))
        
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        cluster_names = ['Bruit' if label == -1 else f'Cluster {label}' for label in unique_labels]
        
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
    
    # ========================================================================
    # TAB 4: COMPARAISONS
    # ========================================================================
    with tab4:
        st.subheader("Comparaison des Exécutions")
        
        if len(st.session_state.results_history) > 1:
            # Tableau comparatif
            comparison_data = []
            
            for i, result in enumerate(st.session_state.results_history):
                row = {
                    'Exécution': f"#{i+1}",
                    'Algorithme': result['algorithm'],
                    'Paramètres': str(result['params']),
                    'Features': ', '.join(result['features'][:3]) + '...' if len(result['features']) > 3 else ', '.join(result['features']),
                }
                
                metrics = result['metrics']
                row['Silhouette'] = metrics.get('Silhouette Score', 'N/A')
                row['Calinski-Harabasz'] = metrics.get('Calinski-Harabasz', 'N/A')
                row['Davies-Bouldin'] = metrics.get('Davies-Bouldin', 'N/A')
                
                # Nombre de clusters
                labels = result['labels']
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                row['Clusters trouvés'] = n_clusters
                
                comparison_data.append(row)
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)
            
            # Graphique comparatif des métriques
            st.markdown("---")
            st.markdown("#### 📊 Comparaison des Métriques")
            
            # Filtrer les résultats avec métriques valides
            valid_results = [r for r in st.session_state.results_history 
                           if r['metrics'].get('Silhouette Score') is not None]
            
            if valid_results:
                metric_names = ['Silhouette Score', 'Davies-Bouldin']
                
                col1, col2 = st.columns(2)
                
                for idx, metric_name in enumerate(metric_names):
                    values = [r['metrics'][metric_name] for r in valid_results]
                    labels = [f"{r['algorithm']}" for r in valid_results]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=labels,
                            y=values,
                            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(values)],
                            text=[f"{v:.3f}" for v in values],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title=metric_name,
                        xaxis_title="Algorithme",
                        yaxis_title="Score",
                        height=400
                    )
                    
                    if idx == 0:
                        col1.plotly_chart(fig, use_container_width=True)
                    else:
                        col2.plotly_chart(fig, use_container_width=True)
            
            # Bouton pour réinitialiser l'historique
            if st.button("🗑️ Réinitialiser l'historique"):
                st.session_state.results_history = []
                st.rerun()
        else:
            st.info("ℹ️ Exécutez plusieurs algorithmes pour comparer les résultats")

else:
    # Message d'accueil
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
    
    **Métriques d'évaluation:**
    - **Silhouette Score** (-1 à 1): Cohésion et séparation des clusters
    - **Calinski-Harabasz**: Ratio variance inter/intra-cluster (plus élevé = meilleur)
    - **Davies-Bouldin** (≥0): Similarité moyenne des clusters (plus faible = meilleur)
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
