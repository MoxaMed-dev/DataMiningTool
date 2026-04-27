
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, fclusterdata
from scipy.spatial.distance import pdist, squareform
import io

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("fdd.css")
# ─────────────────────────────────────────────────────────────
#  config page 
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fouille de Données - Interface Interactive",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# # ─────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#151829",
    "axes.facecolor":    "#1E2235",
    "axes.edgecolor":    "#2A2D45",
    "axes.labelcolor":   "#E8EAF6",
    "axes.titlecolor":   "#E8EAF6",
    "xtick.color":       "#8B90B8",
    "ytick.color":       "#8B90B8",
    "text.color":        "#E8EAF6",
    "grid.color":        "#2A2D45",
    "legend.facecolor":  "#151829",
    "legend.edgecolor":  "#2A2D45",
    "figure.dpi":        110,
})
PALETTE = ["#6C63FF", "#00D4AA", "#FF5F7E", "#FFB84D", "#A78BFA",
           "#34D399", "#F472B6", "#60A5FA", "#FBBF24", "#4ADE80"]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def styled_fig(figsize=(12, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def section(title: str):
    st.markdown(f"<div style='margin:1.5rem 0 0.75rem;border-left:3px solid #6C63FF;padding-left:12px'>"
                f"<span style='font-weight:700;font-size:1rem;color:#E8EAF6'>{title}</span></div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("⚗️ Fouille de Données - Interface Interactive")
st.markdown('<i class="fas fa-database"></i>', unsafe_allow_html=True)
st.markdown("<p style='color:#8B90B8;font-size:0.95rem;margin-top:-0.5rem'>"
            "Pipeline complet : Prétraitement · Clustering · Classification supervisée</p>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
for key, default in {
    "df": None,
    "processed_df": None,
    "cluster_labels": None,
    "selected_features": None,
    "target_col": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────
# SIDEBAR — DATA UPLOAD
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Chargement des données")
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"✓ {uploaded_file.name}")
        st.info(f"**Lignes** : {st.session_state.df.shape[0]}  \n"
                f"**Colonnes** : {st.session_state.df.shape[1]}")

    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem;color:#8B90B8;text-align:center'>"
                "Mini-Projet FD1 · 2025-2026</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📋 Prétraitement",
    "🔵 Clustering",
    "🤖 Classification"
])

# ══════════════════════════════════════════════════════════════
#  VOLET 1 — PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════════
with tab1:
    if st.session_state.df is None:
        st.info("📁 Veuillez charger un fichier CSV depuis la barre latérale.")
    else:
        df = st.session_state.df

        # ── Overview ──────────────────────────────────────────
        section("Vue d'ensemble")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes", df.shape[0])
        c2.metric("Colonnes", df.shape[1])
        c3.metric("Valeurs manquantes", df.isnull().sum().sum())
        c4.metric("Types numériques", len(df.select_dtypes(include=[np.number]).columns))

        st.dataframe(df.head(10), use_container_width=True)

        # ── Exploration ───────────────────────────────────────
        section("Exploration")
        with st.expander("📊 Statistiques descriptives", expanded=False):
            st.dataframe(df.describe(), use_container_width=True)

        with st.expander("🔎 Types & valeurs manquantes", expanded=True):
            info_df = pd.DataFrame({
                "Colonne":    df.columns,
                "Type":       df.dtypes.values,
                "Manquants":  df.isnull().sum().values,
                "Manquants%": (df.isnull().sum() / len(df) * 100).round(2).values,
            })
            st.dataframe(info_df, use_container_width=True)

        # ── Nettoyage & normalisation ─────────────────────────
        section("Nettoyage & Normalisation")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Valeurs manquantes**")
            st.caption("Numérique → Moyenne · Catégoriel → Mode")
        with col_b:
            scaling_method = st.selectbox(
                "Méthode de normalisation",
                ["Aucune", "Min-Max Scaler (0-1)", "Standard Scaler (Z-score)"],
                key="scaling"
            )

        if st.button("⚙️ Appliquer le prétraitement", type="primary"):
            df_proc = df.copy()

            # Missing values
            for col in df_proc.select_dtypes(include=[np.number]).columns:
                if df_proc[col].isnull().any():
                    df_proc[col].fillna(df_proc[col].mean(), inplace=True)
            for col in df_proc.select_dtypes(include=["object"]).columns:
                if df_proc[col].isnull().any():
                    df_proc[col].fillna(df_proc[col].mode()[0], inplace=True)

            # Scaling
            num_cols = df_proc.select_dtypes(include=[np.number]).columns
            if scaling_method == "Min-Max Scaler (0-1)":
                df_proc[num_cols] = MinMaxScaler().fit_transform(df_proc[num_cols])
            elif scaling_method == "Standard Scaler (Z-score)":
                df_proc[num_cols] = StandardScaler().fit_transform(df_proc[num_cols])

            st.session_state.processed_df = df_proc
            st.success("✓ Prétraitement appliqué avec succès.")

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Valeurs manquantes traitées",
                      df.isnull().sum().sum())
            m2.metric("Colonnes numériques normalisées",
                      len(num_cols) if scaling_method != "Aucune" else 0)
            m3.metric("Méthode", scaling_method.split(" ")[0]
                      if scaling_method != "Aucune" else "—")

            st.dataframe(df_proc.head(10), use_container_width=True)

            # Before / after boxplots
            if scaling_method != "Aucune":
                section("Comparaison avant / après normalisation")
                cols_to_show = list(num_cols[:min(6, len(num_cols))])
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].boxplot([df[c].dropna().values for c in cols_to_show],
                                labels=cols_to_show, patch_artist=True,
                                boxprops=dict(facecolor="#6C63FF", alpha=0.6),
                                medianprops=dict(color="#00D4AA", linewidth=2))
                axes[0].set_title("Avant normalisation", fontweight="bold")
                axes[0].tick_params(axis="x", rotation=35)

                axes[1].boxplot([df_proc[c].values for c in cols_to_show],
                                labels=cols_to_show, patch_artist=True,
                                boxprops=dict(facecolor="#00D4AA", alpha=0.6),
                                medianprops=dict(color="#6C63FF", linewidth=2))
                axes[1].set_title(f"Après {scaling_method}", fontweight="bold")
                axes[1].tick_params(axis="x", rotation=35)
                plt.tight_layout()
                st.pyplot(fig)

            # Download
            st.download_button(
                "📥 Télécharger les données prétraitées",
                data=st.session_state.processed_df.to_csv(index=False),
                file_name="donnees_pretraitees.csv",
                mime="text/csv"
            )

        # ── Visualisation (toujours accessible) ──────────────
        section("Visualisation des données")

        num_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols_raw:
            st.warning("Aucune colonne numérique détectée.")
        else:
            viz_type = st.radio(
                "Type de graphique",
                ["Boxplot", "Scatter Plot", "Distribution (Histogramme)"],
                horizontal=True,
                key="viz_type"
            )

            if viz_type == "Boxplot":
                cols_box = st.multiselect(
                    "Colonnes à afficher",
                    num_cols_raw,
                    default=num_cols_raw[:min(5, len(num_cols_raw))],
                    key="box_cols"
                )
                if cols_box:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    bp = ax.boxplot(
                        [df[c].dropna().values for c in cols_box],
                        labels=cols_box, patch_artist=True,
                        notch=False, vert=True
                    )
                    for patch, color in zip(bp["boxes"], PALETTE):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    for median in bp["medians"]:
                        median.set_color("#00D4AA")
                        median.set_linewidth(2)
                    ax.set_title("Boxplot des variables sélectionnées", fontweight="bold")
                    ax.tick_params(axis="x", rotation=30)
                    ax.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

            elif viz_type == "Scatter Plot":
                if len(num_cols_raw) < 2:
                    st.warning("Il faut au moins 2 colonnes numériques.")
                else:
                    c1, c2 = st.columns(2)
                    col_x = c1.selectbox("Axe X", num_cols_raw, key="sx")
                    col_y = c2.selectbox("Axe Y", num_cols_raw,
                                         index=min(1, len(num_cols_raw)-1), key="sy")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sc = ax.scatter(df[col_x], df[col_y],
                                    c=range(len(df)), cmap="plasma",
                                    alpha=0.7, edgecolors="#2A2D45",
                                    linewidth=0.4, s=55)
                    ax.set_xlabel(col_x, fontweight="bold")
                    ax.set_ylabel(col_y, fontweight="bold")
                    ax.set_title(f"Scatter : {col_x} vs {col_y}", fontweight="bold")
                    ax.grid(alpha=0.3)
                    plt.colorbar(sc, ax=ax, label="Index")
                    plt.tight_layout()
                    st.pyplot(fig)

            else:  # Histogramme
                col_hist = st.selectbox("Colonne", num_cols_raw, key="hist_col")
                bins = st.slider("Nombre de bins", 5, 80, 30, key="hist_bins")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df[col_hist].dropna(), bins=bins,
                        color="#6C63FF", edgecolor="#0D0F1A", alpha=0.85)
                ax.set_xlabel(col_hist, fontweight="bold")
                ax.set_ylabel("Fréquence", fontweight="bold")
                ax.set_title(f"Distribution de {col_hist}", fontweight="bold")
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
#  VOLET 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.processed_df is None:
        st.info("📋 Veuillez d'abord prétraiter vos données dans l'onglet Prétraitement.")
    else:
        df_c = st.session_state.processed_df
        num_cols = df_c.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            st.error("❌ Il faut au moins 2 colonnes numériques.")
        else:
            section("Configuration du clustering")

            method = st.selectbox(
                "Algorithme",
                ["K-Means", "K-Medoids", "DBSCAN", "AGNES", "DIANA", "DENCLUE"],
                key="cluster_algo"
            )

            st.info(f"📊 **{len(num_cols)} features** utilisées : {', '.join(num_cols)}")

            X = df_c[num_cols].values
            labels = None

            # ── K-MEANS ───────────────────────────────────────
            if method == "K-Means":
                section("Courbe d'Elbow & Score de Silhouette")
                K_range = range(2, 11)
                inertias, sil_scores = [], []
                for k in K_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X)
                    inertias.append(km.inertia_)
                    sil_scores.append(silhouette_score(X, km.labels_))

                best_k = list(K_range)[np.argmax(sil_scores)]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
                ax1.plot(K_range, inertias, marker="o", lw=2, color=PALETTE[0])
                ax1.axvline(x=best_k, color=PALETTE[2], ls="--", lw=2,
                            label=f"k optimal = {best_k}")
                ax1.set(title="Elbow — Inertie", xlabel="k", ylabel="Inertie")
                ax1.legend(); ax1.grid(alpha=0.3)

                ax2.plot(K_range, sil_scores, marker="s", lw=2, color=PALETTE[1])
                ax2.axvline(x=best_k, color=PALETTE[2], ls="--", lw=2,
                            label=f"k optimal = {best_k}")
                ax2.set(title="Silhouette Score", xlabel="k", ylabel="Score")
                ax2.legend(); ax2.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                st.success(f"✓ K optimal suggéré : **{best_k}** (meilleur Silhouette)")

                n_clusters = st.slider("Choisir k (ou laisser optimal)",
                                       2, 10, best_k, key="km_k")

                if st.button("▶ Lancer K-Means", type="primary"):
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = km.fit_predict(X)
                    st.session_state.cluster_labels = labels
                    st.session_state.selected_features = num_cols

                    sil = silhouette_score(X, labels)
                    dbi = davies_bouldin_score(X, labels)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clusters", n_clusters)
                    m2.metric("Inertie", f"{km.inertia_:.1f}")
                    m3.metric("Silhouette", f"{sil:.3f}")
                    m4.metric("Davies-Bouldin", f"{dbi:.3f}")

            # ── K-MEDOIDS ─────────────────────────────────────
            elif method == "K-Medoids":
                from sklearn.metrics import pairwise_distances

                c1, c2, c3 = st.columns(3)
                n_clusters = c1.slider("Nombre de clusters", 2, 10, 3, key="kmed_k")
                seed = c2.slider("Random state", 0, 100, 42, key="kmed_seed")
                metric = c3.selectbox("Métrique", ["euclidean", "manhattan", "cosine"],
                                      key="kmed_metric")

                if st.button("▶ Lancer K-Medoids", type="primary"):
                    dist_mat = pairwise_distances(X, metric=metric)
                    np.random.seed(seed)
                    med_idx = np.random.choice(len(X), n_clusters, replace=False)

                    for _ in range(100):
                        dists = dist_mat[:, med_idx]
                        labels = np.argmin(dists, axis=1)
                        new_med = []
                        for i in range(n_clusters):
                            pts = np.where(labels == i)[0]
                            if len(pts) == 0:
                                new_med.append(med_idx[i])
                            else:
                                costs = dist_mat[pts][:, pts].sum(axis=1)
                                new_med.append(pts[np.argmin(costs)])
                        if np.array_equal(sorted(new_med), sorted(med_idx)):
                            break
                        med_idx = new_med

                    st.session_state.cluster_labels = labels
                    st.session_state.selected_features = num_cols
                    sil = silhouette_score(X, labels)
                    dbi = davies_bouldin_score(X, labels)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clusters", n_clusters)
                    m2.metric("Métrique", metric)
                    m3.metric("Silhouette", f"{sil:.3f}")
                    m4.metric("Davies-Bouldin", f"{dbi:.3f}")

            # ── DBSCAN ────────────────────────────────────────
            elif method == "DBSCAN":
                c1, c2, c3 = st.columns(3)
                eps = c1.slider("Epsilon", 0.1, 3.0, 0.5, step=0.05, key="db_eps")
                min_samp = c2.slider("Min samples", 2, 30, 5, key="db_ms")
                metric = c3.selectbox("Métrique", ["euclidean", "manhattan"], key="db_m")

                if st.button("▶ Lancer DBSCAN", type="primary"):
                    model = DBSCAN(eps=eps, min_samples=min_samp, metric=metric)
                    labels = model.fit_predict(X)
                    st.session_state.cluster_labels = labels
                    st.session_state.selected_features = num_cols

                    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = (labels == -1).sum()
                    mask = labels != -1
                    sil = silhouette_score(X[mask], labels[mask]) if n_cl > 1 and mask.sum() > n_cl else 0

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clusters", n_cl)
                    m2.metric("Points bruit", n_noise)
                    m3.metric("Silhouette", f"{sil:.3f}")
                    m4.metric("Epsilon", eps)

            # ── AGNES ─────────────────────────────────────────
            elif method == "AGNES":
                c1, c2 = st.columns(2)
                n_clusters = c1.slider("Nombre de clusters", 2, 10, 3, key="ag_k")
                link_m = c2.selectbox("Linkage", ["ward", "complete", "average", "single"],
                                      key="ag_link")

                if st.button("▶ Lancer AGNES", type="primary"):
                    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=link_m)
                    labels = hc.fit_predict(X)
                    st.session_state.cluster_labels = labels
                    st.session_state.selected_features = num_cols

                    sil = silhouette_score(X, labels)
                    dbi = davies_bouldin_score(X, labels)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clusters", n_clusters)
                    m2.metric("Linkage", link_m)
                    m3.metric("Silhouette", f"{sil:.3f}")
                    m4.metric("Davies-Bouldin", f"{dbi:.3f}")

                    Z = linkage(X, method=link_m)
                    fig, ax = plt.subplots(figsize=(13, 5))
                    dendrogram(Z, ax=ax, color_threshold=Z[-n_clusters+1, 2],
                               above_threshold_color="#8B90B8")
                    ax.axhline(y=Z[-n_clusters+1, 2], color=PALETTE[2],
                               ls="--", lw=2, label=f"{n_clusters} clusters")
                    ax.set_title(f"Dendrogramme AGNES ({link_m})", fontweight="bold")
                    ax.set_xlabel("Index"); ax.set_ylabel("Distance")
                    ax.legend(); ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

            # ── DIANA ─────────────────────────────────────────
            elif method == "DIANA":
                n_clusters = st.slider("Nombre de clusters", 2, 10, 3, key="di_k")

                if st.button("▶ Lancer DIANA", type="primary"):
                    Z = linkage(X, method="complete")
                    labels = fclusterdata(X, n_clusters, criterion="maxclust",
                                         method="complete") - 1
                    st.session_state.cluster_labels = labels
                    st.session_state.selected_features = num_cols

                    sil = silhouette_score(X, labels)
                    dbi = davies_bouldin_score(X, labels)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Clusters", n_clusters)
                    m2.metric("Silhouette", f"{sil:.3f}")
                    m3.metric("Davies-Bouldin", f"{dbi:.3f}")

                    fig, ax = plt.subplots(figsize=(13, 5))
                    dendrogram(Z, ax=ax, color_threshold=Z[-n_clusters+1, 2])
                    ax.axhline(y=Z[-n_clusters+1, 2], color=PALETTE[2],
                               ls="--", lw=2, label=f"{n_clusters} clusters")
                    ax.set_title("Dendrogramme DIANA", fontweight="bold")
                    ax.set_xlabel("Index"); ax.set_ylabel("Distance")
                    ax.legend(); ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

            # ── DENCLUE ───────────────────────────────────────
            elif method == "DENCLUE":
                from sklearn.neighbors import NearestNeighbors
                c1, c2 = st.columns(2)
                sigma = c1.slider("Sigma (bandwidth)", 0.1, 2.0, 0.5, step=0.05, key="den_s")
                xi = c2.slider("Xi (seuil densité)", 0.1, 1.0, 0.5, step=0.05, key="den_x")

                if st.button("▶ Lancer DENCLUE", type="primary"):
                    nbrs = NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(X)
                    dists, idxs = nbrs.kneighbors(X)
                    dens = 1.0 / (dists.mean(axis=1) + 1e-10)
                    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-10)

                    labels = np.zeros(len(X), dtype=int)
                    cluster_id, visited = 0, np.zeros(len(X), dtype=bool)
                    for i in range(len(X)):
                        if visited[i] or dens[i] < xi:
                            continue
                        queue = [i]
                        while queue:
                            cur = queue.pop(0)
                            if visited[cur]:
                                continue
                            visited[cur] = True
                            if dens[cur] >= xi:
                                labels[cur] = cluster_id
                                for nb in idxs[cur]:
                                    if not visited[nb] and dens[nb] >= xi * 0.5:
                                        queue.append(nb)
                        cluster_id += 1

                    st.session_state.cluster_labels = labels
                    st.session_state.selected_features = num_cols
                    n_cl = len(np.unique(labels))
                    sil = silhouette_score(X, labels) if n_cl > 1 else 0
                    dbi = davies_bouldin_score(X, labels) if n_cl > 1 else 0
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clusters", n_cl)
                    m2.metric("Sigma", sigma)
                    m3.metric("Silhouette", f"{sil:.3f}")
                    m4.metric("Davies-Bouldin", f"{dbi:.3f}")

            # ── Visualisation PCA 2D ──────────────────────────
            if st.session_state.cluster_labels is not None:
                section("Visualisation 2D — Projection PCA")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                unique_cl = np.unique(st.session_state.cluster_labels)

                fig, ax = plt.subplots(figsize=(10, 6))
                for idx, cl in enumerate(unique_cl):
                    mask = st.session_state.cluster_labels == cl
                    label_name = "Bruit" if cl == -1 else f"Cluster {cl}"
                    color = "#8B90B8" if cl == -1 else PALETTE[idx % len(PALETTE)]
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                               c=[color], label=label_name,
                               s=80, alpha=0.8, edgecolors="#0D0F1A",
                               linewidth=0.5)

                ax.set_xlabel(f"PC1 — {pca.explained_variance_ratio_[0]:.1%} variance",
                              fontweight="bold")
                ax.set_ylabel(f"PC2 — {pca.explained_variance_ratio_[1]:.1%} variance",
                              fontweight="bold")
                ax.set_title(f"{method} — Visualisation PCA", fontweight="bold")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                ax.grid(alpha=0.25)
                plt.tight_layout()
                st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
#  VOLET 3 — CLASSIFICATION SUPERVISÉE
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown("""
    <style>
    .swal-dev {
        background: #1E2235;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        border: 1px solid #2A2D45;
        animation: fadeIn 0.4s ease;
    }

    .swal-icon {
        font-size: 50px;
        margin-bottom: 15px;
        animation: pulse 1.5s infinite;
    }

    .swal-title {
        color: #E8EAF6;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .swal-text {
        color: #AAB0D6;
        font-size: 14px;
    }

    .swal-btn {
        margin-top: 20px;
        padding: 10px 20px;
        background: linear-gradient(135deg, #6C63FF, #00D4AA);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: 0.3s;
    }

    .swal-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,212,170,0.3);
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    </style>

    <div class="swal-dev">
        <div class="swal-icon">🚧</div>
        <div class="swal-title">En cours de développement</div>
        <div class="swal-text">
            Cette fonctionnalité sera bientôt disponible.<br>
        </div>
        
    </div>
    """, unsafe_allow_html=True)

## if asked to do class supervisée uncomment 
# with tab3:
#     if st.session_state.processed_df is None:
#         st.info("📋 Veuillez d'abord prétraiter vos données dans l'onglet Prétraitement.")
#     else:
#         df_s = st.session_state.processed_df

#         section("Configuration")

#         all_cols = df_s.columns.tolist()
#         num_cols_s = df_s.select_dtypes(include=[np.number]).columns.tolist()

#         # ── Sélection de la variable cible ────────────────────
#         target_col = st.selectbox(
#             "🎯 Variable cible (label à prédire)",
#             all_cols,
#             index=len(all_cols) - 1,
#             key="target_select"
#         )
#         st.session_state.target_col = target_col

#         feature_cols = st.multiselect(
#             "📐 Features (variables d'entrée)",
#             [c for c in num_cols_s if c != target_col],
#             default=[c for c in num_cols_s if c != target_col],
#             key="feature_select"
#         )

#         if not feature_cols:
#             st.warning("Sélectionnez au moins une feature.")
#             st.stop()

#         # ── Partitionnement ───────────────────────────────────
#         section("Partitionnement Train / Test")

#         c1, c2, c3 = st.columns(3)
#         test_size = c1.slider("Taille du jeu de test (%)", 10, 40, 20,
#                               key="test_size") / 100
#         seed = c2.slider("Random state", 0, 100, 42, key="clf_seed")
#         stratify = c3.checkbox("Stratification", value=True, key="stratify")

#         # Encodage de la cible si catégorielle
#         y_raw = df_s[target_col]
#         if y_raw.dtype == object:
#             le = LabelEncoder()
#             y = le.fit_transform(y_raw.astype(str))
#             class_names = le.classes_
#         else:
#             y = y_raw.values
#             class_names = np.unique(y).astype(str)

#         X_clf = df_s[feature_cols].values

#         try:
#             strat_y = y if stratify else None
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_clf, y, test_size=test_size, random_state=seed,
#                 stratify=strat_y
#             )
#         except ValueError:
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_clf, y, test_size=test_size, random_state=seed
#             )

#         m1, m2, m3 = st.columns(3)
#         m1.metric("Train", len(X_train))
#         m2.metric("Test", len(X_test))
#         m3.metric("Classes", len(class_names))

#         # ── Choix du modèle ───────────────────────────────────
#         section("Choix & entraînement du modèle")

#         model_name = st.selectbox(
#             "Algorithme de classification",
#             ["K-Nearest Neighbors (KNN)",
#              "Decision Tree",
#              "Random Forest",
#              "Support Vector Machine (SVM)",
#              "Logistic Regression"],
#             key="clf_model"
#         )

#         # Hyperparamètres
#         with st.expander("⚙️ Hyperparamètres", expanded=True):
#             if model_name == "K-Nearest Neighbors (KNN)":
#                 k_val = st.slider("k (voisins)", 1, 20, 5, key="knn_k")
#                 metric_knn = st.selectbox("Métrique", ["euclidean", "manhattan",
#                                                         "minkowski"], key="knn_m")
#                 clf = KNeighborsClassifier(n_neighbors=k_val, metric=metric_knn)

#             elif model_name == "Decision Tree":
#                 max_depth = st.slider("Profondeur max", 1, 20, 5, key="dt_d")
#                 criterion = st.selectbox("Critère", ["gini", "entropy"], key="dt_c")
#                 clf = DecisionTreeClassifier(max_depth=max_depth,
#                                              criterion=criterion,
#                                              random_state=seed)

#             elif model_name == "Random Forest":
#                 n_est = st.slider("Nombre d'arbres", 10, 300, 100, key="rf_n")
#                 max_depth_rf = st.slider("Profondeur max", 1, 30, 10, key="rf_d")
#                 clf = RandomForestClassifier(n_estimators=n_est,
#                                              max_depth=max_depth_rf,
#                                              random_state=seed)

#             elif model_name == "Support Vector Machine (SVM)":
#                 kernel = st.selectbox("Noyau", ["rbf", "linear", "poly"], key="svm_k")
#                 C_val = st.slider("C (régularisation)", 0.01, 10.0, 1.0, step=0.1,
#                                   key="svm_c")
#                 clf = SVC(kernel=kernel, C=C_val, random_state=seed)

#             else:  # Logistic Regression
#                 C_lr = st.slider("C (régularisation)", 0.01, 10.0, 1.0, step=0.1,
#                                  key="lr_c")
#                 solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"],
#                                       key="lr_s")
#                 clf = LogisticRegression(C=C_lr, solver=solver,
#                                          max_iter=1000, random_state=seed)

#         if st.button("▶ Entraîner & Évaluer", type="primary"):
#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_test)

#             # ── Métriques ─────────────────────────────────────
#             section("Métriques de performance")

#             avg = "weighted" if len(class_names) > 2 else "binary"
#             acc  = accuracy_score(y_test, y_pred)
#             prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
#             rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
#             f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)

#             m1, m2, m3, m4 = st.columns(4)
#             m1.metric("Accuracy",  f"{acc:.4f}")
#             m2.metric("Precision", f"{prec:.4f}")
#             m3.metric("Recall",    f"{rec:.4f}")
#             m4.metric("F1-Score",  f"{f1:.4f}")

#             # ── Matrice de confusion ──────────────────────────
#             section("Matrice de Confusion")

#             cm = confusion_matrix(y_test, y_pred)
#             fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2),
#                                             max(5, len(class_names) * 1.0)))
#             sns.heatmap(cm, annot=True, fmt="d", ax=ax,
#                         cmap="Blues",
#                         xticklabels=class_names,
#                         yticklabels=class_names,
#                         linewidths=0.5,
#                         linecolor="#2A2D45",
#                         cbar_kws={"shrink": 0.8})
#             ax.set_xlabel("Prédiction", fontweight="bold", labelpad=10)
#             ax.set_ylabel("Réalité",    fontweight="bold", labelpad=10)
#             ax.set_title(f"Matrice de Confusion — {model_name}", fontweight="bold",
#                          pad=15)
#             plt.tight_layout()
#             st.pyplot(fig)

#             # ── Rapport de classification ─────────────────────
#             section("Rapport de classification détaillé")
#             report_str = classification_report(
#                 y_test, y_pred,
#                 target_names=class_names,
#                 zero_division=0
#             )
#             st.code(report_str, language="text")

#             # ── Importance des features (si applicable) ───────
#             if hasattr(clf, "feature_importances_"):
#                 section("Importance des features")
#                 importances = clf.feature_importances_
#                 sorted_idx = np.argsort(importances)[::-1]
#                 fig, ax = plt.subplots(figsize=(10, 4))
#                 colors_bar = [PALETTE[i % len(PALETTE)]
#                               for i in range(len(feature_cols))]
#                 ax.bar(range(len(feature_cols)),
#                        importances[sorted_idx],
#                        color=[colors_bar[i] for i in range(len(feature_cols))],
#                        edgecolor="#0D0F1A", linewidth=0.5)
#                 ax.set_xticks(range(len(feature_cols)))
#                 ax.set_xticklabels(
#                     [feature_cols[i] for i in sorted_idx],
#                     rotation=30, ha="right"
#                 )
#                 ax.set_title("Importance des features (Gini)", fontweight="bold")
#                 ax.set_ylabel("Importance", fontweight="bold")
#                 ax.grid(axis="y", alpha=0.3)
#                 plt.tight_layout()
#                 st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8B90B8;font-size:0.78rem;font-family:Space Mono'>"
    " Data Mining Tool · BOUDA Mohamed · 2025-2026"
    "</p>",
    unsafe_allow_html=True
)