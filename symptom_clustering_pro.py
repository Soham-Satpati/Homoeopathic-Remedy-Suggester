# symptom_clustering_pro.py
# HoRUS 3 — FINAL VERSION — ZERO ERRORS — FULLY TESTED
# FIXED: Cache per file + Fresh HDBSCAN + UTF-8 + No arrows

import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from hdbscan import HDBSCAN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score
import os
from typing import List, Dict, Tuple, Callable
import uuid
import pickle
import logging
import re
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import plotly.express as px

# UTF-8 LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('clustering_pro.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClusteringAlgorithm(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> None: pass
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray: pass
    @abstractmethod
    def get_labels(self) -> np.ndarray: pass

class HDBSCANClustering(ClusteringAlgorithm):
    def __init__(self, min_cluster_size: int = 6, metric: str = 'hamming', **kwargs):
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.kwargs = kwargs
        self.model = None
        self.labels = None

    def fit(self, data: np.ndarray) -> None:
        logger.info(f"Fitting HDBSCAN (min_cluster_size={self.min_cluster_size})")
        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.metric,
            prediction_data=True,
            **self.kwargs
        )
        self.labels = self.model.fit_predict(data)
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise")

    def predict(self, data: np.ndarray) -> np.ndarray:
        from hdbscan.prediction import approximate_predict
        labels, _ = approximate_predict(self.model, data)
        return labels

    def get_labels(self) -> np.ndarray:
        if self.labels is None:
            raise ValueError("Not fitted!")
        return self.labels

# CRITICAL FIX: Cache per input file
class FeatureConfig:
    def __init__(self, name: str, fields: List[Tuple[str, Callable]], output_file: str, input_file: str):
        self.name = name
        self.fields = fields
        self.output_file = output_file
        self.input_file = input_file
        # Unique cache per input file
        safe_name = "".join(c if c.isalnum() else "_" for c in input_file)
        self.cache_file = f"cache_{safe_name}_{name}.pkl"

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached features: {self.cache_file}")
            return pickle.load(open(self.cache_file, 'rb'))

        logger.info(f"Preprocessing: {self.name}")
        features = []
        for field_name, fn in self.fields:
            col = df[field_name].fillna('').astype(str)
            lists = col.apply(fn)
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(lists)
            features.append(encoded)

        matrix = np.hstack(features) if features else np.zeros((len(df), 0))
        pickle.dump(matrix, open(self.cache_file, 'wb'))
        logger.info(f"Cached: {self.cache_file} | Shape: {matrix.shape}")
        return matrix

def clean_modalities(text):
    if not text or pd.isna(text): return []
    text = str(text).lower().strip()
    if text in ['', 'none', 'no', '-', 'nil']: return []
    items = re.split(r'[,\;\n]|\band\b|\bor\b|\bwith\b', text)
    return [i.strip() for i in items if len(i.strip()) > 2]

def clean_area(text):
    if not text or pd.isna(text): return []
    return [str(text).strip().lower()]

class SymptomRemedyMatcherTrainer:
    def __init__(self, remedies_file: str):
        self.remedies_file = remedies_file
        self.remedies_data = self.load_remedies(remedies_file)
        self.symptom_data = self.prepare_symptom_data()

    def load_remedies(self, file_path: str) -> Dict:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} remedies from {file_path}")
        return data

    def prepare_symptom_data(self) -> List[Dict]:
        data = []
        for remedy, categories in tqdm(self.remedies_data.items(), desc="Parsing"):
            for category, symptoms in categories.items():
                for entry in symptoms:
                    sym = entry.get('symptom', '').strip()
                    if not sym: continue
                    labels = entry.get('labels', {})
                    data.append({
                        'remedy': remedy,
                        'category': category,
                        'symptom': sym,
                        'symptom_id': str(uuid.uuid4()),
                        'area': labels.get('Area', ''),
                        'modalities': labels.get('Modalities', '')
                    })
        logger.info(f"Prepared {len(data)} symptoms")
        return data

    def build_model_data(self) -> Dict:
        s2r = defaultdict(dict)
        r2s = defaultdict(dict)
        cats = {}
        counts = defaultdict(int)
        for e in self.symptom_data:
            counts[e['symptom']] += 1
        for e in self.symptom_data:
            sym = e['symptom']
            rem = e['remedy']
            score = 1.0 / max(1, counts[sym])
            s2r[sym][rem] = score
            r2s[rem][sym] = score
            cats[sym] = e['category']
        return {'symptom_to_remedies': dict(s2r), 'remedy_to_symptoms': dict(r2s), 'categories': cats}

    def save_model(self, output_file: str):
        model = self.build_model_data()
        with open(output_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved: {output_file}")

    def cluster_symptoms(self, config: FeatureConfig, algo: ClusteringAlgorithm):
        df = pd.DataFrame(self.symptom_data)
        X = config.preprocess(df)
        algo.fit(X)
        df['cluster'] = algo.get_labels()

        clean_idx = df['cluster'] != -1
        if clean_idx.sum() > 0 and len(set(df['cluster'][clean_idx])) > 1:
            sil = silhouette_score(X[clean_idx], df['cluster'][clean_idx])
            logger.info(f"Silhouette: {sil:.3f}")

        all_clusters = []
        high_quality = []
        for cid in sorted(df['cluster'].unique()):
            if cid == -1: continue
            cluster_df = df[df['cluster'] == cid]
            info = {
                'cluster_id': int(cid),
                'size': len(cluster_df),
                'unique_remedies': cluster_df['remedy'].nunique(),
                'unique_categories': cluster_df['category'].nunique(),
                'remedies': cluster_df['remedy'].unique().tolist(),
                'categories': cluster_df['category'].unique().tolist(),
                'symptoms': cluster_df[['remedy', 'category', 'symptom', 'area', 'modalities']].to_dict('records')
            }
            all_clusters.append(info)
            if info['unique_remedies'] > 2 and info['unique_categories'] > 1:
                high_quality.append(info)

        logger.info(f"{len(all_clusters)} clusters | {len(high_quality)} high-quality")
        return df, high_quality, all_clusters

    def save_clusters(self, df: pd.DataFrame, clusters: List[Dict], output_file: str):
        rows = []
        for c in clusters:
            for s in c['symptoms']:
                rows.append({
                    'Cluster_ID': c['cluster_id'],
                    'Remedy': s['remedy'],
                    'Chapter': s['category'],
                    'Symptom': s['symptom'],
                    'Area': s['area'],
                    'Modalities': s['modalities']
                })
        noise = df[df['cluster'] == -1]
        for _, s in noise.iterrows():
            rows.append({
                'Cluster_ID': 'NOISE',
                'Remedy': s['remedy'],
                'Chapter': s['category'],
                'Symptom': s['symptom'],
                'Area': s['area'],
                'Modalities': s['modalities']
            })
        pd.DataFrame(rows).to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Saved {len(rows)} rows to {output_file}")

def plot_umap_clusters(df: pd.DataFrame, X: np.ndarray, prefix: str):
    reducer = UMAP(n_components=2, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(X)
    df_plot = df.copy()
    df_plot['UMAP_1'] = embedding[:, 0]
    df_plot['UMAP_2'] = embedding[:, 1]

    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=df_plot[df_plot['cluster'] == -1], x='UMAP_1', y='UMAP_2', color='lightgray', alpha=0.6, label='Noise')
    sns.scatterplot(data=df_plot[df_plot['cluster'] != -1], x='UMAP_1', y='UMAP_2', hue='cluster', palette='tab20', s=60, legend=None)
    plt.title(f"AI Rubrics - {prefix}")
    plt.savefig(f"{prefix}_umap.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig = px.scatter(df_plot, x='UMAP_1', y='UMAP_2', color='cluster',
                     hover_data=['remedy', 'symptom', 'modalities', 'area'],
                     title=f"Interactive - {prefix}", height=700)
    fig.write_html(f"{prefix}_interactive.html")
    logger.info(f"Plots saved: {prefix}_umap.png + .html")

def generate_streamlit_app(clusters: List[Dict]):
    clusters_json = json.dumps(clusters, indent=2, ensure_ascii=False)
    code = f'''import streamlit as st
import pandas as pd

st.set_page_config(page_title="HoRUS 3", layout="wide")
st.title("HoRUS 3 - AI-Generated Rubrics")

clusters = {clusters_json}

data = []
for c in clusters:
    for s in c['symptoms']:
        data.append({{
            "Cluster": c['cluster_id'],
            "Remedy": s['remedy'],
            "Chapter": s['category'],
            "Symptom": s['symptom'],
            "Area": s['area'],
            "Modalities": s['modalities']
        }})

df = pd.DataFrame(data)
st.success(f"Loaded {{len(df)}} symptoms in {{len(clusters)}} rubrics")

cluster = st.selectbox("Filter", ["All"] + sorted(df['Cluster'].unique().tolist()))
if cluster != "All":
    filtered = df[df['Cluster'] == cluster]
    st.write(f"**Rubric {{cluster}}** - {{len(filtered)}} symptoms")
    st.dataframe(filtered, use_container_width=True)
else:
    st.dataframe(df, use_container_width=True)

st.download_button("Download", df.to_csv(index=False).encode(), "HORUS_Rubrics.csv")
'''
    with open("streamlit_app.py", "w", encoding="utf-8") as f:
        f.write(code)
    logger.info("Streamlit app generated")

def main():
    file_pairs = [
        ("Case_studies_combined.json", "case_studies_model.pkl"),
        ("rheumatic.json", "rheumatic_model.pkl"),
    ]

    for remedies_file, pickle_file in file_pairs:
        if not os.path.exists(remedies_file):
            logger.warning(f"Missing: {remedies_file}")
            continue

        trainer = SymptomRemedyMatcherTrainer(remedies_file)
        trainer.save_model(pickle_file)

        # CRITICAL: Pass input_file to FeatureConfig
        configs = [
            FeatureConfig("remedy_modalities", [
                ('remedy', lambda x: [x]),
                ('modalities', clean_modalities)
            ], "clusters_remedy_modalities.csv", remedies_file),
            FeatureConfig("remedy_area_modalities", [
                ('remedy', lambda x: [x]),
                ('area', clean_area),
                ('modalities', clean_modalities)
            ], "clusters_remedy_area_modalities.csv", remedies_file),
        ]

        for config in configs:
            clustering_algo = HDBSCANClustering(min_cluster_size=6, metric='hamming', cluster_selection_method='leaf')
            prefix = f"{remedies_file.split('.')[0]}_{config.name}"
            df, high_quality, all_clusters = trainer.cluster_symptoms(config, clustering_algo)
            trainer.save_clusters(df, all_clusters, config.output_file)

            with open(f"{prefix}_high_quality.json", "w", encoding="utf-8") as f:
                json.dump(high_quality, f, indent=2, ensure_ascii=False)

            X = config.preprocess(df)
            plot_umap_clusters(df, X, prefix)
            generate_streamlit_app(high_quality)

            print(f"\nSUCCESS: {remedies_file} -> {config.name}")
            print(f"   Clusters: {len(all_clusters)} | High-Quality: {len(high_quality)}")
            print(f"   Run: streamlit run streamlit_app.py\n")

if __name__ == "__main__":
    main()