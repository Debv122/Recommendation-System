"""
Recommendation System for getINNOtized
A comprehensive recommendation system leveraging user behavior and preferences
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import warnings
warnings.filterwarnings('ignore')

# Runtime/memory configuration (tunable via env vars; safe defaults for Colab)
MAX_ROWS_EVENTS = int(os.getenv('MAX_ROWS_EVENTS', '300000'))
MAX_ROWS_ITEM_PROPERTIES = int(os.getenv('MAX_ROWS_ITEM_PROPERTIES', '200000'))
EVENT_COLUMNS = ['timestamp', 'visitorid', 'event', 'itemid', 'transactionid']
TOP_USERS_LIMIT = int(os.getenv('TOP_USERS_LIMIT', '300'))
TOP_ITEMS_LIMIT = int(os.getenv('TOP_ITEMS_LIMIT', '300'))
COLLAB_SIMILAR_USERS = int(os.getenv('COLLAB_SIMILAR_USERS', '5'))
CONTENT_BASED_ITEM_LIMIT = int(os.getenv('CONTENT_BASED_ITEM_LIMIT', '5000'))
N_TEST_USERS = int(os.getenv('N_TEST_USERS', '50'))
EVAL_TOP_N = int(os.getenv('EVAL_TOP_N', '5'))
EVAL_USER_LIMIT = int(os.getenv('EVAL_USER_LIMIT', '30'))
SKIP_HEAVY_ANALYTICS = os.getenv('SKIP_HEAVY_ANALYTICS', '1') == '1'

# Ensure UTF-8 console output on Windows to prevent UnicodeEncodeError in prints
try:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

class RecommendationSystem:
    def __init__(self):
        self.events_df = None
        self.category_tree_df = None
        self.item_properties_df = None
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        
    def load_data(self):
        """Load and preprocess all data files"""
        print("Loading data files...")
        
        # Prefer cleaned datasets if available
        events_path = 'events_cleaned.csv' if os.path.exists('events_cleaned.csv') else 'events.csv'
        cat_path = 'category_tree_cleaned.csv' if os.path.exists('category_tree_cleaned.csv') else 'category_tree.csv'
        
        # Load events data with sampling for large files
        print("Loading events data...")
        try:
            # Try to load with row cap and selected columns
            self.events_df = pd.read_csv(events_path, usecols=[c for c in EVENT_COLUMNS if c != 'transactionid' or 'transactionid' in pd.read_csv(events_path, nrows=1).columns], nrows=MAX_ROWS_EVENTS)
        except MemoryError:
            print("Memory error, loading smaller sample of events data...")
            self.events_df = pd.read_csv(events_path, usecols=[c for c in EVENT_COLUMNS if c != 'transactionid' or 'transactionid' in pd.read_csv(events_path, nrows=1).columns], nrows=max(50000, MAX_ROWS_EVENTS // 2))
        
        print(f"Events data shape: {self.events_df.shape}")
        
        # Load category tree
        self.category_tree_df = pd.read_csv(cat_path)
        print(f"Category tree shape: {self.category_tree_df.shape}")
        
        # Load item properties (prefer cleaned file, else combine known parts)
        if os.path.exists('item_properties_cleaned.csv'):
            self.item_properties_df = pd.read_csv('item_properties_cleaned.csv', usecols=['itemid', 'property', 'value'], nrows=MAX_ROWS_ITEM_PROPERTIES)
            print(f"Item properties shape: {self.item_properties_df.shape}")
        else:
            try:
                if os.path.exists('item_properties.csv'):
                    self.item_properties_df = pd.read_csv('item_properties.csv', usecols=['itemid', 'property', 'value'], nrows=MAX_ROWS_ITEM_PROPERTIES)
                else:
                    frames = []
                    remaining = MAX_ROWS_ITEM_PROPERTIES
                    if os.path.exists('item_properties_part1.1.csv') and remaining > 0:
                        n1 = remaining // 2 if os.path.exists('item_properties_part2.csv') else remaining
                        frames.append(pd.read_csv('item_properties_part1.1.csv', usecols=['itemid', 'property', 'value'], nrows=n1))
                        remaining -= n1
                    if os.path.exists('item_properties_part2.csv') and remaining > 0:
                        frames.append(pd.read_csv('item_properties_part2.csv', usecols=['itemid', 'property', 'value'], nrows=remaining))
                    self.item_properties_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                print(f"Item properties shape: {self.item_properties_df.shape}")
            except Exception:
                print("Could not load item properties, using empty DataFrame")
                self.item_properties_df = pd.DataFrame()
        
        # Convert timestamp to datetime
        self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'], unit='ms', errors='coerce')
        self.events_df = self.events_df.dropna(subset=['timestamp'])
        
        print("Data loading completed!")
        
    def create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        print("Creating user-item matrix...")
        
        # Create interaction matrix (view events)
        view_events = self.events_df[self.events_df['event'] == 'view']
        
        # Sample data for memory efficiency (use top users and items)
        print("Sampling data for memory efficiency...")
        
        # Get top users by activity
        user_activity = view_events.groupby('visitorid').size()
        top_users = user_activity.nlargest(TOP_USERS_LIMIT).index.tolist()
        
        # Get top items by popularity
        item_popularity = view_events.groupby('itemid').size()
        top_items = item_popularity.nlargest(TOP_ITEMS_LIMIT).index.tolist()
        
        # Filter to top users and items
        filtered_events = view_events[
            (view_events['visitorid'].isin(top_users)) &
            (view_events['itemid'].isin(top_items))
        ]
        
        # Create user-item matrix with interaction counts
        self.user_item_matrix = filtered_events.groupby(['visitorid', 'itemid']).size().unstack(fill_value=0)
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"Using top {len(top_users)} users and {len(top_items)} items")
        return self.user_item_matrix
    
    def collaborative_filtering_recommendations(self, user_id, n_recommendations=5):
        """Collaborative filtering based recommendations"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Calculate user similarity
        user_similarities = cosine_similarity([self.user_item_matrix.loc[user_id]])
        
        # Find similar users
        similar_users = np.argsort(user_similarities[0])[::-1][1:COLLAB_SIMILAR_USERS+1]
        
        # Get items liked by similar users
        recommendations = []
        for similar_user_idx in similar_users:
            similar_user_id = self.user_item_matrix.index[similar_user_idx]
            user_items = self.user_item_matrix.loc[similar_user_id]
            liked_items = user_items[user_items > 0].index.tolist()
            recommendations.extend(liked_items)
        
        # Remove items user already interacted with
        user_items = self.user_item_matrix.loc[user_id]
        user_interacted_items = user_items[user_items > 0].index.tolist()
        recommendations = [item for item in recommendations if item not in user_interacted_items]
        
        # Return top N recommendations
        return list(dict.fromkeys(recommendations))[:n_recommendations]
    
    def content_based_recommendations(self, user_id, n_recommendations=5):
        """Content-based recommendations using item properties"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's interacted items
        user_items = self.user_item_matrix.loc[user_id]
        user_interacted_items = user_items[user_items > 0].index.tolist()
        
        if not user_interacted_items:
            return []
        
        # Get item properties for user's items
        user_item_properties = self.item_properties_df[
            self.item_properties_df['itemid'].isin(user_interacted_items)
        ]
        
        if user_item_properties.empty:
            return []
        
        # Create item profile based on user preferences
        user_profile = user_item_properties.groupby('property')['value'].apply(
            lambda x: ' '.join(x.astype(str))
        ).to_dict()
        
        # Find similar items based on properties
        all_items = self.item_properties_df['itemid'].unique()
        if len(all_items) > CONTENT_BASED_ITEM_LIMIT:
            all_items = all_items[:CONTENT_BASED_ITEM_LIMIT]
        recommendations = []
        
        for item_id in all_items:
            if item_id not in user_interacted_items:
                item_properties = self.item_properties_df[
                    self.item_properties_df['itemid'] == item_id
                ]
                
                if not item_properties.empty:
                    item_profile = item_properties.groupby('property')['value'].apply(
                        lambda x: ' '.join(x.astype(str))
                    ).to_dict()
                    
                    # Calculate similarity (simple overlap for now)
                    similarity = len(set(user_profile.keys()) & set(item_profile.keys()))
                    if similarity > 0:
                        recommendations.append((item_id, similarity))
        
        # Sort by similarity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in recommendations[:n_recommendations]]
    
    def popularity_based_recommendations(self, n_recommendations=5):
        """Popularity-based recommendations"""
        # Count item interactions
        item_popularity = self.events_df[self.events_df['event'] == 'view']['itemid'].value_counts()
        return item_popularity.head(n_recommendations).index.tolist()
    
    def category_based_recommendations(self, user_id, n_recommendations=5):
        """Category-based recommendations"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's interacted items
        user_items = self.user_item_matrix.loc[user_id]
        user_interacted_items = user_items[user_items > 0].index.tolist()
        
        if not user_interacted_items:
            return []
        
        # Get categories of user's items
        user_item_categories = self.item_properties_df[
            (self.item_properties_df['itemid'].isin(user_interacted_items)) &
            (self.item_properties_df['property'] == 'categoryid')
        ]['value'].unique()
        
        # Find items in same categories
        category_items = self.item_properties_df[
            (self.item_properties_df['property'] == 'categoryid') &
            (self.item_properties_df['value'].isin(user_item_categories))
        ]['itemid'].unique()
        
        # Remove items user already interacted with
        recommendations = [item for item in category_items if item not in user_interacted_items]
        
        return recommendations[:n_recommendations]
    
    def hybrid_recommendations(self, user_id, n_recommendations=5):
        """Hybrid recommendations combining multiple approaches"""
        collab_recs = self.collaborative_filtering_recommendations(user_id, n_recommendations)
        content_recs = self.content_based_recommendations(user_id, n_recommendations)
        category_recs = self.category_based_recommendations(user_id, n_recommendations)
        
        # Combine and weight recommendations
        all_recs = collab_recs + content_recs + category_recs
        rec_counts = {}
        
        for rec in all_recs:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        # Sort by frequency and return top N
        sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_recs[:n_recommendations]]
    
    def analyze_user_behavior(self):
        """Analyze user behavior patterns"""
        print("\n=== User Behavior Analysis ===")
        
        # Event distribution
        event_counts = self.events_df['event'].value_counts()
        print(f"Event distribution:\n{event_counts}")
        
        # User activity analysis
        user_activity = self.events_df.groupby('visitorid').size()
        print(f"\nUser activity statistics:")
        print(f"Average events per user: {user_activity.mean():.2f}")
        print(f"Median events per user: {user_activity.median():.2f}")
        print(f"Most active user: {user_activity.idxmax()} with {user_activity.max()} events")
        
        # Item popularity analysis
        item_popularity = self.events_df[self.events_df['event'] == 'view']['itemid'].value_counts()
        print(f"\nItem popularity statistics:")
        print(f"Most viewed item: {item_popularity.index[0]} with {item_popularity.iloc[0]} views")
        print(f"Average views per item: {item_popularity.mean():.2f}")
        
        return {
            'event_counts': event_counts,
            'user_activity': user_activity,
            'item_popularity': item_popularity
        }
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in user behavior"""
        print("\n=== Temporal Pattern Analysis ===")
        
        # Hourly activity
        self.events_df['hour'] = self.events_df['timestamp'].dt.hour
        hourly_activity = self.events_df.groupby('hour').size()
        
        print(f"Peak activity hour: {hourly_activity.idxmax()} ({hourly_activity.max()} events)")
        print(f"Lowest activity hour: {hourly_activity.idxmin()} ({hourly_activity.min()} events)")
        
        # Daily activity
        self.events_df['day_of_week'] = self.events_df['timestamp'].dt.day_name()
        daily_activity = self.events_df.groupby('day_of_week').size()
        
        print(f"\nMost active day: {daily_activity.idxmax()} ({daily_activity.max()} events)")
        print(f"Least active day: {daily_activity.idxmin()} ({daily_activity.min()} events)")
        
        return {
            'hourly_activity': hourly_activity,
            'daily_activity': daily_activity
        }
    
    def evaluate_recommendations(self, test_users=None, n_test_users=100, top_n=10, min_interactions=3, eval_user_limit=100):
        """Evaluate recommendation quality
        - Computes average precision, recall, and F1 across evaluated users
        - Computes Top-N Accuracy (HitRate@N): fraction of users with at least one relevant item in top-N
        """
        if test_users is None:
            # Sample users for evaluation
            active_users = self.user_item_matrix.sum(axis=1).sort_values(ascending=False)
            test_users = active_users.head(n_test_users).index.tolist()
        
        print(f"\n=== Recommendation Evaluation ===")
        print(f"Evaluating recommendations for {len(test_users)} users")
        
        # Calculate metrics
        precision_scores = []
        recall_scores = []
        hits_topn = 0
        evaluated_users = 0
        
        for user_id in test_users:
            # Get actual user interactions
            user_items_series = self.user_item_matrix.loc[user_id]
            user_interactions = set(user_items_series[user_items_series > 0].index)
            
            if len(user_interactions) < min_interactions:
                continue
            
            # Get recommendations (top-N)
            recommendations = set(self.hybrid_recommendations(user_id, top_n))
            
            # Only evaluate users for whom we produced recommendations
            if recommendations:
                relevant_recommended = len(user_interactions & recommendations)
                precision = relevant_recommended / len(recommendations) if len(recommendations) else 0
                recall = relevant_recommended / len(user_interactions) if len(user_interactions) else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                
                if relevant_recommended > 0:
                    hits_topn += 1
                
                evaluated_users += 1
                
                if evaluated_users >= eval_user_limit:
                    break
        
        avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        topn_accuracy = (hits_topn / evaluated_users) if evaluated_users > 0 else 0.0
        
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"F1 Score: {f1_score:.3f}")
        print(f"Top-{top_n} Accuracy (HitRate@{top_n}): {topn_accuracy:.3f} ({hits_topn}/{evaluated_users} users)")
        
        return {
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'f1_score': f1_score,
            'topn_accuracy': topn_accuracy,
            'evaluated_users': evaluated_users,
            'top_n': top_n,
        }

def clean_and_save_datasets():
    """Data Cleaning + Save Cleaned Files"""
    print("\n" + "="*60)
    print("Data Cleaning + Save Cleaned Files")
    print("="*60)

    dtype_events = {
        "timestamp": str,
        "visitorid": str,
        "event": str,
        "itemid": str,
        "transactionid": str
    }
    dtype_cat = {"categoryid": str, "parentid": str}
    dtype_props = {"itemid": str, "property": str, "value": str}

    # -----------------------
    # Step 1: Load raw datasets
    # -----------------------
    events = pd.read_csv("events.csv", dtype=dtype_events, usecols=[c for c in EVENT_COLUMNS if c != 'transactionid' or 'transactionid' in pd.read_csv('events.csv', nrows=1).columns], nrows=MAX_ROWS_EVENTS)
    category_tree = pd.read_csv("category_tree.csv", dtype=dtype_cat)

    # Support multiple item properties file layouts
    if os.path.exists("item_properties.csv"):
        item_props = pd.read_csv("item_properties.csv", dtype=dtype_props, usecols=["itemid", "property", "value"], nrows=MAX_ROWS_ITEM_PROPERTIES)
    else:
        frames = []
        if os.path.exists("item_properties_part1.1.csv"):
            n1 = MAX_ROWS_ITEM_PROPERTIES // 2 if os.path.exists("item_properties_part2.csv") else MAX_ROWS_ITEM_PROPERTIES
            frames.append(pd.read_csv("item_properties_part1.1.csv", dtype=dtype_props, usecols=["itemid", "property", "value"], nrows=n1))
        if os.path.exists("item_properties_part2.csv"):
            n2 = MAX_ROWS_ITEM_PROPERTIES - (frames[0].shape[0] if frames else 0)
            if n2 > 0:
                frames.append(pd.read_csv("item_properties_part2.csv", dtype=dtype_props, usecols=["itemid", "property", "value"], nrows=n2))
        item_props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(dtype_props.keys()))

    print("Initial shapes:")
    print("Events:", events.shape)
    print("Category Tree:", category_tree.shape)
    print("Item Properties:", item_props.shape)

    # -----------------------
    # Step 2: Clean abnormal values
    # -----------------------

    # Clean numeric-looking property values in item_properties
    def clean_numeric(val):
        if isinstance(val, str) and val.startswith("n"):
            try:
                return float(val[1:])
            except Exception:
                return np.nan
        return val

    if not item_props.empty:
        item_props["value"] = item_props["value"].apply(clean_numeric)
        item_props = item_props.dropna(subset=["itemid", "property", "value"])

    # -----------------------
    # Step 3: Outlier detection (abnormal users)
    # -----------------------
    def build_user_features(events_df):
        feats = []
        if events_df.empty:
            return pd.DataFrame(columns=["visitorid", "total_events", "views", "adds", "buys", "add_rate", "conv_rate"]).set_index("visitorid")
        for uid, g in events_df.groupby("visitorid"):
            total = len(g)
            views = (g["event"] == "view").sum()
            adds = (g["event"] == "addtocart").sum()
            buys = (g["event"] == "transaction").sum()
            feats.append({
                "visitorid": uid,
                "total_events": total,
                "views": views,
                "adds": adds,
                "buys": buys,
                "add_rate": adds/total if total > 0 else 0,
                "conv_rate": buys/total if total > 0 else 0
            })
        return pd.DataFrame(feats).set_index("visitorid")

    user_feats = build_user_features(events)

    if not user_feats.empty:
        iso = IsolationForest(contamination=0.02, random_state=42)
        user_feats["outlier"] = (iso.fit_predict(user_feats) == -1).astype(int)
        outlier_users = user_feats[user_feats["outlier"] == 1].index
        events = events[~events["visitorid"].isin(outlier_users)]
        print(f"Removed {len(outlier_users)} abnormal users.")
    else:
        print("No user features to build after cleaning. Skipping outlier detection.")

    # -----------------------
    # Step 4: Save cleaned datasets
    # -----------------------
    events.to_csv("events_cleaned.csv", index=False)
    category_tree.to_csv("category_tree_cleaned.csv", index=False)
    item_props.to_csv("item_properties_cleaned.csv", index=False)

    print("Cleaned datasets saved.")

def build_user_features_from_events(events_df, users_limit=None):
    """Build per-user feature matrix for anomaly detection from events DataFrame."""
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    df = events_df
    if users_limit is not None and users_limit > 0:
        active_counts = df.groupby('visitorid').size().nlargest(users_limit)
        df = df[df['visitorid'].isin(active_counts.index)]
    grouped = df.groupby('visitorid')
    total_events = grouped.size().rename('total_events')
    views = grouped.apply(lambda g: (g['event'] == 'view').sum()).rename('views')
    adds = grouped.apply(lambda g: (g['event'] == 'addtocart').sum()).rename('adds')
    buys = grouped.apply(lambda g: (g['event'] == 'transaction').sum()).rename('buys')
    unique_items_viewed = grouped.apply(lambda g: g.loc[g['event'] == 'view', 'itemid'].nunique()).rename('unique_items_viewed')
    unique_items_bought = grouped.apply(lambda g: g.loc[g['event'] == 'transaction', 'itemid'].nunique()).rename('unique_items_bought')
    days_active = grouped.apply(lambda g: g['timestamp'].dt.date.nunique()).rename('days_active')
    features = pd.concat([
        total_events, views, adds, buys, unique_items_viewed, unique_items_bought, days_active
    ], axis=1).fillna(0)
    features['add_rate'] = features['adds'] / features['total_events'].replace(0, np.nan)
    features['buy_rate'] = features['buys'] / features['total_events'].replace(0, np.nan)
    features['view_to_add_rate'] = features['adds'] / features['views'].replace(0, np.nan)
    features['add_to_buy_rate'] = features['buys'] / features['adds'].replace(0, np.nan)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def detect_abnormal_users_from_events(events_df, contamination=None, users_limit=None):
    """Detect abnormal users using IsolationForest; returns dict with features and flags."""
    if contamination is None:
        contamination = float(os.getenv('ANOMALY_CONTAMINATION', '0.02'))
    if users_limit is None:
        users_limit = int(os.getenv('ANOMALY_MAX_USERS', '50000'))
    features = build_user_features_from_events(events_df, users_limit=users_limit)
    if features.empty:
        return {
            'user_features': features,
            'labels': pd.Series(dtype=int),
            'scores': pd.Series(dtype=float),
            'flagged_user_ids': []
        }
    feature_columns = [
        'total_events','views','adds','buys',
        'unique_items_viewed','unique_items_bought','days_active',
        'add_rate','buy_rate','view_to_add_rate','add_to_buy_rate'
    ]
    X = features[feature_columns].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X_scaled)  # -1 outlier, 1 normal
    scores = -model.decision_function(X_scaled)  # higher = more anomalous
    features['outlier'] = (labels == -1).astype(int)
    features['anomaly_score'] = scores
    flagged_ids = features.index[features['outlier'] == 1].tolist()
    return {
        'user_features': features,
        'labels': pd.Series(labels, index=features.index),
        'scores': pd.Series(scores, index=features.index),
        'flagged_user_ids': flagged_ids
    }


def evaluate_anomaly_quality(events_df, user_features):
    """Evaluate anomaly detection quality using business-centric metrics and separation."""
    if user_features is None or user_features.empty:
        print("No user features available for evaluation.")
        return {}
    flagged_ids = set(user_features.index[user_features['outlier'] == 1])
    total_events = len(events_df)
    events_flagged = events_df[events_df['visitorid'].isin(flagged_ids)] if total_events else pd.DataFrame()
    events_normal = events_df[~events_df['visitorid'].isin(flagged_ids)] if total_events else pd.DataFrame()

    def conversion_rate(df):
        if df is None or df.empty:
            return 0.0
        buys = (df['event'] == 'transaction').sum()
        return buys / len(df)

    conv_flagged = float(conversion_rate(events_flagged))
    conv_normal = float(conversion_rate(events_normal))
    share_flagged_events = float(len(events_flagged) / total_events) if total_events else 0.0

    quality_score = 0.7 * max(0.0, (conv_normal - conv_flagged) / (conv_normal + 1e-9)) \
        + 0.3 * max(0.0, 1.0 - min(1.0, share_flagged_events / 0.25))

    silhouette = None
    try:
        num_flagged = int(user_features['outlier'].sum())
        num_normal = int(len(user_features) - num_flagged)
        if num_flagged > 1 and num_normal > 1:
            feat_cols = [
                'total_events','views','adds','buys',
                'unique_items_viewed','unique_items_bought','days_active',
                'add_rate','buy_rate','view_to_add_rate','add_to_buy_rate'
            ]
            X = StandardScaler().fit_transform(user_features[feat_cols].astype(float).values)
            y = user_features['outlier'].astype(int).values
            silhouette = float(silhouette_score(X, y))
    except Exception:
        silhouette = None

    results = {
        'num_flagged_users': int(user_features['outlier'].sum()),
        'share_flagged_users': float(user_features['outlier'].mean()) if len(user_features) else 0.0,
        'share_flagged_events': share_flagged_events,
        'conversion_rate_flagged': conv_flagged,
        'conversion_rate_normal': conv_normal,
        'quality_score': float(quality_score),
        'silhouette_separation': silhouette,
    }

    print("\n=== Anomaly Detection Evaluation ===")
    for k, v in results.items():
        print(f"{k}: {v}")
    return results

def main():
    """Main function to run the recommendation system"""
    print("=== getINNOtized Recommendation System ===")
    
    # Data Cleaning + Save Cleaned Files (replaces business questions section)
    clean_and_save_datasets()
    
    # Initialize recommendation system
    rs = RecommendationSystem()
    
    # Load data
    rs.load_data()
    
    # Create user-item matrix
    rs.create_user_item_matrix()
 
    # Abnormal user detection and evaluation
    detection = detect_abnormal_users_from_events(
        rs.events_df,
        contamination=float(os.getenv('ANOMALY_CONTAMINATION', '0.02')),
        users_limit=int(os.getenv('ANOMALY_MAX_USERS', '50000'))
    )
    evaluate_anomaly_quality(rs.events_df, detection['user_features'])
    
    # Import and run analytics (optionally skip heavy modules on constrained runtimes)
    if not SKIP_HEAVY_ANALYTICS:
        try:
            from analytics_report import AnalyticsReport
            from business_analytics import BusinessAnalytics
            
            print("\n" + "="*80)
            print("RUNNING COMPREHENSIVE ANALYTICS REPORT")
            print("="*80)
            
            # Create analytics report
            analytics = AnalyticsReport(
                rs.events_df, 
                rs.category_tree_df, 
                rs.item_properties_df, 
                rs.user_item_matrix
            )
            
            # Generate comprehensive report
            analytics_results = analytics.generate_comprehensive_report()
            
            print("\n" + "="*80)
            print("RUNNING BUSINESS ANALYTICS REPORT")
            print("="*80)
            
            # Create business analytics report
            business_analytics = BusinessAnalytics(
                rs.events_df, 
                rs.category_tree_df, 
                rs.item_properties_df, 
                rs.user_item_matrix
            )
            
            # Generate business report
            business_results = business_analytics.generate_business_report()
            
        except ImportError:
            print("Analytics modules not available, running basic analyses...")
            # Perform basic analyses
            rs.analyze_user_behavior()
            rs.analyze_temporal_patterns()
    else:
        print("Skipping heavy analytics modules for memory-constrained runtime (set SKIP_HEAVY_ANALYTICS=0 to enable).")
        rs.analyze_user_behavior()
        rs.analyze_temporal_patterns()
    
    # Evaluate recommendations
    rs.evaluate_recommendations(n_test_users=N_TEST_USERS, top_n=EVAL_TOP_N, eval_user_limit=EVAL_USER_LIMIT)
    
    # Example recommendations
    print("\n=== Example Recommendations ===")
    sample_users = rs.user_item_matrix.head(3).index.tolist()
    
    for user_id in sample_users:
        print(f"\nRecommendations for user {user_id}:")
        
        # Collaborative filtering
        collab_recs = rs.collaborative_filtering_recommendations(user_id, 3)
        print(f"  Collaborative: {collab_recs}")
        
        # Content-based
        content_recs = rs.content_based_recommendations(user_id, 3)
        print(f"  Content-based: {content_recs}")
        
        # Hybrid
        hybrid_recs = rs.hybrid_recommendations(user_id, 3)
        print(f"  Hybrid: {hybrid_recs}")
    
    print("\n=== Recommendation System Analysis Complete ===")

if __name__ == "__main__":
    main()
