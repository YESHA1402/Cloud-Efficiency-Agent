from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk")
np.random.seed(42)

# ------------------- MING-Inspired Training --------------------
N, T, n_temp, n_spatial = 2000, 10, 20, 10
y_binary = np.random.choice([0, 1], size=(N,), p=[0.8, 0.2])
X_temporal = np.random.rand(N, T, n_temp)
X_spatial = np.random.rand(N, n_spatial)
failure_score = y_binary * 5 + 0.5 * np.random.randn(N)
failure_score = np.clip(failure_score, 0, None)

X_temp_train, X_temp_test, X_spatial_train, X_spatial_test, y_train, y_test, score_train, score_test = train_test_split(
    X_temporal, X_spatial, y_binary, failure_score, test_size=0.2, random_state=42)

X_temp_flat_train = X_temp_train.reshape(X_temp_train.shape[0], -1)
X_temp_flat_test = X_temp_test.reshape(X_temp_test.shape[0], -1)

pca = PCA(n_components=128)
pca.fit(X_temp_flat_train)
temp_embed_train = pca.transform(X_temp_flat_train)
temp_embed_test = pca.transform(X_temp_flat_test)

rf = RandomForestClassifier(n_estimators=128, random_state=42)
rf.fit(X_spatial_train, y_train)

def get_rf_embed(model, X):
    return np.array([tree.predict_proba(X)[:, 1] for tree in model.estimators_]).T

spatial_embed_train = get_rf_embed(rf, X_spatial_train)
spatial_embed_test = get_rf_embed(rf, X_spatial_test)

X_rank_train = np.concatenate([temp_embed_train, spatial_embed_train], axis=1)
X_rank_test = np.concatenate([temp_embed_test, spatial_embed_test], axis=1)

score_train_int = pd.qcut(score_train, q=3, labels=False, duplicates='drop')
group_train = [X_rank_train.shape[0]]
lgb_train = lgb.Dataset(X_rank_train, label=score_train_int)
lgb_train.set_group(group_train)

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'verbose': -1
}

rank_model = lgb.train(params, lgb_train, num_boost_round=100)
fixed_threshold = np.median(rank_model.predict(X_rank_test))

# -------------------- Web App --------------------
app = Flask(__name__)

def generate_plot():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(1, 201)
    y = [np.abs(np.sin(i / 20) * 100) for i in x]
    ax.plot(x, y, color="purple")
    ax.set_xlabel("Ranked Nodes")
    ax.set_ylabel("Estimated Cost")
    ax.set_title("Cost-Sensitive Thresholding (Demo)")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            lines = request.form['temporal'].strip().splitlines()
            temporal_data = [list(map(float, line.split(","))) for line in lines]
            if len(temporal_data) != 10 or any(len(row) != 20 for row in temporal_data):
                raise ValueError()
            temporal_array = np.array(temporal_data).reshape(1, -1)
        except:
            temporal_array = np.random.rand(1, T * n_temp)

        try:
            spatial_values = list(map(float, request.form['spatial'].strip().split(",")))
            if len(spatial_values) != 10:
                raise ValueError()
            spatial_array = np.array(spatial_values).reshape(1, -1)
        except:
            spatial_array = np.random.rand(1, n_spatial)

        temp_embed = pca.transform(temporal_array)
        spatial_embed = get_rf_embed(rf, spatial_array)
        final_input = np.concatenate([temp_embed, spatial_embed], axis=1)
        score = rank_model.predict(final_input)[0]
        is_faulty = score > fixed_threshold

        result = {
            "score": round(score, 3),
            "status": "FAULTY ⚠️" if is_faulty else "HEALTHY ✅",
            "color": "red" if is_faulty else "green",
            "plot": generate_plot()
        }

        # ✅ Fixed UnicodeError: use UTF-8 encoding
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(f"{score:.3f},{result['status']}\n")

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
