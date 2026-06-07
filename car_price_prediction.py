# =============================================================================
# CAR PRICE PREDICTION — CodeAlpha Data Science Internship
# Author   : CodeAlpha Intern
# Dataset  : CarPrice_Assignment.csv
# Models   : Linear Regression  &  Random Forest Regressor
# Task     : Predict car prices from specifications
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings('ignore')               # Suppress non-critical warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                           # Non-interactive backend — saves plots to disk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 2. SETUP
# ─────────────────────────────────────────────────────────────────────────────
IMAGES_DIR   = "images"
DATASET_PATH = "CarPrice_Assignment.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

os.makedirs(IMAGES_DIR, exist_ok=True)          # Create images/ folder if absent
sns.set_theme(style="whitegrid", palette="muted")

print("=" * 65)
print("    CAR PRICE PREDICTION — CodeAlpha Data Science Internship")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/8] Loading Dataset...")

if not os.path.exists(DATASET_PATH):
    print(f"\n  ✘  '{DATASET_PATH}' not found.")
    print("     Download: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction")
    print("     Place it alongside this script and re-run.\n")
    raise FileNotFoundError(f"{DATASET_PATH} is required.")

df = pd.read_csv(DATASET_PATH)
print(f"  ✔  Loaded — {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DISPLAY DATASET INFORMATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] Dataset Information")
print("─" * 65)

print("\n  ── First 5 Rows ──")
print(df.head().to_string(index=False))

print("\n  ── Shape ──")
print(f"  Rows: {df.shape[0]}   Columns: {df.shape[1]}")

print("\n  ── Column Types ──")
print(f"  {'Column':<25} {'Dtype':<12} {'Unique':<10} {'Sample'}")
print(f"  {'──────':<25} {'─────':<12} {'──────':<10} {'──────'}")
for col in df.columns:
    sample = str(df[col].iloc[0])[:20]
    print(f"  {col:<25} {str(df[col].dtype):<12} {df[col].nunique():<10} {sample}")

print("\n  ── Statistical Summary (numeric) ──")
print(df.describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5. CHECK MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Checking Missing Values...")
print("─" * 65)

missing = df.isnull().sum()
print(f"\n  {'Column':<25} {'Missing':<10} {'%'}")
print(f"  {'──────':<25} {'───────':<10} {'─'}")
for col, cnt in missing.items():
    flag = "⚠ " if cnt > 0 else "✔ "
    print(f"  {flag}{col:<23} {cnt:<10} {cnt/len(df)*100:.1f}%")

if missing.sum() == 0:
    print("\n  ✔  No missing values — dataset is clean.")
else:
    print(f"\n  Total missing: {missing.sum()} — will be handled in preprocessing.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/8] Exploratory Data Analysis (EDA)...")
print("─" * 65)

# Extract brand name from the CarName column (first word)
# e.g. "toyota corolla" → "toyota"
df['brand'] = df['CarName'].str.split().str[0].str.lower().str.strip()

# Fix known typos in brand names
brand_fixes = {
    'maxda': 'mazda', 'vokswagen': 'volkswagen',
    'vw': 'volkswagen', 'porcshce': 'porsche',
    'toyouta': 'toyota', 'Nissan': 'nissan'
}
df['brand'] = df['brand'].replace(brand_fixes)

print(f"\n  Unique brands ({df['brand'].nunique()}): {sorted(df['brand'].unique())}")
print(f"\n  Price range : ${df['price'].min():,.0f}  –  ${df['price'].max():,.0f}")
print(f"  Mean price  : ${df['price'].mean():,.0f}")
print(f"  Median price: ${df['price'].median():,.0f}")

# ── Plot 1: Price Distribution ───────────────────────────────────────────────
print("\n  Generating: Price Distribution plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with KDE
axes[0].hist(df['price'], bins=30, color='#2563EB', edgecolor='white',
             linewidth=0.5, alpha=0.85)
axes[0].set_title('Car Price Distribution (Raw)',   fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('Price ($)',  fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].axvline(df['price'].mean(),   color='#DC2626', linestyle='--', linewidth=1.6,
                label=f"Mean  ${df['price'].mean():,.0f}")
axes[0].axvline(df['price'].median(), color='#16A34A', linestyle='--', linewidth=1.6,
                label=f"Median ${df['price'].median():,.0f}")
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

# Log-scale histogram (right-skewed data often looks better log-transformed)
log_prices = np.log1p(df['price'])
axes[1].hist(log_prices, bins=30, color='#7C3AED', edgecolor='white',
             linewidth=0.5, alpha=0.85)
axes[1].set_title('Car Price Distribution (Log Scale)', fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('log(Price + 1)', fontsize=12)
axes[1].set_ylabel('Frequency',      fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle('Price Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
p1 = os.path.join(IMAGES_DIR, 'price_distribution.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔  Saved → {p1}")

# ── Plot 2: Brand vs Price ────────────────────────────────────────────────────
print("  Generating: Brand vs Price plot...")

brand_stats = (df.groupby('brand')['price']
               .median()
               .sort_values(ascending=False)
               .reset_index())

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(brand_stats['brand'], brand_stats['price'],
              color=sns.color_palette("Blues_r", len(brand_stats)),
              edgecolor='white', linewidth=0.5)

# Annotate each bar with price
for bar, price in zip(bars, brand_stats['price']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f'${price/1000:.1f}k',
            ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_title('Median Car Price by Brand', fontsize=16, fontweight='bold', pad=14)
ax.set_xlabel('Brand',         fontsize=12)
ax.set_ylabel('Median Price ($)', fontsize=12)
ax.set_xticklabels(brand_stats['brand'], rotation=45, ha='right', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
p2 = os.path.join(IMAGES_DIR, 'brand_vs_price.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔  Saved → {p2}")

# ── Plot 3: Correlation Heatmap ───────────────────────────────────────────────
print("  Generating: Correlation Heatmap...")

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
mask[np.diag_indices_from(mask)] = False

sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0, vmin=-1, vmax=1,
    linewidths=0.4, linecolor='white',
    square=True, ax=ax,
    annot_kws={'size': 7}
)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)
plt.tight_layout()
p3 = os.path.join(IMAGES_DIR, 'correlation_heatmap.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔  Saved → {p3}")

# Top numeric correlations with price
print("\n  ── Top 10 Numeric Features Correlated with Price ──")
price_corr = corr['price'].drop('price').abs().sort_values(ascending=False).head(10)
for feat, val in price_corr.items():
    bar = '█' * int(val * 30)
    print(f"  {feat:<25} {val:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/8] Data Preprocessing...")
print("─" * 65)

df_model = df.copy()

# ── Handle missing values ─────────────────────────────────────────────────────
# Numeric columns → fill with median (robust to outliers)
for col in df_model.select_dtypes(include=[np.number]).columns:
    if df_model[col].isnull().any():
        df_model[col].fillna(df_model[col].median(), inplace=True)
        print(f"  Filled numeric NaN  → {col} (median)")

# Categorical columns → fill with mode
for col in df_model.select_dtypes(include=['object']).columns:
    if df_model[col].isnull().any():
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)
        print(f"  Filled categorical NaN → {col} (mode)")

print("  ✔  Missing value handling complete")

# ── Drop non-informative columns ─────────────────────────────────────────────
# car_ID is just a row index; CarName is replaced by 'brand'
drop_cols = [c for c in ['car_ID', 'CarName'] if c in df_model.columns]
df_model.drop(columns=drop_cols, inplace=True)
print(f"  ✔  Dropped columns: {drop_cols}")

# ── Encode categorical columns ────────────────────────────────────────────────
cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
print(f"\n  Categorical columns to encode ({len(cat_cols)}): {cat_cols}")

le = LabelEncoder()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    print(f"  ✔  Encoded → {col}")

# ── Feature Selection ─────────────────────────────────────────────────────────
# Use correlation with price to keep the most informative features
corr_with_price = df_model.corr()['price'].drop('price').abs()
selected_features = corr_with_price[corr_with_price >= 0.10].index.tolist()

print(f"\n  ── Feature Selection (|r| ≥ 0.10 with price) ──")
print(f"  Selected {len(selected_features)} features out of {len(df_model.columns)-1}")
for f in selected_features:
    print(f"    • {f:<25} r = {corr_with_price[f]:.4f}")

X = df_model[selected_features].values
y = df_model['price'].values

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n  Train samples : {len(X_train)}  ({(1-TEST_SIZE)*100:.0f}%)")
print(f"  Test  samples : {len(X_test)}   ({TEST_SIZE*100:.0f}%)")

# ── Feature Scaling ───────────────────────────────────────────────────────────
# StandardScaler: zero mean, unit variance — benefits Linear Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)     # Fit on train only — prevents data leakage
X_test_sc  = scaler.transform(X_test)          # Apply same transform to test
print("  ✔  Features scaled with StandardScaler")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/8] Training Models...")
print("─" * 65)

# ── Model 1: Linear Regression ───────────────────────────────────────────────
print("\n  Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_sc, y_train)              # Uses scaled features
lr_pred  = lr_model.predict(X_test_sc)
print("  ✔  Linear Regression trained")

# ── Model 2: Random Forest Regressor ─────────────────────────────────────────
print("  Training Random Forest Regressor (100 trees)...")
rf_model = RandomForestRegressor(
    n_estimators=100,                          # 100 decision trees in the ensemble
    max_depth=None,                            # Trees grow until leaves are pure
    min_samples_split=2,
    random_state=RANDOM_STATE,
    n_jobs=-1                                  # Parallelise across all CPU cores
)
rf_model.fit(X_train, y_train)                 # RF is scale-invariant — use raw features
rf_pred  = rf_model.predict(X_test)
print("  ✔  Random Forest trained")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Evaluating Models...")
print("─" * 65)

def evaluate(name: str, y_true, y_pred_vals) -> dict:
    """Compute and print MAE, RMSE, R² for one model."""
    mae  = mean_absolute_error(y_true, y_pred_vals)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_vals))
    r2   = r2_score(y_true, y_pred_vals)
    print(f"\n  ── {name} ──")
    print(f"  MAE   : ${mae:>10,.2f}")
    print(f"  RMSE  : ${rmse:>10,.2f}")
    print(f"  R²    :  {r2:>10.4f}  ({r2*100:.1f}% variance explained)")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'pred': y_pred_vals}

lr_metrics = evaluate("Linear Regression",     y_test, lr_pred)
rf_metrics = evaluate("Random Forest",         y_test, rf_pred)

# ── Model Comparison ──────────────────────────────────────────────────────────
print("\n  ── Model Comparison Table ──")
print(f"\n  {'Metric':<8} {'Linear Regression':>20} {'Random Forest':>18} {'Winner':>10}")
print(f"  {'──────':<8} {'─────────────────':>20} {'─────────────':>18} {'──────':>10}")

metrics_list = [('MAE', True), ('RMSE', True), ('R2', False)]
winners = {}
for metric, lower_is_better in metrics_list:
    lv = lr_metrics[metric]
    rv = rf_metrics[metric]
    if lower_is_better:
        winner = "Linear Reg." if lv < rv else "Random Forest"
    else:
        winner = "Linear Reg." if lv > rv else "Random Forest"
    winners[metric] = winner
    fmt_l = f"${lv:,.2f}" if metric != 'R2' else f"{lv:.4f}"
    fmt_r = f"${rv:,.2f}" if metric != 'R2' else f"{rv:.4f}"
    print(f"  {metric:<8} {fmt_l:>20} {fmt_r:>18} {winner:>10}")

# Overall winner: highest R² + lowest RMSE
rf_score = (rf_metrics['R2'] > lr_metrics['R2']) + (rf_metrics['RMSE'] < lr_metrics['RMSE'])
best_model_name = "Random Forest" if rf_score >= 1 else "Linear Regression"

print(f"\n  ┌─────────────────────────────────────────────────────┐")
print(f"  │  🏆  BEST MODEL: {best_model_name:<35} │")
print(f"  │      R² = {max(lr_metrics['R2'], rf_metrics['R2']):.4f}   RMSE = ${min(lr_metrics['RMSE'], rf_metrics['RMSE']):,.2f}               │")
print(f"  └─────────────────────────────────────────────────────┘")

# Feature importances (Random Forest only)
print("\n  ── Top 10 Feature Importances (Random Forest) ──")
importances = rf_model.feature_importances_
feat_imp = sorted(zip(selected_features, importances), key=lambda x: -x[1])[:10]
for feat, imp in feat_imp:
    bar = '█' * int(imp * 50)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALISE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] Saving Result Plots...")
print("─" * 65)

# ── Actual vs Predicted — side by side ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, metrics, color, title in [
    (axes[0], lr_metrics, '#2563EB', 'Linear Regression'),
    (axes[1], rf_metrics, '#16A34A', 'Random Forest'),
]:
    preds = metrics['pred']
    r2    = metrics['R2']
    ax.scatter(y_test, preds, color=color, alpha=0.6, s=50,
               edgecolors='white', linewidth=0.3)
    lims = [min(y_test.min(), preds.min()) * 0.95,
            max(y_test.max(), preds.max()) * 1.05]
    ax.plot(lims, lims, color='#DC2626', linewidth=1.8, linestyle='--',
            label='Perfect fit')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f'{title}\nR² = {r2:.4f}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Actual Price ($)',    fontsize=11)
    ax.set_ylabel('Predicted Price ($)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.35)

plt.suptitle('Actual vs Predicted Car Prices', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
avp_path = os.path.join(IMAGES_DIR, 'actual_vs_predicted.png')
plt.savefig(avp_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔  Saved → {avp_path}")

# ── Residuals plots ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metrics, color, title in [
    (axes[0], lr_metrics, '#2563EB', 'Linear Regression'),
    (axes[1], rf_metrics, '#16A34A', 'Random Forest'),
]:
    residuals = y_test - metrics['pred']
    ax.scatter(metrics['pred'], residuals, color=color, alpha=0.6, s=45,
               edgecolors='white', linewidth=0.3)
    ax.axhline(0, color='#DC2626', linewidth=1.8, linestyle='--')
    ax.set_title(f'{title} — Residuals', fontsize=13, fontweight='bold')
    ax.set_xlabel('Fitted Price ($)',             fontsize=11)
    ax.set_ylabel('Residual (Actual − Predicted)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.35)

plt.suptitle('Residuals vs Fitted Values', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
res_path = os.path.join(IMAGES_DIR, 'residuals_plot.png')
plt.savefig(res_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔  Saved → {res_path}")

# ── Feature Importance Bar Chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
feat_names = [f[0] for f in feat_imp]
feat_vals  = [f[1] for f in feat_imp]

bars = ax.barh(feat_names[::-1], feat_vals[::-1],
               color=sns.color_palette("Blues_r", len(feat_imp)),
               edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, feat_vals[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

ax.set_title('Top 10 Feature Importances (Random Forest)',
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Importance Score', fontsize=11)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
fi_path = os.path.join(IMAGES_DIR, 'feature_importances.png')
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔  Saved → {fi_path}")

# ── Model Comparison Bar Chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
models  = ['Linear Reg.', 'Random Forest']
colors  = ['#2563EB',     '#16A34A']

for ax, (metric, lower_is_better) in zip(axes, metrics_list):
    vals = [lr_metrics[metric], rf_metrics[metric]]
    bars = ax.bar(models, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.4)
    best_idx = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
    bars[best_idx].set_edgecolor('#FFD700')
    bars[best_idx].set_linewidth(2.5)
    for bar, val in zip(bars, vals):
        label = f'${val:,.0f}' if metric != 'R2' else f'{val:.4f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                label, ha='center', fontsize=10, fontweight='bold')
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_ylim(0, max(vals) * 1.18)

plt.suptitle('Model Comparison: Linear Regression vs Random Forest',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
cmp_path = os.path.join(IMAGES_DIR, 'model_comparison.png')
plt.savefig(cmp_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✔  Saved → {cmp_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("    PROJECT COMPLETE — FINAL SUMMARY")
print("=" * 65)
print(f"\n  Dataset        : {DATASET_PATH}  ({df.shape[0]} rows)")
print(f"  Features used  : {len(selected_features)}")
print(f"  Train/Test     : {(1-TEST_SIZE)*100:.0f}% / {TEST_SIZE*100:.0f}%")
print(f"\n  {'Model':<22} {'MAE':>12} {'RMSE':>12} {'R²':>8}")
print(f"  {'─────':<22} {'───':>12} {'────':>12} {'──':>8}")
for m in [lr_metrics, rf_metrics]:
    print(f"  {m['model']:<22} ${m['MAE']:>10,.2f} ${m['RMSE']:>10,.2f} {m['R2']:>8.4f}")

print(f"\n  🏆  Best Model : {best_model_name}")
print(f"\n  Plots saved to ./{IMAGES_DIR}/")
print(f"    • price_distribution.png")
print(f"    • brand_vs_price.png")
print(f"    • correlation_heatmap.png")
print(f"    • actual_vs_predicted.png")
print(f"    • residuals_plot.png")
print(f"    • feature_importances.png")
print(f"    • model_comparison.png")
print("\n  ✔  All tasks completed successfully!")
print("=" * 65)