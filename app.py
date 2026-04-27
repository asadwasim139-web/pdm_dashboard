# ════════════════════════════════════════════════════════════════════
#  MOTORMIND AI — Complete ML Training Pipeline
#  Team: PREDICT X
#  Project: AI-Powered Predictive Maintenance for DC Motors
#  Hardware: Arduino + Voltage/Current/Vibration Sensors
# ════════════════════════════════════════════════════════════════════
#
#  HOW TO RUN ON GOOGLE COLAB:
#    1. Open https://colab.research.google.com
#    2. File -> New notebook
#    3. Upload `final_features_vib_included.csv` to Files (left sidebar)
#    4. Copy this ENTIRE script into a single cell
#    5. Runtime -> Run all
#    6. Wait ~2 minutes -- all models will be saved as .pkl files
#    7. Download .pkl files for your Streamlit dashboard
# ════════════════════════════════════════════════════════════════════

# ─── STEP 1: Install dependencies ────────────────────────────────────
import subprocess, sys
def pip_install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

for pkg in ['xgboost', 'shap', 'joblib']:
    try:
        __import__(pkg)
    except ImportError:
        pip_install(pkg)

# ─── STEP 2: Imports ─────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, joblib, json
from datetime import datetime

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest)
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.decomposition import PCA
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

# Plot styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print('=' * 70)
print('  MOTORMIND AI — TRAINING PIPELINE')
print('  Team PREDICT X — Electrical Engineering Department')
print('=' * 70)

# ─── STEP 3: Load dataset ────────────────────────────────────────────
DATA_PATH = 'final_features_vib_included.csv'   # adjust if needed
df = pd.read_csv(DATA_PATH)

print(f'\nDataset shape:    {df.shape}')
print(f'Total samples:    {len(df)}')
print(f'Total features:   {len(df.columns) - 1}')
print(f'Missing values:   {df.isnull().sum().sum()}')
print(f'\nClass distribution:')
print(df['condition'].value_counts())

# ─── STEP 4: EDA — Class distribution & correlation ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_counts = df['condition'].value_counts()
colors = ['#10b981', '#6366f1', '#f59e0b', '#ef4444']

axes[0].bar(range(len(class_counts)), class_counts.values, color=colors)
axes[0].set_xticks(range(len(class_counts)))
axes[0].set_xticklabels(class_counts.index, rotation=20, ha='right')
axes[0].set_title('Class Distribution', fontweight='bold')
axes[0].set_ylabel('Sample Count')

axes[1].pie(class_counts.values, labels=class_counts.index, colors=colors,
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Class Balance', fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlation matrix
feature_cols_all = [c for c in df.columns if c != 'condition']
fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(df[feature_cols_all].corr(), annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=ax, annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── STEP 5: Preprocessing ───────────────────────────────────────────
le = LabelEncoder()
df['condition_encoded'] = le.fit_transform(df['condition'])
class_names = le.classes_

print('\nClass encoding:')
for i, c in enumerate(class_names):
    print(f'  {i} -> {c}')

FEATURE_COLS = [c for c in df.columns if c not in
                ['condition', 'condition_encoded']]
X = df[FEATURE_COLS].values
y = df['condition_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f'\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# ─── STEP 6: Train & compare 4 classifiers ───────────────────────────
print('\n' + '=' * 70)
print('TRAINING 4 CLASSIFIERS')
print('=' * 70)

models = {
    'Random Forest':     RandomForestClassifier(n_estimators=200, max_depth=20,
                                                random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                    random_state=42),
    'XGBoost':           xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                           learning_rate=0.1, random_state=42,
                                           eval_metric='mlogloss',
                                           use_label_encoder=False),
    'SVM (RBF)':         SVC(kernel='rbf', C=10, gamma='scale',
                             probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    print(f'\nTraining: {name}')
    Xtr = X_train_scaled if 'SVM' in name else X_train
    Xte = X_test_scaled  if 'SVM' in name else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv,
                                 scoring='accuracy', n_jobs=-1)

    results[name] = {
        'model':     model,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall':    recall_score(y_test, y_pred, average='weighted'),
        'f1':        f1_score(y_test, y_pred, average='weighted'),
        'cv_mean':   cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'y_pred':    y_pred,
    }
    print(f'  Test Accuracy: {results[name]["accuracy"]:.4f} | '
          f'F1: {results[name]["f1"]:.4f} | '
          f'CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

# Summary
summary_df = pd.DataFrame({
    'Model':     list(results.keys()),
    'Accuracy':  [r['accuracy']  for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall':    [r['recall']    for r in results.values()],
    'F1-Score':  [r['f1']        for r in results.values()],
    'CV Mean':   [r['cv_mean']   for r in results.values()],
}).round(4).sort_values('F1-Score', ascending=False).reset_index(drop=True)

print('\n' + '=' * 70)
print('MODEL COMPARISON SUMMARY')
print('=' * 70)
print(summary_df.to_string(index=False))

BEST_MODEL_NAME = summary_df.iloc[0]['Model']
best_model = results[BEST_MODEL_NAME]['model']
print(f'\nBEST MODEL: {BEST_MODEL_NAME}')

# Comparison plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(summary_df))
w = 0.2
ax.bar(x - 1.5*w, summary_df['Accuracy'],  w, label='Accuracy',  color='#10b981')
ax.bar(x - 0.5*w, summary_df['Precision'], w, label='Precision', color='#6366f1')
ax.bar(x + 0.5*w, summary_df['Recall'],    w, label='Recall',    color='#f59e0b')
ax.bar(x + 1.5*w, summary_df['F1-Score'],  w, label='F1-Score',  color='#ef4444')
ax.set_xticks(x); ax.set_xticklabels(summary_df['Model'])
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.set_ylim([0.85, 1.01]); ax.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── STEP 7: Confusion matrix & feature importance ───────────────────
y_pred_best = results[BEST_MODEL_NAME]['y_pred']
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title(f'Confusion Matrix — {BEST_MODEL_NAME}', fontweight='bold')
ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.xticks(rotation=20, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\nClassification Report — {BEST_MODEL_NAME}')
print('=' * 70)
print(classification_report(y_test, y_pred_best,
                             target_names=class_names, digits=4))

if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({'feature': FEATURE_COLS,
                       'importance': best_model.feature_importances_}
                      ).sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(fi['feature'], fi['importance'],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(fi))))
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance — {BEST_MODEL_NAME}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('\nTop 10 features:')
    print(fi.tail(10)[::-1].to_string(index=False))

# ─── STEP 8: Hyperparameter tuning ───────────────────────────────────
print('\n' + '=' * 70)
print(f'HYPERPARAMETER TUNING — {BEST_MODEL_NAME}')
print('=' * 70)

if BEST_MODEL_NAME == 'Random Forest':
    param_grid = {'n_estimators': [200, 300], 'max_depth': [15, 20, None]}
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    Xtr_g = X_train
elif BEST_MODEL_NAME == 'XGBoost':
    param_grid = {'n_estimators': [200, 300], 'max_depth': [5, 7],
                  'learning_rate': [0.05, 0.1]}
    base = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss',
                             use_label_encoder=False)
    Xtr_g = X_train
elif BEST_MODEL_NAME == 'Gradient Boosting':
    param_grid = {'n_estimators': [200, 300], 'max_depth': [3, 5, 7],
                  'learning_rate': [0.05, 0.1]}
    base = GradientBoostingClassifier(random_state=42)
    Xtr_g = X_train
else:
    param_grid = {'C': [1, 10, 50], 'gamma': ['scale', 'auto']}
    base = SVC(kernel='rbf', probability=True, random_state=42)
    Xtr_g = X_train_scaled

grid = GridSearchCV(base, param_grid, cv=5, scoring='f1_weighted',
                    n_jobs=-1, verbose=1)
grid.fit(Xtr_g, y_train)
print(f'\nBest params: {grid.best_params_}')
print(f'Best CV F1:  {grid.best_score_:.4f}')

best_model = grid.best_estimator_
Xte = X_test_scaled if 'SVM' in BEST_MODEL_NAME else X_test
y_pred_tuned = best_model.predict(Xte)
final_acc = accuracy_score(y_test, y_pred_tuned)
final_f1  = f1_score(y_test, y_pred_tuned, average='weighted')
print(f'Tuned accuracy: {final_acc:.4f} | F1: {final_f1:.4f}')

# ─── STEP 9: Health Score Engine ─────────────────────────────────────
def compute_health_score(row):
    """Health Score (0-100): higher = healthier motor."""
    v_norm   = np.clip((12.0 - row['V_mean']) / 2.0, 0, 1)
    i_norm   = np.clip(row['I_mean'] / 0.5, 0, 1)
    vib_norm = np.clip(row['VIB_rms'] / 5.0, 0, 1)
    fft_norm = np.clip(row['VIB_fft_peak'] / 50.0, 0, 1)
    stress = 0.20*v_norm + 0.25*i_norm + 0.30*vib_norm + 0.25*fft_norm
    return round((1 - stress) * 100, 2)

df['health_score'] = df.apply(compute_health_score, axis=1)
print('\nAverage Health Score per condition:')
print(df.groupby('condition')['health_score'].agg(['mean','min','max']).round(2))

# ─── STEP 10: RUL Regressor ──────────────────────────────────────────
print('\n' + '=' * 70)
print('TRAINING RUL REGRESSOR')
print('=' * 70)

RUL_BASE_HOURS = {
    'no load':                   500,
    'normal load(controlled)':   300,
    'normal load(Uncontrolled)': 150,
    'high load(controlled)':     50,
}
df['rul_target'] = df['condition'].map(RUL_BASE_HOURS) + np.random.normal(0, 15, len(df))
df['rul_target'] = df['rul_target'].clip(lower=0)

X_rul = df[FEATURE_COLS].values
y_rul = df['rul_target'].values

X_rul_train, X_rul_test, y_rul_train, y_rul_test = train_test_split(
    X_rul, y_rul, test_size=0.2, random_state=42)

rul_model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                       learning_rate=0.1, random_state=42)
rul_model.fit(X_rul_train, y_rul_train)
y_rul_pred = rul_model.predict(X_rul_test)

mae  = mean_absolute_error(y_rul_test, y_rul_pred)
rmse = np.sqrt(mean_squared_error(y_rul_test, y_rul_pred))
r2   = r2_score(y_rul_test, y_rul_pred)

print(f'  MAE:  {mae:.2f} hours')
print(f'  RMSE: {rmse:.2f} hours')
print(f'  R^2:  {r2:.4f}')

# ─── STEP 11: Anomaly Detection (Isolation Forest) ───────────────────
print('\n' + '=' * 70)
print('TRAINING ANOMALY DETECTOR (Isolation Forest)')
print('=' * 70)

normal_data = df[df['condition'] == 'no load'][FEATURE_COLS].values
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(normal_data)

df['is_anomaly'] = (iso_forest.predict(df[FEATURE_COLS].values) == -1).astype(int)
print('\nAnomalies per condition:')
print(df.groupby('condition')['is_anomaly'].agg(['sum','mean']).round(3))

# ─── STEP 12: Save all models ────────────────────────────────────────
print('\n' + '=' * 70)
print('SAVING MODELS')
print('=' * 70)

joblib.dump(best_model,  'motormind_classifier.pkl')
joblib.dump(rul_model,   'motormind_rul_model.pkl')
joblib.dump(iso_forest,  'motormind_anomaly_model.pkl')
joblib.dump(scaler,      'motormind_scaler.pkl')
joblib.dump(le,          'motormind_label_encoder.pkl')

metadata = {
    'team':            'PREDICT X',
    'project':         'MotorMind AI',
    'created':         datetime.now().isoformat(),
    'best_model_name': BEST_MODEL_NAME,
    'test_accuracy':   float(final_acc),
    'test_f1':         float(final_f1),
    'rul_mae_hours':   float(mae),
    'rul_r2':          float(r2),
    'feature_columns': FEATURE_COLS,
    'classes':         list(class_names),
    'rul_baseline_map': RUL_BASE_HOURS,
}

with open('motormind_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('\nFiles saved:')
for fname in ['motormind_classifier.pkl', 'motormind_rul_model.pkl',
              'motormind_anomaly_model.pkl', 'motormind_scaler.pkl',
              'motormind_label_encoder.pkl', 'motormind_metadata.json']:
    print(f'  ✓ {fname}')

# ─── STEP 13: Production inference function ──────────────────────────
def predict_motor_state(features_dict):
    """Production inference — input dict of features, returns prediction."""
    x = np.array([[features_dict[f] for f in FEATURE_COLS]])
    pred_idx   = best_model.predict(x)[0]
    pred_proba = best_model.predict_proba(x)[0]
    pred_class = le.inverse_transform([pred_idx])[0]
    confidence = float(pred_proba[pred_idx] * 100)
    rul_hours  = float(max(0, rul_model.predict(x)[0]))
    health     = compute_health_score(pd.Series(features_dict))
    is_anom    = bool(iso_forest.predict(x)[0] == -1)

    severity_map = {
        'high load(controlled)':     ('CRITICAL', 'Reduce load. Inspect bearings.'),
        'normal load(Uncontrolled)': ('WARNING',  'Stabilize control. Inspect within 48h.'),
        'normal load(controlled)':   ('NOMINAL',  'Operating normally.'),
        'no load':                   ('IDLE',     'Standby state.'),
    }
    severity, action = severity_map.get(pred_class, ('UNKNOWN','Check system'))

    return {
        'predicted_condition': pred_class,
        'confidence':          round(confidence, 2),
        'health_score':        health,
        'rul_hours':           round(rul_hours, 1),
        'is_anomaly':          is_anom,
        'severity':            severity,
        'recommended_action':  action,
        'class_probabilities': {cls: round(float(p)*100, 2)
                                for cls, p in zip(class_names, pred_proba)},
        'timestamp':           datetime.now().isoformat(),
    }

# DEMO
print('\n' + '=' * 70)
print('INFERENCE DEMO — predicting on a test sample')
print('=' * 70)
sample = df.iloc[100][FEATURE_COLS].to_dict()
actual = df.iloc[100]['condition']
result = predict_motor_state(sample)
print(f'\nActual class: {actual}')
print(f'\nPrediction result:')
print(json.dumps(result, indent=2))

# ─── FINAL SUMMARY ───────────────────────────────────────────────────
print('\n' + '=' * 70)
print('  TRAINING PIPELINE COMPLETE — TEAM PREDICT X')
print('=' * 70)
print(f'  Best model:     {BEST_MODEL_NAME}')
print(f'  Test accuracy:  {final_acc*100:.2f}%')
print(f'  Test F1:        {final_f1*100:.2f}%')
print(f'  RUL MAE:        {mae:.2f} hours')
print(f'  RUL R^2:        {r2:.4f}')
print(f'  Total features: {len(FEATURE_COLS)}')
print(f'  Classes:        {len(class_names)}')
print('\n  Models saved as .pkl — ready for Streamlit dashboard')
print('=' * 70)
