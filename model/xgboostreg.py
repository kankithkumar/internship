import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ephem
import joblib
import json

from category_encoders import TargetEncoder
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load data
df = pd.read_excel("combined.xlsx")
df = df.drop(columns=["Unnamed: 3"], errors='ignore')
df = df.dropna(subset=["Close", "Date"])

# Date features
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Lunar features
def moon_phase(date):
    moon = ephem.Moon()
    moon.compute(date)
    return moon.phase

df['LunarPhase'] = df['Date'].apply(moon_phase)
df['NearFullMoon'] = (df['LunarPhase'] > 95).astype(int)
df['NearNewMoon'] = (df['LunarPhase'] < 5).astype(int)

# Categorical & interaction features
categorical_cols = ['Nakshatra', 'Tithi', 'Karna', 'Yoga']
df['Nak_Tithi'] = df['Nakshatra'].astype(str) + "_" + df['Tithi'].astype(str)
df['Yoga_Karna'] = df['Yoga'].astype(str) + "_" + df['Karna'].astype(str)
interaction_cols = ['Nak_Tithi', 'Yoga_Karna']
date_features = ['DayOfWeek']
moon_features = ['LunarPhase', 'NearFullMoon', 'NearNewMoon']

# Rolling stats and lag
df['RollingMean_3'] = df['Close'].rolling(window=3).mean()
df['RollingStd_3'] = df['Close'].rolling(window=3).std()
df['PrevDayChange'] = df['Close'].pct_change().shift(1)

# Create and shift PercentageDiff target
df['PercentageDiff'] = df['Close'].pct_change() * 100
df['PercentageDiff'] = df['PercentageDiff'].shift(-1)

# Group mean encoding
for col in categorical_cols:
    agg = df.groupby(col)['Close'].mean().rename(f"{col}_close_mean")
    df = df.merge(agg, on=col, how='left')

extra_features = ['RollingMean_3', 'RollingStd_3', 'PrevDayChange'] + [f"{col}_close_mean" for col in categorical_cols]
feature_cols = categorical_cols + interaction_cols + date_features + moon_features + extra_features

# Drop NA rows
df = df.dropna(subset=feature_cols + ['PercentageDiff'])

# Encode categoricals
encoder = TargetEncoder(cols=categorical_cols + interaction_cols)
df[categorical_cols + interaction_cols] = encoder.fit_transform(df[categorical_cols + interaction_cols], df['PercentageDiff'])

# Prepare input and target
X = df[feature_cols].values
y = df['PercentageDiff'].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'min_child_weight': [1, 3, 5]
}

xgb = XGBRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=25,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)

best_model = random_search.best_estimator_
print("✅ Best Parameters Found:\n", random_search.best_params_)

# Cross-validation with tuned model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
r2_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)

    print(f"\nFold {fold} MSE: {mse:.2f}, R² Score: {r2:.4f}")

print(f"\n✅ Average MSE after Tuning: {np.mean(mse_scores):.2f}")
print(f"✅ Average R² Score after Tuning: {np.mean(r2_scores):.4f}")

joblib.dump(best_model, 'xgb_reg_percentagediff_model.pkl')
joblib.dump(encoder, 'xgbencoder.pkl')
joblib.dump(scaler, 'xgbscaler.pkl')
