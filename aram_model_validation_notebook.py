# ARAM Model Validation Notebook (Oct 22 Pre-Data Only)

import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =====================
# 1. Load Data
# =====================

file_path = "match_stats_20260329_1826.csv"
df = pd.read_csv(file_path)

# =====================
# 2. Basic Preprocess
# =====================

# convert time
if 'gameCreation' in df.columns:
    df['gameCreation'] = pd.to_datetime(df['gameCreation'], unit='ms', errors='coerce')

# filter pre Oct 22
cutoff_date = pd.to_datetime("2022-10-22")
df = df[df['gameCreation'] < cutoff_date]

# sort
df = df.sort_values(['puuid', 'gameCreation'])

# =====================
# 3. Feature Engineering
# =====================

# KDA
if set(['kills','assists','deaths']).issubset(df.columns):
    df['kda'] = (df['kills'] + df['assists']) / (df['deaths'] + 1)

# Damage per min
if set(['totalDamageDealt','gameDuration']).issubset(df.columns):
    df['dpm'] = df['totalDamageDealt'] / (df['gameDuration'] / 60 + 1e-6)

# Avg performance (rolling)
df['avg_kda'] = df.groupby('puuid')['kda'].transform(lambda x: x.rolling(5, min_periods=1).mean())

# Volatility (rolling std)
df['volatility'] = df.groupby('puuid')['kda'].transform(lambda x: x.rolling(5, min_periods=2).std())

# Loss indicator
if 'win' in df.columns:
    df['loss'] = 1 - df['win']

# Lag variables
df['loss_lag'] = df.groupby('puuid')['loss'].shift(1)
df['volatility_lag'] = df.groupby('puuid')['volatility'].shift(1)
df['avg_kda_lag'] = df.groupby('puuid')['avg_kda'].shift(1)

# Streak loss

def compute_loss_streak(x):
    streak = []
    count = 0
    for val in x:
        if val == 1:
            count += 1
        else:
            count = 0
        streak.append(count)
    return pd.Series(streak, index=x.index)

if 'loss' in df.columns:
    df['loss_streak'] = df.groupby('puuid')['loss'].apply(compute_loss_streak).reset_index(level=0, drop=True)
    df['loss_streak_lag'] = df.groupby('puuid')['loss_streak'].shift(1)

# Repeat play (t+1)
df['repeat'] = df.groupby('puuid')['matchId'].shift(-1).notnull().astype(int)

# =====================
# 4. Drop NA
# =====================

model_df = df.dropna(subset=['loss_lag','volatility_lag','avg_kda_lag'])

# =====================
# 5. Model 1: Switching Behavior
# =====================

model1 = smf.logit(
    formula="repeat ~ loss_lag + volatility_lag + avg_kda_lag + loss_streak_lag",
    data=model_df
).fit()

print("\n=== Model 1: Switching Behavior ===")
print(model1.summary())

# =====================
# 6. Model 2: Volatility Effect
# =====================

model2 = smf.logit(
    formula="repeat ~ volatility_lag + avg_kda_lag",
    data=model_df
).fit()

print("\n=== Model 2: Volatility Effect ===")
print(model2.summary())

# =====================
# 7. Model 3: Outcome vs Process
# =====================

if 'dpm' in model_df.columns:
    model3 = smf.logit(
        formula="repeat ~ loss_lag + kda + dpm",
        data=model_df
    ).fit()

    print("\n=== Model 3: Outcome vs Process ===")
    print(model3.summary())

# =====================
# 8. Quick Interpretation Helper
# =====================

print("\n\nKey Interpretation Guide:")
print("- loss_lag: if negative → losing reduces engagement")
print("- volatility_lag: if positive → randomness increases engagement")
print("- avg_kda_lag: performance effect")
print("- loss_streak_lag: behavioral fatigue / tilt")

print("\nDONE ✅")