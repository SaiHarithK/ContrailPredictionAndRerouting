   

import numpy as np
import pandas as pd


                                                              
                    
                                                              

def _sigmoid(x: np.ndarray, center: float, scale: float) -> np.ndarray:
           
    return 1.0 / (1.0 + np.exp(-(x - center) / scale))


def _linear_clip(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
           
    score = (x - x_min) / (x_max - x_min)
    return np.clip(score, 0.0, 1.0)


                                                              
                                          
                                                              

def formation_score(delta_t: np.ndarray,
                    center: float = 0.0,
                    scale: float = 1.5) -> np.ndarray:
           
    return _sigmoid(-delta_t, center=center, scale=scale)


def persistence_score(rhi: np.ndarray,
                      center: float = 100.0,
                      scale: float = 10.0) -> np.ndarray:
           
    return _sigmoid(rhi, center=center, scale=scale)


def wind_score(wind_speed: np.ndarray,
               wind_shear: np.ndarray,
               speed_optimal: float = 10.0,
               speed_limit: float = 40.0,
               shear_limit: float = 0.005) -> np.ndarray:
           
    speed_component = 1.0 - _linear_clip(wind_speed, 0.0, speed_limit)
    shear_component = 1.0 - _linear_clip(np.abs(wind_shear), 0.0, shear_limit)
    return 0.5 * speed_component + 0.5 * shear_component


def stability_score(static_stability: np.ndarray,
                    low_stab: float = 1e-4,
                    high_stab: float = 5e-4) -> np.ndarray:
           
    return _linear_clip(static_stability, low_stab, high_stab)


def issr_score(issr_depth: np.ndarray,
               min_depth: float = 0.0,
               max_depth: float = 4.0) -> np.ndarray:
           
    return _linear_clip(issr_depth, min_depth, max_depth)


                                                              
                                         
                                                              

def compute_p_contrail(
        formation:   np.ndarray,
        persistence: np.ndarray,
        wind:        np.ndarray,
        stability:   np.ndarray,
        issr:        np.ndarray,
        weights:     dict = None) -> np.ndarray:
           
    if weights is None:
        weights = {
            'formation':   0.30,                                
            'persistence': 0.30,                                            
            'wind':        0.15,                               
            'stability':   0.15,                                
            'issr':        0.10,                         
        }

    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, (
        f"Weights must sum to 1.0, got {total:.4f}")

                                                              
    eps = 1e-6
    f = np.clip(formation,   eps, 1 - eps)
    p = np.clip(persistence, eps, 1 - eps)
    w = np.clip(wind,        eps, 1 - eps)
    s = np.clip(stability,   eps, 1 - eps)
    i = np.clip(issr,        eps, 1 - eps)

    p_contrail = (
        f ** weights['formation']  *
        p ** weights['persistence'] *
        w ** weights['wind']       *
        s ** weights['stability']  *
        i ** weights['issr']
    )
    return p_contrail


def generate_labels(
        df: pd.DataFrame,
        threshold: float = 0.5,
        formation_center: float = 0.0,
        formation_scale: float = 1.5,
        persistence_center: float = 100.0,
        persistence_scale: float = 10.0,
        stochastic_labels: bool = False
) -> pd.DataFrame:

    required = ['delta_t', 'rhi']

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    # SAC formation probability
    df['formation_score'] = formation_score(
        df['delta_t'].values,
        center=formation_center,
        scale=formation_scale
    )

    # Persistence probability
    df['persistence_score'] = persistence_score(
        df['rhi'].values,
        center=persistence_center,
        scale=persistence_scale
    )

    # Final contrail probability
    df['p_contrail'] = (
        df['formation_score']
        *
        df['persistence_score']
    )

    # Binary labels
    if stochastic_labels:

        rng = np.random.default_rng(42)

        df['label'] = rng.binomial(
            n=1,
            p=df['p_contrail'].values
        )

    else:

        df['label'] = (
            df['p_contrail'] > threshold
        ).astype(int)

    return df

                                                              
                         
                                                              

def label_diagnostics(df: pd.DataFrame) -> None:
           
    print("=" * 55)
    print("LABEL GENERATION DIAGNOSTICS")
    print("=" * 55)

    total = len(df)
    n_pos = df['label'].sum()
    n_neg = total - n_pos
    print(f"Total rows      : {total:,}")
    print(f"Label = 1 (contrail)     : {n_pos:,}  ({100*n_pos/total:.1f}%)")
    print(f"Label = 0 (no contrail)  : {n_neg:,}  ({100*n_neg/total:.1f}%)")

    print("\nScore statistics (mean ± std):")
    score_cols = [
        'formation_score',
        'persistence_score',
        'p_contrail'
    ]
    for col in score_cols:
        if col in df.columns:
            print(f"  {col:<22}: {df[col].mean():.3f} ± {df[col].std():.3f}"
                  f"  [min={df[col].min():.3f}, max={df[col].max():.3f}]")

    print("\np_contrail distribution (deciles):")
    deciles = df['p_contrail'].quantile(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    for q, v in deciles.items():
        print(f"  {int(q*100):3d}th percentile : {v:.4f}")
    print("=" * 55)


                                                              
                                      
                                                              

if __name__ == "__main__":
           
    import os

    FEATURE_FILE = "era5_features.parquet"                                   

    if not os.path.exists(FEATURE_FILE):
        print(f"Feature file '{FEATURE_FILE}' not found.")
        print("Run feature_engineering.py first to generate this file.")
        raise SystemExit(1)

    print(f"Loading features from {FEATURE_FILE} ...")
    df_features = pd.read_parquet(FEATURE_FILE)
    print(f"Loaded {len(df_features):,} rows.")

    print("\nGenerating labels ...")
    df_labeled = generate_labels(
        df_features,
        threshold=0.5,
        weights=None,                                              
    )

    label_diagnostics(df_labeled)

    out_file = "era5_labeled.parquet"
    df_labeled.to_parquet(out_file, index=False)
    print(f"\nSaved labeled dataset to: {out_file}")
    print("Columns available for ML training:")
    print([c for c in df_labeled.columns])