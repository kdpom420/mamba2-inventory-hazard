# =====================================================
# 0. Libraries & Device
# =====================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mamba2 import Mamba2, Mamba2Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

def data_preprocessing(file_name):
    # -----------------------------
    # 1. 원본 데이터
    # -----------------------------
    df = pd.read_csv(file_name, sep="\t")
    
    # -----------------------------
    # 2. 날짜 타입 변환
    # -----------------------------
    df["DLNG_YMD"] = pd.to_datetime(df["DLNG_YMD"], format="%Y%m%d")
    
    # -----------------------------
    # 3. 날짜 인덱스로 설정
    # -----------------------------
    df = df.set_index("DLNG_YMD")
    
    # -----------------------------
    # 4. 전체 날짜 범위 생성
    # -----------------------------
    full_dates = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D"
    )
    
    # -----------------------------
    # 5. 날짜 채우기
    # -----------------------------
    df = df.reindex(full_dates)
    
    # -----------------------------
    # 6. USE_QTY = 0, 나머지는 이전 값 유지
    # -----------------------------
    df["USE_QTY"] = df["USE_QTY"].fillna(0)
    df["BF_IVNT_QTY"] = df["BF_IVNT_QTY"].fillna(0)
    df = df.ffill()
    
    # -----------------------------
    # 7. DLNG_YMD 다시 컬럼으로
    # -----------------------------
    df = df.reset_index().rename(columns={"index": "DLNG_YMD"})
    
    # 필요하면 다시 YYYYMMDD 정수로
    df["DLNG_YMD"] = df["DLNG_YMD"].dt.strftime("%Y%m%d").astype(int)
    return df
# =====================================================
# 1. Data Load & Preprocess
# =====================================================
#df = data_preprocessing("sparse_data_260108.txt")
df = data_preprocessing("dense_data_260108.txt")

df = df.sort_values(["DLNG_YMD"]).reset_index(drop=True)
cols = [
    "USE_QTY",
    "BF_IVNT_QTY",
    "NOW_IVNT_QTY",
    "SFEM_IVNT_QTY",
    "YMD_AVG_USE_QTY",
]
x_raw = df[cols].values.astype(np.float32)
mean = x_raw.mean(axis=0)
std = x_raw.std(axis=0) + 1e-6
x_norm = (x_raw - mean) / std
x = torch.tensor(x_norm).unsqueeze(0).to(device)  # (1, T, D)

# =====================================================
# 2. Model
# =====================================================
class InventoryMamba(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        config = Mamba2Config(
            d_model=64,
            n_layer=4,
            d_state=16,
            expand=2,
            chunk_size=1
        )
        self.mamba = Mamba2(config)
        self.output_proj = nn.Linear(64, input_dim)

    def forward(self, x):
        h = self.input_proj(x)
        y = self.mamba(h)
        if isinstance(y, tuple):
            y = y[0]
        y = self.output_proj(y)  # Reconstruction output
        return y, h  # y = recon, h = latent state

model = InventoryMamba(x.shape[-1]).to(device)

# Train the model (autoencoder-like reconstruction)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
model.train()
for epoch in range(100):
    y, _ = model(x)
    loss = criterion(y, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()

# =====================================================
# 3. Latent State → Energy, Delta, Cosine (State-Dependent DOC 적용)
# =====================================================
with torch.no_grad():
    _, latent = model(x)
latent = latent[0].cpu()  # (T, H)
state_mag = latent.norm(dim=1)  # E = latent magnitude (T,)

# =====================================================
# 4. Latent Invariant (Current Time t and Time Window t-k)
# =====================================================
k = 14 # Time window size (14 days in this case)
# Initialize the lists to hold the invariant metrics

tau_values = []
doc_dynamic = []
validity_values = []

for t in range(k, len(latent)):
    # Define the time window (h_{t-k:t})
    window = latent[t-k:t]
    # E: Energy (latent state magnitude)
    E = window.norm(dim=1)  # (k,)
    
    # Δh: Latent state 변화량 (t-k부터 t까지의 차이)
    delta_h = torch.norm(window[1:] - window[:-1], dim=1)  # (k-1,)
    
    # Cosine: 방향 변화 (cosine similarity)
    cosine_similarity = torch.nn.functional.cosine_similarity(window[1:], window[:-1], dim=1)  # (k-1,) - Adjusted to match delta_h length
    
    # Calculate validity: Using MSE-based losses
    delta_loss = torch.nn.functional.mse_loss(delta_h, torch.zeros_like(delta_h))  # Δh close to zero
    cosine_loss = torch.nn.functional.mse_loss(cosine_similarity, torch.ones_like(cosine_similarity))  # Cosine close to 1
    validity = delta_loss + cosine_loss  # Combined validity
    
    # tau calculation: Redesigned as validity / E[-1] (higher validity means higher uncertainty, normalize by energy)
    tau = validity / (E[-1] + 1e-6)
    tau_values.append(tau.item())
    
    # DOC 동적 계산
    current_stock = df["SFEM_IVNT_QTY"].iloc[t]  # 현재 재고
    avg_use = df["USE_QTY"].rolling(k, min_periods=1).mean().iloc[t]  # 기준 소비량 (NaN handling)
    doc = current_stock / (avg_use + 1e-6)  # 최소값 방지
    
    # tau에 비례한 DOC 변화 - Soft adjustment: doc * 1/(1+tau)
    doc_dynamic.append(doc * (1 / (1 + tau)))
    validity_values.append(validity.item())  # validity 추적

# Convert to numpy for further calculations
tau_values = np.array(tau_values)
doc_dynamic = np.array(doc_dynamic)

# =====================================================
# 5. Hazard Calibration (State-Dependent DOC 적용)
# =====================================================
BASE_HAZARD = avg_use /(df.tail(1)['SFEM_IVNT_QTY'].mean()+1e-6)  # 하루 기본 위험 (도메인 튜닝 대상)
#BETA = 0.3  # 상태 민감도

# For simplicity, compute hazard for last t (or average; here using last for demo)
t = len(latent) - 1  # Last time point
window = latent[t-k:t]
E = window.norm(dim=1)
delta_h = torch.norm(window[1:] - window[:-1], dim=1)
phi = delta_h.mean().item()  # 상태 변화량에 대한 평균 값 (float)
tau_now = tau_values[-1]  # 마지막 tau 값
# tau calibration 
scale = 2.0
tau_calibrated = 1/ (1 + np.exp(-scale*(tau_now-1.0)))

# hazard_now는 phi와 tau를 고려하여 동적으로 계산
#hazard_now = BASE_HAZARD * np.exp(BETA * state_mag[t].item()) * (1 + phi) * (1+tau_calibrated)
hazard_multiplier = 1 + tau_calibrated * 5
hazard_now = BASE_HAZARD * hazard_multiplier
hazard_now = np.clip(hazard_now, 1e-4, 0.3)  # 위험도를 0.3으로 제한

# =====================================================
# 6. Future Risk (Hazard → Survival)
# =====================================================
HORIZON = 60
hazard_future = np.full(HORIZON, hazard_now)
# 누적 위험
cum_hazard = np.cumsum(hazard_future)
survival = np.exp(-cum_hazard)
risk = 1 - survival

# =====================================================
# 7. Time Window Risk (CDF 기반 – 진짜 누적 확률)
# =====================================================
windows = {
    "0–7 days": 7,
    "7–14 days": 14,
    "14–30 days": 30,
    "30–60 days": 60,
}
window_interval_prob = {}  # 구간별 first-occurrence 확률
window_cdf = {}  # CDF 누적 확률
prev_surv = 1.0
for name, t_val in windows.items():
    t_val = min(t_val, len(survival))
    
    # CDF: t일까지 발생 확률
    cdf_t = 1 - survival[t_val - 1]
    
    # 구간 확률: 이전까지 생존 − 현재 생존
    interval_p = prev_surv - survival[t_val - 1]
    
    window_cdf[name] = cdf_t
    window_interval_prob[name] = interval_p
    
    prev_surv = survival[t_val - 1]

# =====================================================
# 8. Visualization
# =====================================================
plt.figure(figsize=(10, 4))
plt.plot(risk, label="Cumulative Risk", linewidth=2)
for name, t_val in windows.items():
    plt.axvline(t_val, linestyle="--", alpha=0.3)
    plt.text(t_val + 0.5, 0.05, name, rotation=90, fontsize=8)
plt.ylim(0, 1)
plt.xlabel("Days Ahead")
plt.ylabel("Probability")
plt.title("Stockout Risk (Hazard-based, Time Window with State-Dependent DOC)")
plt.grid(True)
plt.legend()
plt.show()

# =====================================================
# 9. Interpretable Output
# =====================================================
print("📌 Stockout Risk (CDF-based Interpretation)")
print("-" * 55)
for k in windows.keys():
    print(
        f"{k:>12} | "
        f"구간 발생 {window_interval_prob[k]*100:5.1f}% | "
        f"CDF 누적 {window_cdf[k]*100:5.1f}%"
    )
print("-" * 55)
print(f"60일 이후 미발생 확률 : {survival[windows['30–60 days']-1]*100:.1f}%")
print(f"\nCurrent hazard rate: {hazard_now:.4f} / day")