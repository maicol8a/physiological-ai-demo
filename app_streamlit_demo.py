"""
Statistical AI for Continuous Physiological Data
Interactive demonstration aligned with digital-health methodological research.

Run:
    python -m streamlit run app_streamlit_demo.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Statistical AI · Continuous Physiological Data",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Synthetic data
# -----------------------------
@st.cache_data
def generate_data(n_subjects=120, n_points=288, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, n_points)
    curves = np.zeros((n_subjects, n_points))
    risk = np.zeros(n_subjects)

    for i in range(n_subjects):
        baseline = rng.normal(110, 15)
        circadian = 8 * np.sin(2 * np.pi * (t - 6) / 24)
        meals = (
            25 * np.exp(-0.5 * ((t - 8) / 1.2) ** 2)
            + 30 * np.exp(-0.5 * ((t - 13) / 1.5) ** 2)
            + 28 * np.exp(-0.5 * ((t - 19) / 1.4) ** 2)
        )
        meal_scale = rng.uniform(0.7, 1.5)
        noise = rng.normal(0, rng.uniform(4, 14), n_points)
        glucose = np.clip(baseline + circadian + meal_scale * meals + noise, 55, 300)
        curves[i] = glucose

        time_hyper = np.mean(glucose > 180) * 100
        cv = (np.std(glucose) / np.mean(glucose)) * 100
        mean_gluc = np.mean(glucose)
        area = np.trapezoid(np.maximum(glucose - 140, 0), t)
        risk[i] = (
            0.45 * time_hyper
            + 0.35 * cv
            + 0.25 * (mean_gluc - 100)
            + 0.002 * area
            + rng.normal(0, 5)
        )

    risk = 45 + 16 * (risk - risk.mean()) / risk.std()
    risk = np.clip(risk, 10, 88)
    return t, curves, risk


t, curves, y = generate_data()

# Shared model objects (computed once for the page)
pca = PCA(n_components=6)
scores = pca.fit_transform(curves)

X_class = np.column_stack(
    [
        curves.mean(1),
        curves.std(1),
        curves.std(1) / curves.mean(1),
        curves.max(1),
        curves.min(1),
        (curves > 180).mean(1),
        (curves < 70).mean(1),
        np.percentile(curves, 75, 1) - np.percentile(curves, 25, 1),
    ]
)
X = np.hstack([scores, X_class])
feature_names = [f"PC{i+1}" for i in range(6)] + [
    "mean",
    "std",
    "cv",
    "tmax",
    "tmin",
    "hyper",
    "hypo",
    "iqr",
]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_tr_s, y_tr)
coef = pd.Series(lasso.coef_, index=feature_names)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
y_rf = rf.predict(X_te_s)

gp = GaussianProcessRegressor(
    kernel=C(1.0) * RBF(1.0) + WhiteKernel(1.0),
    n_restarts_optimizer=1,
    random_state=42,
).fit(X_tr_s[:, :4], y_tr)
y_gp, y_std = gp.predict(X_te_s[:, :4], return_std=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Overview & motivation",
        "1. Continuous physiological data",
        "2. Functional / dimension reduction",
        "3. Variable selection",
        "4. Uncertainty quantification",
        "5. Mapping to the position",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Companion materials**
- Full Jupyter notebook with FDA, autoencoders, Horseshoe prior, conformal prediction (when available), and complete pipeline.
"""
)
st.sidebar.caption("Synthetic data · Methodological illustration · Not for clinical use")

# -----------------------------
# Helper plot style
# -----------------------------
def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


# =========================================================
# OVERVIEW
# =========================================================
if section == "Overview & motivation":
    st.title("Statistical AI for Continuous Physiological Data")
    st.subheader("A compact methodological demonstration for digital health")

    st.markdown(
        """
This interactive demo illustrates how **methodological statistical AI** can be applied to
continuous physiological signals collected from wearables and smartphones — precisely the
setting described in the research position.

It is a **prototype**, not a clinical tool: the data are synthetic, the models are deliberately
compact, and the aim is to show the *scope of the methodological toolkit* (functional
representation, latent structure, sparse selection, and uncertainty-aware prediction) in a
form that can be inspected in minutes.
"""
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Synthetic subjects", f"{len(y)}")
    c2.metric("Sampling", "5 min · 24 h")
    c3.metric("Outcome", "Glycemic Risk Score")

    st.markdown("### Research framing")
    st.markdown(
        """
**Scientific problem.** Continuous glucose monitoring (CGM) and related wearable streams
produce dense functional trajectories. The challenge is not only prediction, but
**representation, selection, and trustworthy uncertainty** for downstream digital-health
decisions (e.g. risk stratification, personalised recommendations).

**Methodological intersection targeted by the position:**
1. **Functional data analysis** – treat each trajectory as a function  
2. **Representation learning** – autoencoders / neural latent structure  
3. **Uncertainty quantification** – conformal methods & Bayesian predictive models  
4. **High-dimensional variable selection** – sparse, interpretable digital biomarkers  
5. **Statistical learning for continuous physiological data** – end-to-end pipeline  

This dashboard shows a **lightweight end-to-end sketch** of that intersection on synthetic
CGM-like data. The companion notebook contains the fuller code (including Bayesian Horseshoe
selection and conformal prediction when libraries are available).
"""
    )

    st.info(
        "Use the sidebar to walk through each methodological block. "
        "Each section ends with a short note on relevance to digital-health research."
    )

# =========================================================
# 1. DATA
# =========================================================
elif section == "1. Continuous physiological data":
    st.header("1. Continuous physiological data (CGM-like)")
    st.markdown(
        """
We generate realistic 24-hour glucose trajectories with circadian structure, meal peaks,
subject-specific amplitude and noise. From each curve we derive a continuous
**Glycemic Risk Score** (proxy for longer-term metabolic risk).
"""
    )

    n_show = st.slider("Curves to display", 5, 30, 12)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(9, 4))
        for i in range(n_show):
            ax.plot(t, curves[i], alpha=0.65, lw=1)
        ax.axhline(180, color="crimson", ls="--", alpha=0.7, label="Hyperglycemia threshold")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Glucose (mg/dL)")
        ax.set_title("Synthetic continuous glucose trajectories")
        ax.legend()
        style_ax(ax)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(y, bins=16, color="steelblue", edgecolor="white")
        ax.set_xlabel("Glycemic Risk Score")
        ax.set_title("Outcome distribution")
        style_ax(ax)
        st.pyplot(fig)
        plt.close()

    st.markdown(
        f"**Summary:** {len(y)} subjects · mean risk = **{y.mean():.1f}** · "
        f"sd = **{y.std():.1f}** · range [{y.min():.0f}, {y.max():.0f}]"
    )

    st.success(
        "**Relevance to the position:** Continuous wearable/smartphone streams are the raw "
        "material for digital-health algorithms. Modelling them as functions (not just tables) "
        "is the starting point of the methodological agenda."
    )

# =========================================================
# 2. FDA / PCA
# =========================================================
elif section == "2. Functional / dimension reduction":
    st.header("2. Functional representation & dimension reduction")
    st.markdown(
        """
Each trajectory is treated as a functional observation. Here we use PCA as a practical,
robust proxy for Functional PCA (FPCA). The companion notebook attempts full FDA tooling
and falls back cleanly when needed.
"""
    )

    st.write("Variance explained by the first 6 components:")
    st.write(np.round(pca.explained_variance_ratio_, 3).tolist())
    st.write("Cumulative:", np.round(np.cumsum(pca.explained_variance_ratio_), 3).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, curves.mean(0), color="black", lw=2)
    axes[0].set_title("Mean trajectory")
    axes[0].set_xlabel("Time (h)")
    axes[0].set_ylabel("Glucose (mg/dL)")
    style_ax(axes[0])

    for i in range(6):
        axes[1].plot(t, curves[i], color="gray", alpha=0.35)
        recon = pca.mean_ + scores[i] @ pca.components_
        axes[1].plot(t, recon, lw=1.4)
    axes[1].set_title("Original (gray) vs low-rank reconstruction")
    axes[1].set_xlabel("Time (h)")
    style_ax(axes[1])
    st.pyplot(fig)
    plt.close()

    st.markdown(
        """
**Interpretation.** A small number of modes captures most between-subject variation
(often dominated by overall level and postprandial shape). These scores become inputs for
downstream selection and prediction.
"""
    )

    st.success(
        "**Relevance to the position:** Functional data analysis is listed explicitly in the "
        "call. Representing continuous physiological curves in a low-dimensional, interpretable "
        "basis is foundational for scalable statistical AI in digital health."
    )

# =========================================================
# 3. VARIABLE SELECTION
# =========================================================
elif section == "3. Variable selection":
    st.header("3. High-dimensional features & variable selection")
    st.markdown(
        """
We concatenate functional scores with classical glycemic summaries (mean, CV, time-in-range
proxies, extrema, IQR). This creates a moderately high-dimensional feature space on which
we apply **Lasso** for sparse selection of candidate digital biomarkers.
"""
    )

    n_sel = int(np.sum(lasso.coef_ != 0))
    st.metric("Lasso-selected features", f"{n_sel} / {X.shape[1]}")

    top = coef.reindex(coef.abs().nlargest(10).index)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    top.sort_values().plot(kind="barh", ax=ax, color="teal")
    ax.set_title("Largest absolute Lasso coefficients")
    ax.set_xlabel("Coefficient")
    style_ax(ax)
    st.pyplot(fig)
    plt.close()

    st.markdown(
        """
**Why this matters.** Wearable pipelines easily produce dozens or hundreds of derived
features. Without principled selection, models become opaque and fragile. Sparse methods
(and Bayesian shrinkage such as the Horseshoe prior, shown in the notebook) help isolate
stable, scientifically meaningful predictors.
"""
    )

    st.success(
        "**Relevance to the position:** Variable selection and high-dimensional statistics "
        "are core to the advertised research programme — especially when multimodal "
        "physiological streams are combined for risk prediction or personalised recommendations."
    )

# =========================================================
# 4. UQ
# =========================================================
elif section == "4. Uncertainty quantification":
    st.header("4. Uncertainty quantification for trustworthy prediction")
    st.markdown(
        """
Digital-health decisions require more than a point forecast. Here we compare:
- a **Random Forest** point predictor, and
- a **Gaussian Process** with approximate 90% predictive intervals.

The companion notebook also implements **conformal prediction** (MAPIE) when the library
is available, providing finite-sample coverage guarantees under exchangeability.
"""
    )

    r2 = r2_score(y_te, y_rf)
    cov = np.mean((y_te >= y_gp - 1.645 * y_std) & (y_te <= y_gp + 1.645 * y_std))
    width = np.mean(2 * 1.645 * y_std)

    m1, m2, m3 = st.columns(3)
    m1.metric("RF R² (test)", f"{r2:.2f}")
    m2.metric("GP empirical coverage", f"{cov:.2f}")
    m3.metric("Avg. GP interval width", f"{width:.1f}")

    lims = [
        min(y_te.min(), y_rf.min(), y_gp.min()) - 5,
        max(y_te.max(), y_rf.max(), y_gp.max()) + 5,
    ]

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.scatter(y_te, y_rf, alpha=0.65, s=28)
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("True risk score")
        ax.set_ylabel("Predicted")
        ax.set_title("Random Forest – point predictions")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        style_ax(ax)
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.errorbar(y_te, y_gp, yerr=1.645 * y_std, fmt="o", alpha=0.55, markersize=4)
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("True risk score")
        ax.set_ylabel("Predicted ± approx. 90% interval")
        ax.set_title("Gaussian Process – Bayesian uncertainty")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        style_ax(ax)
        st.pyplot(fig)
        plt.close()

    st.markdown(
        """
**Take-home.** Uncertainty is not optional in clinical-adjacent applications. Combining
flexible predictors with calibrated intervals (Bayesian and/or conformal) is a central
theme of modern statistical AI for digital health.
"""
    )

    st.success(
        "**Relevance to the position:** Uncertainty quantification and conformal prediction "
        "are explicitly listed. Showing both Bayesian predictive uncertainty and the path to "
        "distribution-free conformal guarantees speaks directly to the methodological agenda."
    )

# =========================================================
# 5. MAPPING
# =========================================================
elif section == "5. Mapping to the position":
    st.header("5. Mapping to the research agenda")

    st.markdown(
        """
This prototype was designed with a single purpose: to **illustrate the methodological
scope** listed in the position — functional representations, latent models, high-dimensional
selection, and uncertainty-aware statistical learning for continuous physiological data.

All analyses use **synthetic CGM-like trajectories**. No real patient data are involved.
The goal is transparency and reproducibility, not clinical claims.
"""
    )

    st.markdown(
        """
| Theme in the call | What this demonstration shows |
|---|---|
| **Functional data analysis** | Continuous CGM-like trajectories treated as functions; dimension reduction via PCA/FPCA-style scores |
| **Autoencoders & neural models** | Latent representations in the companion notebook (reconstruction + embeddings) |
| **Uncertainty quantification & conformal prediction** | Gaussian Process predictive intervals here; conformal methods in the notebook when available |
| **Variable selection & high-dimensional statistics** | Sparse Lasso selection of digital biomarkers; Bayesian Horseshoe selection in the notebook |
| **Statistical learning for continuous physiological data** | End-to-end sketch: raw curves → features → sparse model → uncertain prediction |
| **Digital-health applications** | Risk-score proxy motivated by metabolic risk monitoring and personalised follow-up |
"""
    )

    st.markdown("### From prototype to a research pathway")
    st.markdown(
        """
With a pipeline of this kind in place, the natural next steps on **real** digital-health
data would follow a structured research workflow:
"""
    )

    st.markdown(
        """
1. **Research question & hypotheses**  
   Formalise predictive and explanatory questions (e.g. risk stratification from continuous
   glucose dynamics; personalised response patterns relevant to nutrition or complication risk).

2. **Objectives**  
   Separate methodological aims (new representation / selection / UQ procedures) from
   applied aims (performance, calibration, and clinical utility on target endpoints).

3. **Focused literature review**  
   Position the work against existing FDA, deep latent, conformal and high-dimensional
   approaches for wearable streams, identifying gaps the methods are meant to close.

4. **Study design & methodology**  
   Pre-specify sampling, pre-processing, functional and latent representations, selection
   rules, uncertainty targets (Bayesian and/or conformal), and validation scheme
   (including sensitivity analyses).

5. **Target venues**  
   Identify methodological and applied outlets appropriate for statistical AI in digital
   health (statistics / ML journals and biomedical informatics or diabetes-technology venues).

6. **Draft for human review**  
   Produce a structured preliminary manuscript skeleton (background, methods, results
   template, limitations, reproducible code appendix) intended for iterative refinement by
   domain experts and statisticians — never as an unsupervised final product.
"""
    )

    st.markdown(
        """
This workflow reflects how methodological prototypes are turned into credible research:
**transparent design, auditable code, and human oversight at every scientific decision**.
"""
    )

    st.markdown("---")
    st.markdown(
        """
**About these materials**
- Dashboard: high-level story and interactive figures.  
- Companion Jupyter notebook: full commented pipeline (FDA/PCA, autoencoders, selection,
  Bayesian shrinkage, Gaussian processes, and conformal prediction when available).  
- Data: **synthetic only**, constructed to stress-test the toolkit advertised in the call.
"""
    )

    st.success(
        "The prototype is intentionally modest in scope and honest about its limits. "
        "Its value is to show readiness to develop the exact combination of methods "
        "required for the next generation of digital-health statistical AI."
    )
