# Statistical AI for Continuous Physiological Data

Interactive methodological demonstration (Streamlit) for digital-health research:

- Functional / dimension reduction (PCA as practical FPCA proxy)
- High-dimensional variable selection (Lasso)
- Uncertainty quantification (Gaussian Process)
- Synthetic CGM-like trajectories only

## Run locally

```bash
pip install -r requirements.txt
streamlit run app_streamlit_demo.py
```

## Deploy

Use [Streamlit Community Cloud](https://share.streamlit.io) with main file `app_streamlit_demo.py`.

**Not for clinical use.** Synthetic data only.
