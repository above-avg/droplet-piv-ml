"""Launch with: streamlit run app.py"""
import io
import streamlit as st
from droplet_lab.synthetic import Config,REGIMES,generate_pair
from droplet_lab.piv import estimate
from droplet_lab.plots import pair_plot
from droplet_lab.features import extract,FEATURE_NAMES
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="DropletFlowLab",page_icon="💧",layout="wide")
st.title("💧 DropletFlowLab")
st.caption("Particle image velocimetry • flow features • synthetic validation")
st.info("Kinematic surrogates, not a validated evaporation simulation. Planar circulation is not proof of Marangoni flow.")
with st.sidebar:
    st.header("Virtual experiment")
    regime=st.selectbox("Synthetic pattern",REGIMES,index=2)
    age=st.slider("Normalized age (amplitude schedule only)",0.,1.,.4)
    amplitude=st.slider("Motion amplitude (px/pair)",.5,4.,2.,.1)
    noise=st.slider("Image noise σ",0.,.2,.025,.005)
    particles=st.slider("Target particle count",400,2400,1500,100)
    window=st.select_slider("PIV window (pixels)",options=[16,24,32,40,48],value=32)
    snr=st.slider("Minimum peak ratio",1.,2.,1.15,.05)
    seed=int(st.number_input("Seed",0,1000000,42))
cfg=Config(regime=regime,age=age,amplitude=amplitude,noise=noise,particles=particles,seed=seed)
a,b,u,v,mask=generate_pair(cfg)
f=estimate(a,b,mask,window=window,step=window//2,max_shift=min(7,window//2-1),min_snr=snr)
from scipy.ndimage import map_coordinates
e=np.hypot(f.u-map_coordinates(u,[f.y,f.x],order=1),f.v-map_coordinates(v,[f.y,f.x],order=1))
cols=st.columns(3)
cols[0].metric("Accepted / eligible vectors",f"{f.valid.sum()} / {f.eligible.sum()}")
cols[1].metric("Mean endpoint error",f"{np.nanmean(e):.3f} px" if f.valid.any() else "No valid vectors")
cols[2].metric("Motion scale",f"{amplitude:.1f} px/pair")
fig=pair_plot(a,b,f,u,v); st.pyplot(fig); plt.close(fig)
with st.expander("Measured flow features"):
    if f.valid.sum()>=5: st.dataframe(dict(zip(FEATURE_NAMES,extract(f,((cfg.size-1)/2,)*2))),use_container_width=True)
    else: st.warning("Not enough accepted vectors for classification features.")
st.subheader("Measurement diagnostics")
st.markdown("Endpoint error compares accepted vectors with center-sampled ground truth. Coverage reports the fraction of eligible windows with accepted estimates. Interrogation-window size controls the tradeoff between spatial resolution and correlation reliability.")
st.markdown("**Batch evaluation:** `droplet-lab benchmark` reports classification metrics on held-out droplet runs, a majority-class baseline, and an independent noise-shift evaluation.")
buffer=io.StringIO()
np.savetxt(buffer,np.column_stack([x.ravel() for x in [f.x,f.y,f.u,f.v,f.snr,f.valid]]),delimiter=",",header="x_px,y_px,u_px_per_pair,v_px_per_pair,peak_ratio,valid",comments="")
st.download_button("Download measured vectors",buffer.getvalue(),"vectors.csv","text/csv")
