import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pair_plot(a, b, f, truth_u=None, truth_v=None):
    fig, ax = plt.subplots(1,3,figsize=(13,4),layout="constrained")
    ax[0].imshow(a,cmap="gray",vmin=0,vmax=1); ax[0].set_title("Particle image A")
    ax[1].imshow(b,cmap="gray",vmin=0,vmax=1); ax[1].set_title("Image B + accepted PIV")
    ax[1].quiver(f.x,f.y,f.u,f.v,color="#23e8cb",angles="xy",scale_units="xy",scale=.25)
    if truth_u is not None:
        from scipy.ndimage import map_coordinates
        error = np.hypot(f.u-map_coordinates(truth_u,[f.y,f.x],order=1),
                         f.v-map_coordinates(truth_v,[f.y,f.x],order=1))
        dots=ax[2].scatter(f.x[f.valid],f.y[f.valid],c=error[f.valid],cmap="magma",s=75,vmin=0)
        fig.colorbar(dots,ax=ax[2],label="Endpoint error (px/pair)")
        ax[2].set_title("Recovery error vs known motion")
    else:
        dots=ax[2].scatter(f.x[f.valid],f.y[f.valid],c=f.snr[f.valid],cmap="viridis",s=75)
        fig.colorbar(dots,ax=ax[2],label="Peak ratio (heuristic)")
        ax[2].set_title("Correlation confidence")
    ax[2].set_xlim(0,a.shape[1]); ax[2].set_ylim(a.shape[0],0)
    for axis in ax:
        axis.set_aspect("equal"); axis.set_xlabel("x (pixels)"); axis.set_ylabel("y down (pixels)")
    return fig


def benchmark_plot(report,path):
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    cm=np.array(report["random_forest"]["confusion_matrix"])
    ax[0].imshow(cm,cmap="Blues")
    for j in range(3):
        for i in range(3): ax[0].text(i,j,str(cm[j,i]),ha="center",va="center")
    ax[0].set_xticks(range(3),report["regimes"],rotation=20)
    ax[0].set_yticks(range(3),report["regimes"])
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Synthetic label")
    ax[0].set_title("Unseen-run confusion matrix")
    vals=[report[k]["balanced_accuracy"] for k in ["dummy","random_forest","noise_shift"]]
    ax[1].bar(["Majority baseline","Random forest","Noise shift"],vals,color=["#94a3b8","#0891b2","#f59e0b"])
    ax[1].set_ylim(0,1.08); ax[1].set_ylabel("Balanced accuracy")
    ax[1].set_title("Synthetic benchmark only")
    fig.savefig(path,dpi=160); plt.close(fig)
