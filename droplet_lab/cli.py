import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .synthetic import Config,REGIMES,generate_pair
from .piv import estimate, physical_velocity
from .plots import pair_plot


def save_vectors(path, field, scale=1.):
    rows=np.column_stack([a.ravel() for a in (field.x,field.y,field.u*scale,field.v*scale,
                                              field.snr,field.eligible,field.valid)])
    np.savetxt(path,rows,delimiter=",",header="x_px,y_px,u,v,peak_ratio,eligible,valid",comments="")


def main():
    parser=argparse.ArgumentParser(description="DropletFlowLab synthetic PIV research workbench")
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo"); demo.add_argument("--regime",choices=REGIMES,default="mixed")
    demo.add_argument("--seed",type=int,default=42); demo.add_argument("--output",default="outputs/demo")
    bench=sub.add_parser("benchmark"); bench.add_argument("--runs",type=int,default=12)
    bench.add_argument("--frames",type=int,default=3); bench.add_argument("--seed",type=int,default=42)
    bench.add_argument("--output",default="outputs/benchmark")
    real=sub.add_parser("piv",help="Analyze two same-sized experimental grayscale images")
    real.add_argument("frame_a"); real.add_argument("frame_b"); real.add_argument("--mask")
    real.add_argument("--microns-per-pixel",type=float,required=True)
    real.add_argument("--dt",type=float,required=True)
    real.add_argument("--output",default="outputs/experimental")
    args=parser.parse_args()
    if args.command=="benchmark":
        from .experiment import benchmark
        print(json.dumps(benchmark(args.output,args.runs,args.frames,args.seed),indent=2)); return
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    if args.command=="demo":
        cfg=Config(regime=args.regime,seed=args.seed)
        a,b,u,v,mask=generate_pair(cfg); f=estimate(a,b,mask)
        fig=pair_plot(a,b,f,u,v); scale=1.
        np.savez_compressed(output/"truth.npz",u=u,v=v,mask=mask)
        metadata={"source":"synthetic kinematic surrogate","config":cfg.__dict__,"velocity_unit":"pixels/pair"}
    else:
        a,b=[np.asarray(Image.open(p).convert("F"),dtype=float) for p in (args.frame_a,args.frame_b)]
        mask=None if args.mask is None else np.asarray(Image.open(args.mask).convert("L"))>0
        f=estimate(a,b,mask); physical_velocity(f,args.microns_per_pixel,args.dt)
        scale=args.microns_per_pixel/args.dt
        fig=pair_plot(a/max(a.max(),1),b/max(b.max(),1),f)
        metadata={"source":"user images","velocity_unit":"microns/second",
                  "microns_per_pixel":args.microns_per_pixel,"dt_seconds":args.dt,
                  "note":"Plot arrows are pixels/pair. CSV is calibrated. No mechanism labels inferred."}
    fig.savefig(output/"piv.png",dpi=160)
    save_vectors(output/"vectors.csv",f,scale)
    if args.command=="demo":
        for name,im in [("frame_a.png",a),("frame_b.png",b)]: Image.fromarray((im*255).astype("uint8")).save(output/name)
    metadata.update({"valid_vectors":int(f.valid.sum()),"eligible_vectors":int(f.eligible.sum())})
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Saved results to {output.resolve()}")


if __name__=="__main__": main()
