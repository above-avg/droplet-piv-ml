"""Seeded benchmark with independent run groups, baseline, and OOD stress test."""

import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.ndimage import map_coordinates
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from .synthetic import Config, REGIMES, generate_pair
from .piv import estimate
from .features import extract, FEATURE_NAMES


def sample(cfg):
    a,b,u,v,mask = generate_pair(cfg)
    f = estimate(a,b,mask)
    truth_u = map_coordinates(u, [f.y, f.x], order=1)
    truth_v = map_coordinates(v, [f.y, f.x], order=1)
    epe = np.hypot(f.u-truth_u, f.v-truth_v)
    return extract(f, ((cfg.size-1)/2, (cfg.size-1)/2)), float(np.nanmean(epe)), f


def dataset(runs_per_class=12, frames=3, seed=42, noise=None):
    if runs_per_class < 4 or frames < 1:
        raise ValueError("need >=4 runs per class and >=1 pair per run")
    rng = np.random.default_rng(seed)
    rows, labels, groups, errors, coverage = [], [], [], [], []
    for label, regime in enumerate(REGIMES):
        for run in range(runs_per_class):
            cfg = Config(regime=regime, amplitude=rng.uniform(1.5,2.7),
                         particles=int(rng.integers(1100,2000)), particle_sigma=rng.uniform(.75,1.15),
                         noise=rng.uniform(.01,.045) if noise is None else noise,
                         direction=int(rng.choice([-1,1])))
            for age in np.linspace(.15,.85,frames):
                feat, err, f = sample(replace(cfg, age=float(age), seed=int(rng.integers(0,2**31))))
                rows.append(feat); labels.append(label); groups.append(label*runs_per_class+run)
                errors.append(err); coverage.append(f.valid.sum()/max(f.eligible.sum(),1))
    return np.array(rows), np.array(labels), np.array(groups), np.array(errors), np.array(coverage)


def split_groups(y, groups, seed=42):
    rng = np.random.default_rng(seed)
    test_groups = []
    for label in np.unique(y):
        ids = np.unique(groups[y==label])
        test_groups.extend(rng.permutation(ids)[:max(1,int(np.ceil(.25*len(ids))))])
    test = np.isin(groups,test_groups)
    return np.flatnonzero(~test), np.flatnonzero(test)


def scores(y, prediction):
    return {"accuracy": float(accuracy_score(y,prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y,prediction)),
            "macro_f1": float(f1_score(y,prediction,average="macro")),
            "confusion_matrix": confusion_matrix(y,prediction,labels=[0,1,2]).tolist()}


def bootstrap_accuracy(y, pred, groups, seed):
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    per_run = np.array([np.mean(y[groups==g] == pred[groups==g]) for g in ids])
    draws = rng.choice(per_run,(1000,len(ids)),replace=True).mean(axis=1)
    return np.quantile(draws,[.025,.975]).tolist()


def benchmark(output="outputs/benchmark", runs=12, frames=3, seed=42):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    X,y,g,e,c = dataset(runs,frames,seed)
    train,test = split_groups(y,g,seed)
    model = RandomForestClassifier(n_estimators=200,max_depth=7,min_samples_leaf=2,
                                   class_weight="balanced",random_state=seed,n_jobs=1)
    model.fit(X[train],y[train])
    pred = model.predict(X[test])
    dummy = DummyClassifier(strategy="most_frequent").fit(X[train],y[train])
    # A new seed creates genuinely new run realizations under a sensor-noise shift.
    Xo,yo,go,eo,co = dataset(max(4,runs//3),frames,seed+10000,noise=.10)
    report = {"seed":seed, "regimes":list(REGIMES), "pairs":len(y),
              "train_runs":len(np.unique(g[train])), "test_runs":len(np.unique(g[test])),
              "group_overlap":len(set(g[train]) & set(g[test])),
              "random_forest":scores(y[test],pred), "dummy":scores(y[test],dummy.predict(X[test])),
              "accuracy_95pct_run_bootstrap":bootstrap_accuracy(y[test],pred,g[test],seed),
              "noise_shift":scores(yo,model.predict(Xo)),
              "piv_mean_endpoint_error_px":float(e.mean()),
              "piv_test_endpoint_error_px":float(e[test].mean()),
              "piv_valid_fraction":float(c.mean()),
              "noise_shift_piv_endpoint_error_px":float(eo.mean()),
              "noise_shift_valid_fraction":float(co.mean()),
              "feature_importance_impurity":dict(zip(FEATURE_NAMES,model.feature_importances_.tolist())),
              "limitations":["Synthetic kinematic classes; not measured evaporation mechanisms.",
                             "Few held-out runs; interval excludes simulator/model uncertainty.",
                             "PIV errors reported only on accepted vectors; inspect coverage.",
                             "Window-averaged PIV compared to velocity at window center."]}
    (output/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez_compressed(output/"dataset.npz",X=X,y=y,groups=g,train=train,test=test,
                        endpoint_error=e,valid_fraction=c)
    from .plots import benchmark_plot
    benchmark_plot(report,output/"benchmark.png")
    return report
