import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift
from droplet_lab.synthetic import Config, generate_pair
from droplet_lab.piv import estimate, physical_velocity
from droplet_lab.experiment import split_groups, sample


@pytest.mark.parametrize("dx,dy",[(2.,-1.),(-2.3,1.4),(0.,0.),(.6,-.7)])
def test_known_translation(dx,dy):
    rng=np.random.default_rng(2)
    a=gaussian_filter(rng.random((128,128)),.8)
    b=shift(a,(dy,dx),order=3,mode="constant")
    f=estimate(a,b,min_snr=1.05)
    assert f.valid.mean() > .8
    assert np.nanmedian(np.hypot(f.u-dx,f.v-dy)) < .22


def test_blank_images_are_rejected():
    f=estimate(np.zeros((64,64)),np.zeros((64,64)))
    assert not f.valid.any()
    assert np.isnan(f.u).all()


def test_mask_and_calibration():
    a,b,*_,mask=generate_pair(Config())
    f=estimate(a,b,mask)
    u,v=physical_velocity(f,2.,.1)
    np.testing.assert_allclose(u,20*f.u,equal_nan=True)
    np.testing.assert_allclose(v,20*f.v,equal_nan=True)
    assert (~f.eligible).any()
    with pytest.raises(ValueError): physical_velocity(f,1,0)


@pytest.mark.parametrize("regime",["radial","circulation","mixed"])
def test_synthetic_recovery(regime):
    feat,error,f=sample(Config(regime=regime))
    assert np.isfinite(feat).all()
    assert error < .4
    assert f.valid.sum()/f.eligible.sum() > .7


def test_determinism():
    for a,b in zip(generate_pair(Config(seed=19)),generate_pair(Config(seed=19))):
        np.testing.assert_array_equal(a,b)


def test_split_never_shares_run():
    y=np.repeat([0,1,2],24)
    groups=np.repeat(np.arange(24),3)
    train,test=split_groups(y,groups)
    assert not set(groups[train]) & set(groups[test])
    assert set(y[test])=={0,1,2}
    assert len(train)+len(test)==len(y)


def test_invalid_inputs():
    with pytest.raises(ValueError): Config(regime="unknown")
    with pytest.raises(ValueError): estimate(np.zeros((64,64)),np.zeros((65,64)))
    with pytest.raises(ValueError): estimate(np.zeros((64,64)),np.zeros((64,64)),window=7)
