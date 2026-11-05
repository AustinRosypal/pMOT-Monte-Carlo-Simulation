from dataclasses import asdict
import numpy as np
import pytest
from pmot.mot_multilevel.matched_fast import config as b
from pmot.mot_multilevel.matched_fast.capture import HybridEvaluator,ADAPTIVE_COLUMNS
from pmot.mot_multilevel.matched_fast.extensions import combine_independent_batches
from pmot.mot_multilevel.matched_capture import RayEvaluator,solve_ray


def test_independent_allocation_and_budget(monkeypatch):
    monkeypatch.setattr(b,'available_discs',lambda p:list(range(800)))
    rng=np.random.default_rng(124);values=rng.lognormal(3.,.9,(32,16))
    a=b.allocation(b.MatchedPoint(0),values)
    assert a['discs']>=128 and a['rays']<=12800
    assert len(a['disc_indices'])==a['discs']
    assert not set(a['disc_indices'])&set(b.PILOT_DISCS+b.VALIDATION_DISCS)
    assert a==b.allocation(b.MatchedPoint(0),values)


def test_zero_pilot_does_not_claim_zero_error(monkeypatch):
    monkeypatch.setattr(b,'available_discs',lambda p:list(range(800)))
    a=b.allocation(b.MatchedPoint(0),np.zeros((32,16)))
    assert a['predicted_relative_half_width'] is None and not a['predicted_target_met']
    assert a['rays']==12800


def test_new_baseline_discs_exclude_all_prior_diagnostics():
    point=b.MatchedPoint(0,benchmark_index=0);a=b.available_discs(point)
    assert min(a)>b.read_json(b.STATISTICS/'migration.json')['original_baseline_max_checkpoint_disc']
    assert not set(a)&set(b.PILOT_DISCS+b.VALIDATION_DISCS)


def test_original_sources_remain_unchanged():
    prior=b.read_json(b.PREVIOUS/'campaign_manifest.json')
    assert prior['source_sha256']==b.original.source_hashes()


class SyntheticHybrid(HybridEvaluator):
    def __init__(self,adaptive_rule,fixed_rule):
        self.adaptive_rule=adaptive_rule;self.fixed_rule=fixed_rule;self.fixed_speeds=set()
    def adaptive(self,speed,dt,duration):
        row=np.zeros(15);row[:4]=[speed,dt,duration,self.adaptive_rule(speed)];return row
    def fixed(self,speed,dt,duration):
        self.fixed_speeds.add(speed)
        row=np.zeros(15);row[:4]=[speed,dt,duration,self.fixed_rule(speed)];return row


def test_shifted_capture_edges_and_islands_receive_fixed_correction():
    def adaptive(v):return int(v<=10 or 20<=v<=22)
    def fixed(v):return int(v<=10.25 or 19.75<=v<=22.25)
    e=SyntheticHybrid(adaptive,fixed);mask=e.direct_mask()
    np.testing.assert_array_equal(mask,[fixed(v) for v in b.VELOCITIES])
    assert {0.,10.25,19.75,22.25}<=e.fixed_speeds


def test_positive_timeout_never_becomes_escaped():
    e=SyntheticHybrid(lambda v:-1,lambda v:-1 if v==.25 else 0)
    with pytest.raises(RuntimeError,match='positive-flux'):e.direct_mask()


def test_adaptive_rows_are_separate_from_fixed_evidence():
    r,d,_=b.load_master_geometry();data=b.prepared(b.MatchedPoint(0))
    e=HybridEvaluator(r[0,0],d[0,0],data)
    e.adaptive(10.,5e-6,.001)
    assert len(e.adaptive_records)==1 and len(e.adaptive_records[0])==len(ADAPTIVE_COLUMNS)
    assert not e.records
    e.fixed(10.,5e-6,.001);assert len(e.records)==1


def payload(batch,discs):
    return dict(stage='production',batch_id=batch,sample_count_fixed_before_batch=True,
      configuration=asdict(b.MatchedPoint(0)),master_geometry_sha256='geometry',disc_indices=discs,
      velocity_m_per_s=[0.,1.,2.],captured_counts=[[0,1,0],[0,2,0],[0,3,0]],points_per_disc=3,disc_area_m2=.001)


def test_future_extensions_combine_independent_disc_batches():
    result=combine_independent_batches([payload('a',[0,1,2]),payload('b',[3,4,5])])
    assert result['disc_count']==6 and result['lower']<result['mean']<result['upper']
    assert sum(x['alpha_spent'] for x in result['batches'])<.05
    with pytest.raises(ValueError,match='independent'):combine_independent_batches([payload('a',[0,1,2]),payload('b',[2,3,4])])
    bad=payload('a',[0,1,2]);bad['stage']='pilot'
    with pytest.raises(ValueError,match='pilot'):combine_independent_batches([bad])


def test_actual_runtime_input_copies_are_immutable_and_exact():
    from pmot.mot_multilevel.matched_fast.campaign import immutable_data,data_digest,verify_runtime_reference
    point=b.definitions()[0][45];original=b.prepared(point);copy=immutable_data(original)
    assert data_digest(original)==data_digest(copy)
    assert all(not a.flags.writeable for a in copy)
    assert all(not np.shares_memory(a,c) for a,c in zip(original,copy))
    verify_runtime_reference(45,copy)
