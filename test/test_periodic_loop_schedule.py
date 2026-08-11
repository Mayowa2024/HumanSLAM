from slam.human_slam_node import (
    apply_weak_scene_consensus_policy,
    apply_scene_only_score_policy,
    candidates_above_threshold,
    should_run_candidate_retrieval,
)
from slam.types import KeyframeRecord


def test_only_candidates_strictly_above_threshold_are_submitted():
    ranked = [
        ("high", 0.90, {}),
        ("just_over", 0.760001, {}),
        ("boundary", 0.76, {}),
        ("low", 0.40, {}),
    ]

    selected = candidates_above_threshold(ranked, 0.76, 5)

    assert [item[0] for item in selected] == ["high", "just_over"]


def test_threshold_filter_respects_response_limit():
    ranked = [(str(index), 0.9 - index * 0.01, {}) for index in range(6)]

    assert len(candidates_above_threshold(ranked, 0.76, 3)) == 3


def test_ok_tracking_runs_only_on_configured_keyframe_period():
    assert should_run_candidate_retrieval(True, 10, 2, 5)
    assert not should_run_candidate_retrieval(True, 11, 2, 5)
    assert not should_run_candidate_retrieval(False, 10, 2, 5)


def test_weak_tracking_always_runs_recovery_retrieval():
    assert should_run_candidate_retrieval(False, 11, 3, 5)
    assert should_run_candidate_retrieval(False, 12, 4, 5)


def test_period_is_clamped_to_one():
    assert should_run_candidate_retrieval(True, 7, 2, 0)


def test_qualifying_scene_only_score_is_demoted_to_0701():
    breakdown = {
        "scene_score": 0.95,
        "object_score": None,
        "text_score": None,
    }

    effective, evidence = apply_scene_only_score_policy(
        breakdown, 0.95, 0.70, 0.701
    )

    assert effective == 0.701
    assert evidence == "weak_scene_only"
    assert candidates_above_threshold([("candidate", effective, {})], 0.70, 5)


def test_scene_only_score_below_threshold_is_not_promoted():
    breakdown = {
        "scene_score": 0.68,
        "object_score": None,
        "text_score": None,
    }

    effective, evidence = apply_scene_only_score_policy(
        breakdown, 0.68, 0.70, 0.701
    )

    assert effective == 0.68
    assert evidence == "weak_scene_only"


def test_multi_layer_score_is_not_changed():
    breakdown = {
        "scene_score": 0.80,
        "object_score": 0.72,
        "text_score": None,
    }

    effective, evidence = apply_scene_only_score_policy(
        breakdown, 0.76, 0.70, 0.701
    )

    assert effective == 0.76
    assert evidence == "strong_multi_layer"


def _scene_only(frame, score):
    return (
        KeyframeRecord(
            keyframe_id=frame, timestamp=0.0, scene=None,
            source_frame_id=frame,
        ),
        score,
        {
            "scene_score": score,
            "object_score": None,
            "text_score": None,
            "raw_unified_score": score,
            "effective_score": score,
            "evidence_label": "weak_scene_only",
        },
    )


def test_weak_scene_cluster_promotes_only_top_candidate_to_cap():
    ranked = [_scene_only(frame, score) for frame, score in (
        (58, 0.597), (61, 0.554), (56, 0.544), (50, 0.513), (48, 0.486)
    )]
    selected = apply_weak_scene_consensus_policy(
        ranked, 0.70, 0.701, raw_threshold=0.55,
        top_k=5, minimum_support=3, cluster_width_frames=30,
    )
    assert selected[0][1] == 0.701
    assert selected[0][2]["evidence_label"] == "weak_scene_consensus"
    assert selected[0][2]["scene_consensus_support"] == 5
    assert len(candidates_above_threshold(selected, 0.70, 5)) == 1


def test_weak_scene_without_cluster_is_not_promoted():
    ranked = [_scene_only(frame, score) for frame, score in (
        (58, 0.60), (200, 0.59), (400, 0.58), (600, 0.57), (800, 0.56)
    )]
    selected = apply_weak_scene_consensus_policy(
        ranked, 0.70, 0.701, raw_threshold=0.55,
        top_k=5, minimum_support=3, cluster_width_frames=30,
    )
    assert selected[0][1] == 0.60
    assert not candidates_above_threshold(selected, 0.70, 5)


def test_normal_above_threshold_candidate_is_not_relabelled():
    ranked = [_scene_only(58, 0.701), _scene_only(61, 0.65), _scene_only(56, 0.64)]
    selected = apply_weak_scene_consensus_policy(ranked, 0.70, 0.701)
    assert selected == ranked
