import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

PDE_ROOT = (
    Path(__file__).parents[2]
    / "trinity/common/workflows/connect_the_dots/pde_discovery"
)
PACKAGE = "pde_discovery_test_modules"


def _load_module(name: str, path: Path | None = None):
    package = sys.modules.setdefault(PACKAGE, types.ModuleType(PACKAGE))
    package.__path__ = [str(PDE_ROOT)]
    module_name = f"{PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(
        module_name, path or PDE_ROOT / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


candidate = _load_module("candidate")
pde_numeric = _load_module("pde_numeric")
ground_truth = _load_module("ground_truth")
regression = _load_module("regression")
cod_utils = _load_module("cod_utils", PDE_ROOT.parent / "utils.py")
prompts = _load_module("prompts", PDE_ROOT / "prompts/__init__.py")
data_generator = _load_module(
    "data_generator",
    Path(__file__).parents[2] / "examples/research_cod/get_pde_discovery_data.py",
)

INITIAL_CONDITION_TEST_PACK_SEED = 1234
INITIAL_CONDITION_TEST_GRID_SIZE = 129
INITIAL_CONDITION_TEST_TRAJECTORY_COUNTS = (1, 3, 5, 8)
INITIAL_CONDITION_TEST_STATE_ABS_LIMIT = 3.0
INITIAL_CONDITION_TEST_SHAPE = pde_numeric.InitialConditionShapeConfig(
    mode_count_range=(1, 4),
)


def _parse_action(response: str):
    payload, parse_error = cod_utils.parse_xml_answer(
        response, {"dictionary", "candidate_equations", "uncertain_terms"}
    )
    return types.SimpleNamespace(payload=payload, parse_error=parse_error or None)


def test_xml_answer_protocol_parses_reasoning_and_sampling():
    response = """I should sample broad regions first.
    <answer>
      <sample_pde_data>
        <point_grid>
          <point x="0.2" t="0.1"/>
          <point x="0.8" t="0.9"/>
        </point_grid>
      </sample_pde_data>
    </answer>
    Additional reasoning outside the answer is ignored.
    """

    result = _parse_action(response)

    assert result.parse_error is None
    assert result.payload == {
        "action": "sample_pde_data",
        "args": {
            "point_grid": {
                "point": [
                    {"x": "0.2", "t": "0.1"},
                    {"x": "0.8", "t": "0.9"},
                ],
            }
        },
    }


def test_xml_answer_protocol_parses_all_pde_tools():
    cases = [
        (
            "<answer><summarize_pack_evidence/></answer>",
            {"action": "summarize_pack_evidence", "args": {}},
        ),
        (
            """<answer><run_sparse_regression>
            <dataset_id>merged_all</dataset_id><alpha>0.05</alpha>
            <dictionary><term>u</term><term>u**3</term></dictionary>
            </run_sparse_regression></answer>""",
            {
                "action": "run_sparse_regression",
                "args": {
                    "dataset_id": "merged_all",
                    "alpha": "0.05",
                    "dictionary": ["u", "u**3"],
                },
            },
        ),
        (
            """<answer><update_scientific_context>
            <preferred_equation>1.2*u - 0.8*u**3</preferred_equation>
            <note>Cubic support is preferred.</note>
            <uncertain_terms><term>sin(u)</term></uncertain_terms>
            </update_scientific_context></answer>""",
            {
                "action": "update_scientific_context",
                "args": {
                    "preferred_equation": "1.2*u - 0.8*u**3",
                    "note": "Cubic support is preferred.",
                    "uncertain_terms": ["sin(u)"],
                },
            },
        ),
    ]

    for response, expected in cases:
        result = _parse_action(response)
        assert result.parse_error is None
        assert result.payload == expected


def test_xml_answer_protocol_parses_candidate_list():
    equations = "".join(f"<equation>{idx}*u</equation>" for idx in range(8))
    response = (
        "<answer><run_sparse_regression><candidate_equations>"
        f"{equations}</candidate_equations></run_sparse_regression></answer>"
    )

    result = _parse_action(response)

    assert result.parse_error is None
    assert len(result.payload["args"]["candidate_equations"]) == 8


def test_xml_answer_protocol_rejects_ambiguous_or_invalid_actions():
    cases = [
        (
            "<answer><summarize_pack_evidence/></answer>"
            "<answer><summarize_pack_evidence/></answer>",
            "expected_exactly_one_answer_tag",
        ),
        (
            "<answer><sample_pde_data></answer>",
            "invalid_answer_xml",
        ),
        (
            '<answer>{"action":"summarize_pack_evidence","args":{}}</answer>',
            "invalid_answer_xml",
        ),
    ]

    for response, expected_error in cases:
        result = _parse_action(response)
        assert result.payload is None
        assert result.parse_error == expected_error


def test_system_prompt_requires_one_terminal_xml_action():
    system_prompt = prompts.load_system_prompt(
        dictionary_terms=["u", "u**3"],
        support_size_hint="1 to 3",
    )

    assert system_prompt.count("## Response protocol") == 1
    assert "one <answer>...</answer> block wrapping exactly one action" in system_prompt
    assert "Never write" in system_prompt
    assert "after the closing </answer>" in system_prompt


def test_candidate_coefficients_match_rendered_equation():
    coefficients = {
        "u": -1.25,
        "exp(u)-1": 0.5,
        "log(1+u**2)": -0.25,
    }
    equation = candidate.format_equation(coefficients)

    assert candidate.candidate_coefficients(equation, candidate.DEFAULT_DICTIONARY) == coefficients
    assert candidate.candidate_coefficients("f(u)=1*u+2*u", ["u"]) == {"u": 3.0}
    assert candidate.candidate_coefficients(
        "1*u+999*unknown", candidate.DEFAULT_DICTIONARY
    ) == {}


def test_expanded_dictionary_terms_are_finite_and_round_trip():
    terms = [
        "u**7",
        "sin(2*u)",
        "u**4/(1+u**4)",
        "u/(1+u+u**2)",
    ]
    u = np.linspace(-2.0, 2.0, 101)
    coefficients = {term: float(index + 1) for index, term in enumerate(terms)}

    for term in terms:
        assert np.all(np.isfinite(candidate.basis_values(term, u)))
    equation = candidate.format_equation(coefficients)
    assert candidate.candidate_coefficients(equation, terms) == coefficients


def test_physical_hard_ground_truth_family_uses_supported_two_term_templates():
    templates = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_hard",
    )

    assert templates
    assert all(len(template["terms"]) == 2 for template in templates)
    assert all(
        term in candidate.DEFAULT_DICTIONARY
        for template in templates
        for term in template["terms"]
    )


def test_physical_full_ground_truth_family_is_deduplicated_one_to_two_terms():
    def template_key(template):
        return (
            tuple(sorted(template["terms"])),
            tuple(sorted(template["signs"].items())),
        )

    full = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full",
    )
    full_keys = [template_key(template) for template in full]

    assert len(full_keys) == len(set(full_keys))
    assert len(full_keys) == 55
    assert {len(template["terms"]) for template in full} == {1, 2}
    assert any(set(template["terms"]) == {"u**3", "u**7"} for template in full)


def test_physical_full_train_eval_splits_are_support_disjoint():
    full = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full",
    )
    train = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full_train",
    )
    evaluation = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full_eval",
    )

    def template_key(template):
        return (
            tuple(sorted(template["terms"])),
            tuple(sorted(template["signs"].items())),
        )

    train_supports = {frozenset(template["terms"]) for template in train}
    eval_supports = {frozenset(template["terms"]) for template in evaluation}
    train_terms = {term for template in train for term in template["terms"]}
    eval_terms = {term for template in evaluation for term in template["terms"]}

    assert len(train) == 40
    assert len(evaluation) == 15
    assert train_supports.isdisjoint(eval_supports)
    assert train_terms == set(candidate.DEFAULT_DICTIONARY)
    assert eval_terms <= train_terms
    assert {template_key(template) for template in full} == {
        template_key(template) for template in [*train, *evaluation]
    }
    assert {len(template["terms"]) for template in train} == {1, 2}
    assert {len(template["terms"]) for template in evaluation} == {2}


def test_physical_full_eval_hard_has_unique_two_term_and_varied_three_term_supports():
    train = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full_train",
    )
    evaluation = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full_eval_hard",
    )
    train_supports = {frozenset(template["terms"]) for template in train}
    eval_supports = {frozenset(template["terms"]) for template in evaluation}
    train_terms = {term for template in train for term in template["terms"]}
    eval_terms = {term for template in evaluation for term in template["terms"]}

    assert len(evaluation) == 18
    assert train_supports.isdisjoint(eval_supports)
    assert eval_terms <= train_terms
    assert sum(len(template["terms"]) == 2 for template in evaluation) == 12
    assert sum(len(template["terms"]) == 3 for template in evaluation) == 6
    assert len(
        {
            frozenset(template["terms"])
            for template in evaluation
            if len(template["terms"]) == 2
        }
    ) == 12
    assert len(
        {
            frozenset(template["terms"])
            for template in evaluation
            if len(template["terms"]) == 3
        }
    ) == 6


def test_physical_full_eval_4000_has_held_out_balanced_supports():
    train = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full_train",
    )
    evaluation = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full_eval_4000",
    )
    train_supports = {frozenset(template["terms"]) for template in train}
    eval_supports = {frozenset(template["terms"]) for template in evaluation}
    train_terms = {term for template in train for term in template["terms"]}
    eval_terms = {term for template in evaluation for term in template["terms"]}

    assert len(evaluation) == 25
    assert len(eval_supports) == 25
    assert train_supports.isdisjoint(eval_supports)
    assert eval_terms <= train_terms
    assert sum(len(template["terms"]) == 2 for template in evaluation) == 17
    assert sum(len(template["terms"]) == 3 for template in evaluation) == 8


def test_stratified_eval_assignments_balance_every_hidden_template():
    assignments = data_generator.build_stratified_eval_template_assignments(
        test_size=288,
        eval_pack_size=8,
        eval_template_count=18,
    )

    assert assignments is not None
    assert len(assignments) == 288
    template_row_counts = {template_index: 0 for template_index in range(18)}
    for pack_index in range(36):
        pack = assignments[pack_index * 8 : (pack_index + 1) * 8]
        assert {row["eval_pack_index"] for row in pack} == {pack_index}
        assert len({row["ground_truth_template_index"] for row in pack}) == 1
        assert len({row["ground_truth_template_instance"] for row in pack}) == 1
        template_row_counts[pack[0]["ground_truth_template_index"]] += len(pack)

    assert set(template_row_counts.values()) == {16}
    assert {
        row["ground_truth_template_instance"] for row in assignments
    } == {0, 1}


def test_larger_stratified_eval_assignments_use_four_packs_per_template():
    assignments = data_generator.build_stratified_eval_template_assignments(
        test_size=576,
        eval_pack_size=8,
        eval_template_count=18,
    )

    assert assignments is not None
    assert len(assignments) == 576
    template_row_counts = Counter(
        row["ground_truth_template_index"] for row in assignments
    )
    assert set(template_row_counts) == set(range(18))
    assert set(template_row_counts.values()) == {32}
    assert {
        row["ground_truth_template_instance"] for row in assignments
    } == {0, 1, 2, 3}


def test_online_eval_subset_keeps_one_fixed_pack_per_template():
    assignments = data_generator.build_stratified_eval_template_assignments(
        test_size=576,
        eval_pack_size=8,
        eval_template_count=18,
    )

    assert assignments is not None
    for row in assignments:
        row["pde_environment_seed"] = row["eval_pack_index"]
    online_assignments = data_generator.select_eval_template_instances(
        assignments,
        eval_pack_size=8,
        eval_template_count=18,
        instances_per_template=1,
    )

    assert len(online_assignments) == 144
    assert {
        row["ground_truth_template_instance"] for row in online_assignments
    } == {0}
    template_row_counts = Counter(
        row["ground_truth_template_index"] for row in online_assignments
    )
    assert set(template_row_counts) == set(range(18))
    assert set(template_row_counts.values()) == {8}


def test_stratified_eval_assignments_reject_unbalanced_pack_counts():
    try:
        data_generator.build_stratified_eval_template_assignments(
            test_size=144,
            eval_pack_size=8,
            eval_template_count=12,
        )
    except ValueError as exc:
        assert "must be divisible" in str(exc)
    else:
        raise AssertionError("Expected unbalanced eval template coverage to fail")


def test_fixed_ground_truth_template_index_selects_requested_support():
    templates = ground_truth.load_hidden_gt_templates(
        candidate.DEFAULT_DICTIONARY,
        family="physical_full_eval_hard",
    )

    for template_index, template in enumerate(templates):
        terms, coefficients = ground_truth.sample_hidden_reaction(
            np.random.default_rng(1000 + template_index),
            templates,
            min_terms=1,
            max_terms=3,
            template_index=template_index,
        )
        assert set(terms) == set(template["terms"])
        assert set(coefficients) == set(template["terms"])

    try:
        ground_truth.sample_hidden_reaction(
            np.random.default_rng(0),
            templates,
            min_terms=1,
            max_terms=3,
            template_index=len(templates),
        )
    except ValueError as exc:
        assert "outside the eligible template range" in str(exc)
    else:
        raise AssertionError("Expected an out-of-range template index to fail")


def test_initial_condition_amplitudes_follow_trajectory_count():
    amplitude_upper = ground_truth.calibrate_family_initial_amplitude_upper(
        ground_truth.load_hidden_gt_templates(
            candidate.DEFAULT_DICTIONARY,
            family="physical_full_train",
        ),
        min_terms=1,
        max_terms=2,
        state_abs_limit=INITIAL_CONDITION_TEST_STATE_ABS_LIMIT,
    )
    for trajectory_count in INITIAL_CONDITION_TEST_TRAJECTORY_COUNTS:
        amplitudes = pde_numeric.initial_condition_amplitudes(
            INITIAL_CONDITION_TEST_PACK_SEED,
            trajectory_count,
            amplitude_upper=amplitude_upper,
        )
        band_edges = np.linspace(
            0.0,
            amplitude_upper,
            trajectory_count + 1,
        )

        assert len(amplitudes) == trajectory_count
        for amplitude, lower, upper in zip(
            sorted(amplitudes), band_edges[:-1], band_edges[1:]
        ):
            assert lower <= amplitude <= upper
        assert amplitudes == pde_numeric.initial_condition_amplitudes(
            INITIAL_CONDITION_TEST_PACK_SEED,
            trajectory_count,
            amplitude_upper=amplitude_upper,
        )


def test_initial_conditions_obey_sampled_amplitude_and_boundaries():
    x_grid = np.linspace(0.0, 1.0, INITIAL_CONDITION_TEST_GRID_SIZE)
    trajectory_count = INITIAL_CONDITION_TEST_TRAJECTORY_COUNTS[-1]
    amplitude_upper = ground_truth.calibrate_family_initial_amplitude_upper(
        ground_truth.load_hidden_gt_templates(
            candidate.DEFAULT_DICTIONARY,
            family="physical_full_eval_hard",
        ),
        min_terms=1,
        max_terms=3,
        state_abs_limit=INITIAL_CONDITION_TEST_STATE_ABS_LIMIT,
    )
    amplitudes = pde_numeric.initial_condition_amplitudes(
        INITIAL_CONDITION_TEST_PACK_SEED,
        trajectory_count,
        amplitude_upper=amplitude_upper,
    )

    for trajectory_index, amplitude in enumerate(amplitudes):
        u0 = pde_numeric.initial_condition(
            x_grid,
            INITIAL_CONDITION_TEST_PACK_SEED,
            trajectory_index,
            trajectory_count=trajectory_count,
            amplitude_upper=amplitude_upper,
            shape_config=INITIAL_CONDITION_TEST_SHAPE,
        )
        assert u0[0] == 0.0
        assert u0[-1] == 0.0
        assert np.all(u0 >= 0.0)
        assert np.isclose(float(np.max(u0)), amplitude)


def test_initial_condition_shape_config_parses_mode_range():
    config = pde_numeric.InitialConditionShapeConfig.from_mapping(
        {
            "mode_count_range": [2, 2],
        }
    )
    assert config.mode_count_range == (2, 2)


def test_reaction_residual_matches_generated_pde_at_terminal_time():
    x_grid = np.linspace(0.0, 1.0, 33)
    t_grid = np.linspace(0.0, 0.1, 101)
    dx = float(x_grid[1] - x_grid[0])
    dt = float(t_grid[1] - t_grid[0])
    ratio = dt / dx**2
    interior = len(x_grid) - 2
    reaction_fn = lambda u: 2.0 * u

    field = pde_numeric.simulate_dense_trajectory(
        x_grid=x_grid,
        t_grid=t_grid,
        dx=dx,
        dt=dt,
        lower=-ratio * np.ones(interior - 1),
        diag=(1.0 + 2.0 * ratio) * np.ones(interior),
        upper=-ratio * np.ones(interior - 1),
        trajectory_index=0,
        reaction_fn=reaction_fn,
        pack_seed=1,
        initial_amplitude_upper=1.0,
        initial_condition_shape=INITIAL_CONDITION_TEST_SHAPE,
        state_abs_limit=INITIAL_CONDITION_TEST_STATE_ABS_LIMIT,
    )

    expected = reaction_fn(field["u"][:, 1:-1])
    np.testing.assert_allclose(field["y"][:, 1:-1], expected, atol=1e-10)


def test_regression_error_matches_rendered_candidate():
    u = np.linspace(0.1, 1.0, 100)
    y = 1.23456 * u
    result = regression.sparse_regression_result(
        u=u,
        y=y,
        u_objective=u,
        y_objective=y,
        dictionary=["u"],
        alpha=0.0,
        threshold=0.0,
        max_reaction_terms=1,
        default_blind_penalty=100.0,
        kappa_threshold=100.0,
    )
    coefficients = candidate.candidate_coefficients(result["equation"], ["u"])
    actual_error = float(
        np.max(np.abs(y - candidate.reaction_value(u, coefficients)))
    )

    assert result["max_error"] == actual_error
