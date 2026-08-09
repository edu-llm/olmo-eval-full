"""Parity tests adapted from the authoritative InFoBench skill structures."""

import numpy as np
import pytest

from olmo_eval.edullm.skill_structure import SkillStructure, parse_dimension_spec

INFO_SKILLS = ("content", "format", "number", "style", "linguistic")


def test_edullm_skill_structure_or_merges_source_columns() -> None:
    structure = parse_dimension_spec(
        "content_style=content+style,format=format,number_linguistic=number+linguistic",
        INFO_SKILLS,
        name="correlated_3d",
    )
    q_source = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=int,
    )

    assert structure.labels == ("content_style", "format", "number_linguistic")
    assert structure.groups == {
        "content_style": ["content", "style"],
        "format": ["format"],
        "number_linguistic": ["number", "linguistic"],
    }
    assert structure.transform_q(q_source).tolist() == [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
    ]


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("first=content+format,second=number+style", "unassigned"),
        (
            "first=content+format,second=format+number+style+linguistic",
            "more than once",
        ),
        (
            "first=content+unknown,second=format+number+style+linguistic",
            "unknown",
        ),
        ("content+format", "label=skill"),
    ],
)
def test_edullm_skill_structure_rejects_nonpartitions(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_dimension_spec(spec, INFO_SKILLS)


def test_edullm_skill_structure_identity_preserves_declared_axis() -> None:
    structure = SkillStructure.identity(INFO_SKILLS)
    q_source = np.eye(len(INFO_SKILLS), dtype=int)

    assert structure.name == "identity_5d"
    assert structure.labels == INFO_SKILLS
    assert structure.n_dims == 5
    assert np.array_equal(structure.transform_q(q_source), q_source)
    assert structure.as_dict()["source_skills"] == list(INFO_SKILLS)
