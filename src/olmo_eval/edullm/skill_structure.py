"""Validated latent-skill structures for confirmatory MIRT calibration.

This module is a provider-independent port of
``eduLLM-Evals/tutor_cat/skill_structure.py`` from ``origin/frq/infobench``
(Git blob ``0304fa75f435c711d07da0686852be236908dfed``).

A benchmark supplies a source Q-matrix.  Calibration may compare that native
structure with simpler structures that merge source skills.  Dimensions must
form a partition of the source axis, preventing a configuration typo from
silently dropping a skill or assigning one to multiple modeled dimensions.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class SkillDimension:
    """One modeled dimension and the source skills merged into it."""

    label: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        label = self.label.strip()
        members = tuple(member.strip() for member in self.members if member.strip())
        if not label or not _LABEL_RE.fullmatch(label):
            raise ValueError(
                "dimension labels must start with a letter and contain only "
                f"letters, digits, and underscores; got {self.label!r}"
            )
        if not members:
            raise ValueError(f"dimension {label!r} must contain at least one source skill")
        if len(set(members)) != len(members):
            raise ValueError(f"dimension {label!r} repeats a source skill: {members}")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "members", members)


@dataclass(frozen=True)
class SkillStructure:
    """A named partition of an ordered source-skill axis."""

    name: str
    source_skills: tuple[str, ...]
    dimensions: tuple[SkillDimension, ...]

    def __post_init__(self) -> None:
        name = self.name.strip()
        source = tuple(skill.strip() for skill in self.source_skills if skill.strip())
        dimensions = tuple(self.dimensions)
        if not name:
            raise ValueError("skill-structure name cannot be blank")
        if not source:
            raise ValueError("source_skills cannot be empty")
        if len(set(source)) != len(source):
            raise ValueError(f"source_skills contains duplicates: {source}")
        if not dimensions:
            raise ValueError("a skill structure needs at least one modeled dimension")

        labels = [dimension.label for dimension in dimensions]
        if len(set(labels)) != len(labels):
            raise ValueError(f"modeled dimension labels must be unique: {labels}")

        flattened = [member for dimension in dimensions for member in dimension.members]
        unknown = sorted(set(flattened) - set(source))
        missing = [skill for skill in source if skill not in flattened]
        repeated = sorted({skill for skill in flattened if flattened.count(skill) > 1})
        problems: list[str] = []
        if unknown:
            problems.append(f"unknown source skills {unknown}")
        if missing:
            problems.append(f"unassigned source skills {missing}")
        if repeated:
            problems.append(f"source skills assigned more than once {repeated}")
        if problems:
            raise ValueError("invalid skill partition: " + "; ".join(problems))

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "source_skills", source)
        object.__setattr__(self, "dimensions", dimensions)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return modeled dimension labels in their declared order."""

        return tuple(dimension.label for dimension in self.dimensions)

    @property
    def n_dims(self) -> int:
        """Return the number of modeled dimensions."""

        return len(self.dimensions)

    @property
    def groups(self) -> dict[str, list[str]]:
        """Return modeled labels mapped to their source-skill members."""

        return {dimension.label: list(dimension.members) for dimension in self.dimensions}

    @classmethod
    def identity(cls, source_skills: Sequence[str], name: str | None = None) -> SkillStructure:
        """Build a structure with one modeled dimension per source skill."""

        source = tuple(source_skills)
        return cls(
            name=name or f"identity_{len(source)}d",
            source_skills=source,
            dimensions=tuple(SkillDimension(skill, (skill,)) for skill in source),
        )

    @classmethod
    def from_groups(
        cls,
        name: str,
        source_skills: Sequence[str],
        groups: Iterable[tuple[str, Sequence[str]]],
    ) -> SkillStructure:
        """Build and validate a structure from explicit labeled groups."""

        return cls(
            name=name,
            source_skills=tuple(source_skills),
            dimensions=tuple(
                SkillDimension(str(label), tuple(str(member) for member in members))
                for label, members in groups
            ),
        )

    def transform_q(self, q_source: np.ndarray) -> np.ndarray:
        """Return the modeled Q-matrix by OR-merging source-skill columns."""

        q_raw = np.asarray(q_source)
        if q_raw.ndim != 2 or q_raw.shape[1] != len(self.source_skills):
            raise ValueError(
                "source Q must have shape (n_items, n_source_skills); "
                f"got {q_raw.shape}, expected second dimension {len(self.source_skills)}"
            )
        if not np.isin(q_raw, (0, 1)).all():
            raise ValueError("source Q must contain only 0/1 values")
        q = q_raw.astype(int, copy=False)
        index = {skill: idx for idx, skill in enumerate(self.source_skills)}
        columns = [
            (q[:, [index[member] for member in dimension.members]].sum(axis=1) > 0).astype(int)
            for dimension in self.dimensions
        ]
        modeled = np.stack(columns, axis=1)
        # A valid partition preserves whether each item maps to any skill.
        if not np.array_equal(q.sum(axis=1) > 0, modeled.sum(axis=1) > 0):
            raise AssertionError("skill-structure transform changed all-zero Q-row status")
        return modeled

    def as_dict(self) -> dict:
        """Return a JSON-serializable representation with stable ordering."""

        return {
            "name": self.name,
            "source_skills": list(self.source_skills),
            "dimensions": [
                {
                    "label": dimension.label,
                    "members": list(dimension.members),
                }
                for dimension in self.dimensions
            ],
        }


def parse_dimension_spec(
    spec: str,
    source_skills: Sequence[str],
    *,
    name: str = "custom",
) -> SkillStructure:
    """Parse ``label=skill+skill,label=skill`` into a validated structure."""

    groups: list[tuple[str, tuple[str, ...]]] = []
    for raw_group in spec.split(","):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        if "=" not in raw_group:
            raise ValueError(
                f"each --dimensions group must use label=skill[+skill...]; got {raw_group!r}"
            )
        label, raw_members = raw_group.split("=", 1)
        members = tuple(member.strip() for member in raw_members.split("+") if member.strip())
        groups.append((label.strip(), members))
    if not groups:
        raise ValueError("--dimensions did not define any modeled dimensions")
    return SkillStructure.from_groups(name, source_skills, groups)
