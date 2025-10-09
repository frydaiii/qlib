from pathlib import Path

import pytest

from qlib.contrib.data.loader import Alpha101DL


def _read_reference_expressions():
    path = Alpha101DL._DEFAULT_FILE
    lines = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if raw_line.startswith("Alpha#"):
                lines.append(raw_line)
    return lines


def test_alpha101_parses_all_expressions():
    fields, names = Alpha101DL.get_feature_config()

    assert len(fields) == 101
    assert len(names) == 101
    assert len(set(names)) == 101

    reference_lines = _read_reference_expressions()
    assert len(reference_lines) == 101

    first_expr = reference_lines[0].split(":", 1)[1].strip()
    assert fields[0] == first_expr
    assert names[0] == "ALPHA101_001"
    assert names[-1] == "ALPHA101_101"


@pytest.mark.parametrize(
    "include, exclude, expected",
    [
        ([1], None, ["ALPHA101_001"]),
        (["ALPHA101_002", "Alpha#3"], None, ["ALPHA101_002", "ALPHA101_003"]),
        ([1, 2, 3], [2], ["ALPHA101_001", "ALPHA101_003"]),
    ],
)
def test_alpha101_selection_filters(include, exclude, expected):
    fields, names = Alpha101DL.get_feature_config({"include": include, "exclude": exclude})

    assert names == expected
    assert len(fields) == len(names)


def test_alpha101_limit_parameter():
    fields, names = Alpha101DL.get_feature_config({"limit": 5})

    assert len(fields) == 5
    assert names == [f"ALPHA101_{idx:03d}" for idx in range(1, 6)]
