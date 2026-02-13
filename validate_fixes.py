#!/usr/bin/env python3
"""Validation script to check all mandatory fixes are in place."""

import sys
from pathlib import Path


def check_pyproject_entry_points():
    """Check if pyproject.toml has entry points."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return False, "pyproject.toml not found"

    content = pyproject_path.read_text()
    if "[project.scripts]" in content and "hcqa-train" in content:
        return True, "Entry points found"
    return False, "Entry points missing"


def check_no_sys_path_manipulation():
    """Check that scripts don't use sys.path.insert()."""
    files_to_check = [
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/predict.py",
        "tests/test_model.py",
        "tests/test_data.py",
        "tests/test_training.py"
    ]

    for file_path in files_to_check:
        if not Path(file_path).exists():
            continue
        content = Path(file_path).read_text()
        if "sys.path.insert" in content:
            return False, f"sys.path.insert found in {file_path}"

    return True, "No sys.path manipulation found"


def check_learnable_weights():
    """Check if hierarchical weights are learnable."""
    components_path = Path("src/hierarchical_contrastive_qa_with_adversarial_unanswerable_detection/models/components.py")
    if not components_path.exists():
        return False, "components.py not found"

    content = components_path.read_text()
    if "self.level_weights = nn.Parameter" in content:
        return True, "Learnable weights implemented"
    return False, "Learnable weights not found"


def check_yaml_no_scientific_notation():
    """Check YAML files don't use scientific notation."""
    import re

    yaml_files = ["configs/default.yaml", "configs/ablation.yaml"]

    for yaml_path in yaml_files:
        if not Path(yaml_path).exists():
            continue
        content = Path(yaml_path).read_text()
        # Check for scientific notation patterns (e.g., 1e-3, 2.5e+4)
        if re.search(r'\d+\.?\d*e[+-]?\d+', content, re.IGNORECASE):
            return False, f"Scientific notation found in {yaml_path}"

    return True, "No scientific notation in YAML"


def check_readme_length():
    """Check README is under 200 lines."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        return False, "README.md not found"

    lines = readme_path.read_text().split("\n")
    line_count = len(lines)

    if line_count <= 200:
        return True, f"README is {line_count} lines (target: <200)"
    return False, f"README is {line_count} lines (exceeds 200)"


def check_license():
    """Check LICENSE file exists with correct copyright."""
    license_path = Path("LICENSE")
    if not license_path.exists():
        return False, "LICENSE file not found"

    content = license_path.read_text()
    if "MIT License" in content and "2026 Alireza Shojaei" in content:
        return True, "LICENSE correct"
    return False, "LICENSE missing or incorrect"


def check_type_hints():
    """Check scripts have proper type hints."""
    files_to_check = [
        "scripts/evaluate.py",
        "scripts/predict.py"
    ]

    for file_path in files_to_check:
        if not Path(file_path).exists():
            continue
        content = Path(file_path).read_text()
        # Check for improved type hints
        if "-> Tuple[" not in content and "-> Dict[" not in content:
            return False, f"Type hints not improved in {file_path}"

    return True, "Type hints added"


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("VALIDATION CHECKS FOR MANDATORY FIXES")
    print("=" * 70)

    checks = [
        ("Entry points in pyproject.toml", check_pyproject_entry_points),
        ("No sys.path manipulation", check_no_sys_path_manipulation),
        ("Learnable hierarchical weights", check_learnable_weights),
        ("YAML without scientific notation", check_yaml_no_scientific_notation),
        ("README under 200 lines", check_readme_length),
        ("LICENSE file correct", check_license),
        ("Type hints added", check_type_hints),
    ]

    all_passed = True

    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n{status}: {check_name}")
            print(f"  → {message}")

            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n✗ ERROR: {check_name}")
            print(f"  → {str(e)}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED ✓")
        print("Project is ready for submission!")
        return 0
    else:
        print("SOME CHECKS FAILED ✗")
        print("Please review the failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
