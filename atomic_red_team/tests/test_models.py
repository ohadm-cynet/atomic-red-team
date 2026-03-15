import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.provisional import urls
from pydantic import AnyUrl, ValidationError
from pydantic.networks import IPvAnyAddress
from ruamel.yaml import YAML

from atomic_red_team.common import atomics_path
from atomic_red_team.models import (
    Technique,
    Atomic,
    StringArg,
    IntArg,
    FloatArg,
    UrlArg,
    ManualExecutor,
    Platform,
    CommandExecutor,
    ExecutorType,
)
from atomic_red_team.validator import format_validation_error

yaml = YAML(typ="safe")
technique_yaml_root = Path(atomics_path)
min_expected_technique_yaml_files = 300
technique_yaml_files = sorted(technique_yaml_root.glob("T*/T*.yaml"))


def _compact_validation_errors(error: ValidationError):
    try:
        formatted_error = format_validation_error(error)
        return [f"{loc}: {msg}" for msg, loc in formatted_error.items()]
    except TypeError:
        # format_validation_error can raise when pydantic errors include unhashable inputs.  # noqa: E501
        pass

    try:
        errors = error.errors(include_input=False)
    except TypeError:
        errors = error.errors()

    compact_errors = []
    for issue in errors:
        loc = ".".join(str(part) for part in issue.get("loc", ()))
        msg = issue.get("msg", "Validation error")
        compact_errors.append(f"{loc}: {msg}")
    return compact_errors

executor_strategy = st.sampled_from(["powershell", "bash", "sh", "command_prompt"])

st.register_type_strategy(IPvAnyAddress, st.ip_addresses())
st.register_type_strategy(AnyUrl, urls())

alphanumeric_underscore_strategy = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    ),
    min_size=4,
)
input_args_types = [
    st.builds(
        StringArg,
        description=st.text(),
        default=st.text(),
        type=st.sampled_from(["string", "path", "String", "Path"]),
    ),
    st.builds(
        IntArg,
        description=st.text(),
        default=st.integers(),
        type=st.sampled_from(["integer", "Integer"]),
    ),
    st.builds(
        FloatArg,
        description=st.text(),
        default=st.floats(),
        type=st.sampled_from(["float", "Float"]),
    ),
    st.builds(
        UrlArg,
        description=st.text(),
        default=st.one_of(urls(), st.ip_addresses()),
        type=st.sampled_from(["url", "Url"]),
    ),
]

platforms_strategy = st.lists(st.sampled_from(list(Platform.__args__)), min_size=1)
input_arguments_strategy = st.dictionaries(
    keys=alphanumeric_underscore_strategy, values=st.one_of(*input_args_types)
)

atomics_strategy = dict(
    input_arguments=input_arguments_strategy,
    name=alphanumeric_underscore_strategy,
    description=st.text(min_size=5),
    supported_platforms=platforms_strategy,
)


def atomic_manual_executor_builder():
    def build_atomic(input_arguments, **kwargs):
        formatted_args = " ".join(
            [f"echo #{key}=#{{{key}}}" for key in input_arguments.keys()]
        )
        return Atomic(
            **kwargs,
            executor=ManualExecutor(
                name="manual", steps=f"{formatted_args} Custom steps here..."
            ),
            input_arguments=input_arguments,
        )

    return st.builds(build_atomic, **atomics_strategy)


def atomic_command_executor_builder():
    def build_atomic(input_arguments, executor_name, **kwargs):
        formatted_args = " ".join(
            [f"echo #{key}=#{{{key}}}" for key in input_arguments.keys()]
        )
        return Atomic(
            executor=CommandExecutor(
                name=executor_name,
                command=f"{formatted_args} Custom steps here...",
                elevation_required="sudo" in formatted_args,
            ),
            input_arguments=input_arguments,
            **kwargs,
        )

    return st.builds(build_atomic, executor_name=executor_strategy, **atomics_strategy)


@given(
    st.builds(
        Technique,
        attack_technique=st.integers(min_value=1000, max_value=9999).map(
            lambda x: f"T{x}"
        ),
        atomic_tests=st.lists(
            st.one_of(
                atomic_manual_executor_builder(), atomic_command_executor_builder()
            ),
            min_size=1,
        ),
    )
)
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_property(instance):
    assert isinstance(instance, Technique)
    assert len(instance.attack_technique) > 4
    assert len(instance.display_name) >= 5
    for test in instance.atomic_tests:
        assert isinstance(test, Atomic)
        assert test.executor.name in ExecutorType.__args__


def test_discovery_finds_expected_technique_yaml_files():
    assert len(technique_yaml_files) >= min_expected_technique_yaml_files, (
        f"Expected >= {min_expected_technique_yaml_files} technique YAML files under "  # noqa: E501
        f"{technique_yaml_root}/T*/T*.yaml, got {len(technique_yaml_files)}. Check discovery glob."  # noqa: E501
    )


@pytest.mark.parametrize(
    "technique_yaml_file",
    technique_yaml_files,
    ids=lambda path: str(path.relative_to(technique_yaml_root)),
)
def test_all_technique_yaml_files_parse(technique_yaml_file: Path):
    with technique_yaml_file.open("r", encoding="utf-8") as stream:
        data = yaml.load(stream)

    assert isinstance(data, dict), (
        f"{technique_yaml_file}: expected top-level YAML mapping with "
        "attack_technique/display_name/atomic_tests keys."
    )

    try:
        technique = Technique(**data)
    except ValidationError as error:
        compact_errors = _compact_validation_errors(error)
        pytest.fail(
            f"{technique_yaml_file}: validation failed\n- "
            + "\n- ".join(compact_errors)
        )

    assert technique.attack_technique == technique_yaml_file.stem, (
        f"{technique_yaml_file}: attack_technique '{technique.attack_technique}' "  # noqa: E501
        f"does not match filename stem '{technique_yaml_file.stem}'."
    )
    assert (
        len(technique.atomic_tests) > 0
    ), f"{technique_yaml_file}: expected at least one atomic test."

    for index, atomic_test in enumerate(technique.atomic_tests, start=1):
        expected_test_number = f"{technique.attack_technique}-{index}"
        assert atomic_test.test_number == expected_test_number, (
            f"{technique_yaml_file}: bad test_number at index {index}; "
            f"expected {expected_test_number}, got {atomic_test.test_number}."
        )
