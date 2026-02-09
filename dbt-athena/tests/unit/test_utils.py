import pytest

from dbt.adapters.athena.utils import (
    clean_sql_comment,
    ellipsis_comment,
    get_chunks,
    get_spark_engine_version,
    is_spark_35,
    is_valid_table_parameter_key,
    stringify_table_parameter_value,
)


def test_clean_comment():
    assert (
        clean_sql_comment(
            """
       my long comment
         on several lines
        with weird spaces and indents.
    """
        )
        == "my long comment on several lines with weird spaces and indents."
    )


def test_stringify_table_parameter_value():
    class NonStringifiableObject:
        def __str__(self):
            raise ValueError("Non-stringifiable object")

    assert stringify_table_parameter_value(True) == "True"
    assert stringify_table_parameter_value(123) == "123"
    assert stringify_table_parameter_value("dbt-athena") == "dbt-athena"
    assert stringify_table_parameter_value(["a", "b", 3]) == '["a", "b", 3]'
    assert stringify_table_parameter_value({"a": 1, "b": "c"}) == '{"a": 1, "b": "c"}'
    assert len(stringify_table_parameter_value("a" * 512001)) == 512000
    assert stringify_table_parameter_value(NonStringifiableObject()) is None
    assert stringify_table_parameter_value([NonStringifiableObject()]) is None


def test_is_valid_table_parameter_key():
    assert is_valid_table_parameter_key("valid_key") is True
    assert is_valid_table_parameter_key("Valid Key 123*!") is True
    assert is_valid_table_parameter_key("invalid \n key") is False
    assert is_valid_table_parameter_key("long_key" * 100) is False


def test_get_chunks_empty():
    assert len(list(get_chunks([], 5))) == 0


def test_get_chunks_uneven():
    chunks = list(get_chunks([1, 2, 3], 2))
    assert chunks[0] == [1, 2]
    assert chunks[1] == [3]
    assert len(chunks) == 2


def test_get_chunks_more_elements_than_chunk():
    chunks = list(get_chunks([1, 2, 3], 4))
    assert chunks[0] == [1, 2, 3]
    assert len(chunks) == 1


@pytest.mark.parametrize(
    ("max_len", "expected"),
    (
        pytest.param(12, "abc def ghi", id="ok string"),
        pytest.param(6, "abc...", id="ellipsis"),
    ),
)
def test_ellipsis_comment(max_len, expected):
    assert expected == ellipsis_comment("abc def ghi", max_len=max_len)


class TestGetSparkEngineVersion:
    """Tests for get_spark_engine_version function."""

    def test_effective_engine_version(self):
        """Test extraction of EffectiveEngineVersion."""
        work_group_config = {
            "WorkGroup": {
                "Configuration": {
                    "EngineVersion": {
                        "EffectiveEngineVersion": "Apache Spark version 3.5",
                        "SelectedEngineVersion": "AUTO",
                    }
                }
            }
        }
        assert get_spark_engine_version(work_group_config) == "Apache Spark version 3.5"

    def test_selected_engine_version_fallback(self):
        """Test fallback to SelectedEngineVersion when EffectiveEngineVersion is missing."""
        work_group_config = {
            "WorkGroup": {
                "Configuration": {
                    "EngineVersion": {"SelectedEngineVersion": "PySpark engine version 3"}
                }
            }
        }
        assert get_spark_engine_version(work_group_config) == "PySpark engine version 3"

    def test_missing_engine_version(self):
        """Test returns None when engine version is missing."""
        work_group_config = {"WorkGroup": {"Configuration": {}}}
        assert get_spark_engine_version(work_group_config) is None

    def test_empty_config(self):
        """Test returns None for empty configuration."""
        work_group_config = {}
        assert get_spark_engine_version(work_group_config) is None

    def test_malformed_config(self):
        """Test handles malformed configuration gracefully."""
        work_group_config = {"WorkGroup": {"Configuration": {"EngineVersion": None}}}
        assert get_spark_engine_version(work_group_config) is None

    def test_nested_key_missing(self):
        """Test handles missing nested keys gracefully."""
        work_group_config = {"WorkGroup": None}
        assert get_spark_engine_version(work_group_config) is None


class TestIsSpark35:
    """Tests for is_spark_35 function."""

    @pytest.mark.parametrize(
        ("version_string", "expected"),
        (
            pytest.param("Apache Spark version 3.5", True, id="spark_35_apache"),
            pytest.param("Apache Spark version 3.5.0", True, id="spark_35_patch"),
            pytest.param("Spark 3.5", True, id="spark_35_short"),
            pytest.param("PySpark engine version 3", False, id="pyspark_3_legacy"),
            pytest.param("PySpark engine version 3.0", False, id="pyspark_30_legacy"),
            pytest.param("Apache Spark version 3.4", False, id="spark_34"),
            pytest.param("Apache Spark version 3.6", False, id="spark_36"),
            pytest.param("Apache Spark version 4.0", False, id="spark_40"),
            pytest.param("Apache Spark version 2.5", False, id="spark_25"),
            pytest.param(None, False, id="none_version"),
            pytest.param("", False, id="empty_string"),
            pytest.param("invalid version string", False, id="no_version_numbers"),
            pytest.param("version 3", False, id="missing_minor_version"),
        ),
    )
    def test_version_detection(self, version_string, expected):
        """Test version detection for various version strings."""
        assert is_spark_35(version_string) == expected

    def test_none_defaults_to_legacy(self):
        """Test that None version defaults to legacy (False) for backward compatibility."""
        assert is_spark_35(None) is False

    def test_unknown_version_defaults_to_legacy(self):
        """Test that unknown versions default to legacy (False) for safety."""
        assert is_spark_35("unknown version format") is False
