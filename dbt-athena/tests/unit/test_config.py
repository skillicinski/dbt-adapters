import importlib.metadata
from unittest.mock import Mock

import pytest

from dbt.adapters.athena.config import AthenaSparkSessionConfig, get_boto3_config


class TestConfig:
    def test_get_boto3_config(self):
        importlib.metadata.version = Mock(return_value="2.4.6")
        num_boto3_retries = 5
        get_boto3_config.cache_clear()
        config = get_boto3_config(num_retries=num_boto3_retries)
        assert config._user_provided_options["user_agent_extra"] == "dbt-athena/2.4.6"
        assert config.retries == {"max_attempts": num_boto3_retries, "mode": "standard"}


class TestAthenaSparkSessionConfig:
    """
    A class to test AthenaSparkSessionConfig
    """

    @pytest.fixture
    def spark_config(self, request):
        """
        Fixture for providing Spark configuration parameters.

        This fixture returns a dictionary containing the Spark configuration parameters. The parameters can be
        customized using the `request.param` object. The default values are:
        - `timeout`: 7200 seconds
        - `polling_interval`: 5 seconds
        - `engine_config`: {"CoordinatorDpuSize": 1, "MaxConcurrentDpus": 2, "DefaultExecutorDpuSize": 1}

        Args:
            self: The test class instance.
            request: The pytest request object.

        Returns:
            dict: The Spark configuration parameters.

        """
        return {
            "timeout": request.param.get("timeout", 7200),
            "polling_interval": request.param.get("polling_interval", 5),
            "engine_config": request.param.get(
                "engine_config",
                {"CoordinatorDpuSize": 1, "MaxConcurrentDpus": 2, "DefaultExecutorDpuSize": 1},
            ),
        }

    @pytest.fixture
    def spark_config_helper(self, spark_config):
        """Fixture for testing AthenaSparkSessionConfig class.

        Args:
            spark_config (dict): Fixture for default spark config.

        Returns:
            AthenaSparkSessionConfig: An instance of AthenaSparkSessionConfig class.
        """
        return AthenaSparkSessionConfig(spark_config)

    @pytest.mark.parametrize(
        "spark_config",
        [
            {"timeout": 5},
            {"timeout": 10},
            {"timeout": 20},
            {},
            pytest.param({"timeout": -1}, marks=pytest.mark.xfail),
            pytest.param({"timeout": None}, marks=pytest.mark.xfail),
        ],
        indirect=True,
    )
    def test_set_timeout(self, spark_config_helper):
        timeout = spark_config_helper.set_timeout()
        assert timeout == spark_config_helper.config.get("timeout", 7200)

    @pytest.mark.parametrize(
        "spark_config",
        [
            {"polling_interval": 5},
            {"polling_interval": 10},
            {"polling_interval": 20},
            {},
            pytest.param({"polling_interval": -1}, marks=pytest.mark.xfail),
        ],
        indirect=True,
    )
    def test_set_polling_interval(self, spark_config_helper):
        polling_interval = spark_config_helper.set_polling_interval()
        assert polling_interval == spark_config_helper.config.get("polling_interval", 5)

    @pytest.mark.parametrize(
        "spark_config",
        [
            {
                "engine_config": {
                    "CoordinatorDpuSize": 1,
                    "MaxConcurrentDpus": 2,
                    "DefaultExecutorDpuSize": 1,
                }
            },
            {
                "engine_config": {
                    "CoordinatorDpuSize": 1,
                    "MaxConcurrentDpus": 2,
                    "DefaultExecutorDpuSize": 2,
                }
            },
            {},
            pytest.param({"engine_config": {"CoordinatorDpuSize": 1}}, marks=pytest.mark.xfail),
            pytest.param({"engine_config": [1, 1, 1]}, marks=pytest.mark.xfail),
        ],
        indirect=True,
    )
    def test_set_engine_config(self, spark_config_helper):
        engine_config = spark_config_helper.set_engine_config()
        diff = set(engine_config.keys()) - {
            "CoordinatorDpuSize",
            "MaxConcurrentDpus",
            "DefaultExecutorDpuSize",
            "SparkProperties",
            "AdditionalConfigs",
        }
        assert len(diff) == 0

    def test_spark_35_serverless_excludes_dpu_parameters(self):
        """Test that Spark 3.5 configuration excludes DPU parameters."""
        config = AthenaSparkSessionConfig(config={}, spark_version="Apache Spark version 3.5")
        engine_config = config.set_engine_config()

        # Should only have SparkProperties, no DPU parameters
        assert "SparkProperties" in engine_config
        assert "CoordinatorDpuSize" not in engine_config
        assert "MaxConcurrentDpus" not in engine_config
        assert "DefaultExecutorDpuSize" not in engine_config

    def test_legacy_pyspark_includes_dpu_parameters(self):
        """Test that legacy PySpark 3 configuration includes DPU parameters."""
        config = AthenaSparkSessionConfig(config={}, spark_version="PySpark engine version 3")
        engine_config = config.set_engine_config()

        # Should include all DPU parameters
        assert "CoordinatorDpuSize" in engine_config
        assert "MaxConcurrentDpus" in engine_config
        assert "DefaultExecutorDpuSize" in engine_config
        assert "SparkProperties" in engine_config

    def test_unknown_version_defaults_to_legacy(self):
        """Test that unknown version defaults to legacy (DPU) configuration."""
        config = AthenaSparkSessionConfig(config={}, spark_version=None)
        engine_config = config.set_engine_config()

        # Should include DPU parameters for backward compatibility
        assert "CoordinatorDpuSize" in engine_config
        assert "MaxConcurrentDpus" in engine_config
        assert "DefaultExecutorDpuSize" in engine_config

    def test_spark_35_with_custom_spark_properties(self):
        """Test that Spark 3.5 can use custom SparkProperties without DPU parameters."""
        config = AthenaSparkSessionConfig(
            config={
                "engine_config": {
                    "SparkProperties": {
                        "spark.executor.memory": "4g",
                        "spark.driver.memory": "2g",
                    }
                }
            },
            spark_version="Apache Spark version 3.5",
        )
        engine_config = config.set_engine_config()

        assert "SparkProperties" in engine_config
        assert engine_config["SparkProperties"]["spark.executor.memory"] == "4g"
        assert engine_config["SparkProperties"]["spark.driver.memory"] == "2g"
        # Ensure no DPU parameters are present
        assert "CoordinatorDpuSize" not in engine_config

    def test_spark_35_rejects_invalid_dpu_keys(self):
        """Test that Spark 3.5 rejects DPU parameters if provided."""
        config = AthenaSparkSessionConfig(
            config={"engine_config": {"MaxConcurrentDpus": 3}},
            spark_version="Apache Spark version 3.5",
        )

        # Should raise KeyError because DPU keys are not valid for Spark 3.5
        with pytest.raises(KeyError, match="engine configuration keys provided"):
            config.set_engine_config()

    def test_legacy_validates_max_concurrent_dpus(self):
        """Test that legacy configuration validates MaxConcurrentDpus >= 2."""
        config = AthenaSparkSessionConfig(
            config={"engine_config": {"MaxConcurrentDpus": 1}},
            spark_version="PySpark engine version 3",
        )

        with pytest.raises(KeyError, match="lowest value supported for MaxConcurrentDpus is 2"):
            config.set_engine_config()

    def test_spark_35_does_not_validate_max_concurrent_dpus(self):
        """Test that Spark 3.5 does not require MaxConcurrentDpus validation."""
        config = AthenaSparkSessionConfig(config={}, spark_version="Apache Spark version 3.5")
        engine_config = config.set_engine_config()

        # Should succeed without MaxConcurrentDpus validation
        assert "MaxConcurrentDpus" not in engine_config
