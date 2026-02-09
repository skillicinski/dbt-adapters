import time
import uuid
from unittest.mock import Mock, patch

import botocore.exceptions
import pytest
from dbt_common.exceptions import DbtRuntimeError

from dbt.adapters.athena.connections import AthenaCredentials
from dbt.adapters.athena.python_submissions import AthenaPythonJobHelper
from dbt.adapters.athena.session import AthenaSparkSessionManager

from .constants import DATABASE_NAME


@pytest.mark.usefixtures("athena_credentials", "athena_client")
class TestAthenaPythonJobHelper:
    """
    A class to test the AthenaPythonJobHelper
    """

    @pytest.fixture
    def parsed_model(self, request):
        config: dict[str, int] = request.param.get("config", {"timeout": 1, "polling_interval": 5})

        return {
            "alias": "test_model",
            "schema": DATABASE_NAME,
            "config": {
                "timeout": config["timeout"],
                "polling_interval": config["polling_interval"],
                "engine_config": request.param.get(
                    "engine_config",
                    {"CoordinatorDpuSize": 1, "MaxConcurrentDpus": 2, "DefaultExecutorDpuSize": 1},
                ),
            },
        }

    @pytest.fixture
    def athena_spark_session_manager(self, athena_credentials, parsed_model):
        return AthenaSparkSessionManager(
            athena_credentials,
            timeout=parsed_model["config"]["timeout"],
            polling_interval=parsed_model["config"]["polling_interval"],
            engine_config=parsed_model["config"]["engine_config"],
        )

    @pytest.fixture
    def athena_job_helper(
        self,
        athena_credentials,
        athena_client,
        athena_spark_session_manager,
        parsed_model,
        monkeypatch,
    ):
        mock_job_helper = AthenaPythonJobHelper(parsed_model, athena_credentials)
        monkeypatch.setattr(mock_job_helper, "athena_client", athena_client)
        monkeypatch.setattr(mock_job_helper, "spark_connection", athena_spark_session_manager)
        return mock_job_helper

    @pytest.mark.parametrize(
        "parsed_model, session_status_response, expected_response",
        [
            (
                {"config": {"timeout": 5, "polling_interval": 5}},
                {
                    "State": "IDLE",
                },
                None,
            ),
            pytest.param(
                {"config": {"timeout": 5, "polling_interval": 5}},
                {
                    "State": "FAILED",
                },
                None,
                marks=pytest.mark.xfail,
            ),
            pytest.param(
                {"config": {"timeout": 5, "polling_interval": 5}},
                {
                    "State": "TERMINATED",
                },
                None,
                marks=pytest.mark.xfail,
            ),
            pytest.param(
                {"config": {"timeout": 1, "polling_interval": 5}},
                {
                    "State": "CREATING",
                },
                None,
                marks=pytest.mark.xfail,
            ),
        ],
        indirect=["parsed_model"],
    )
    def test_poll_session_idle(
        self,
        session_status_response,
        expected_response,
        athena_job_helper,
        athena_spark_session_manager,
        monkeypatch,
    ):
        with patch.multiple(
            athena_spark_session_manager,
            get_session_status=Mock(return_value=session_status_response),
            get_session_id=Mock(return_value="test_session_id"),
        ):

            def mock_sleep(_):
                pass

            monkeypatch.setattr(time, "sleep", mock_sleep)
            poll_response = athena_job_helper.poll_until_session_idle()
            assert poll_response == expected_response

    @pytest.mark.parametrize(
        "parsed_model, execution_status, expected_response",
        [
            (
                {"config": {"timeout": 1, "polling_interval": 5}},
                {
                    "Status": {
                        "State": "COMPLETED",
                    }
                },
                "COMPLETED",
            ),
            pytest.param(
                {"config": {"timeout": 1, "polling_interval": 5}},
                {
                    "Status": {
                        "State": "FAILED",
                    }
                },
                None,
                marks=pytest.mark.xfail,
            ),
            pytest.param(
                {"config": {"timeout": 1, "polling_interval": 5}},
                {
                    "Status": {
                        "State": "RUNNING",
                    }
                },
                "RUNNING",
                marks=pytest.mark.xfail,
            ),
        ],
        indirect=["parsed_model"],
    )
    def test_poll_execution(
        self,
        execution_status,
        expected_response,
        athena_job_helper,
        athena_spark_session_manager,
        athena_client,
        monkeypatch,
    ):
        with patch.multiple(
            athena_spark_session_manager,
            get_session_id=Mock(return_value=uuid.uuid4()),
        ):
            with patch.multiple(
                athena_client,
                get_calculation_execution=Mock(return_value=execution_status),
            ):

                def mock_sleep(_):
                    pass

                monkeypatch.setattr(time, "sleep", mock_sleep)
                poll_response = athena_job_helper.poll_until_execution_completion(
                    "test_calculation_id"
                )
                assert poll_response == expected_response

    @pytest.mark.parametrize(
        "parsed_model, test_calculation_execution_id, test_calculation_execution",
        [
            pytest.param(
                {"config": {"timeout": 1, "polling_interval": 5}},
                {"CalculationExecutionId": "test_execution_id"},
                {
                    "Result": {"ResultS3Uri": "test_results_s3_uri"},
                    "Status": {"State": "COMPLETED"},
                },
            ),
            pytest.param(
                {"config": {"timeout": 1, "polling_interval": 5}},
                {"CalculationExecutionId": "test_execution_id"},
                {"Result": {}, "Status": {"State": "FAILED"}},
                marks=pytest.mark.xfail,
            ),
            pytest.param(
                {"config": {"timeout": 1, "polling_interval": 5}},
                {},
                {"Result": {}, "Status": {"State": "FAILED"}},
                marks=pytest.mark.xfail,
            ),
        ],
        indirect=["parsed_model"],
    )
    def test_submission(
        self,
        test_calculation_execution_id,
        test_calculation_execution,
        athena_job_helper,
        athena_spark_session_manager,
        athena_client,
    ):
        with patch.multiple(
            athena_spark_session_manager, get_session_id=Mock(return_value=uuid.uuid4())
        ):
            with patch.multiple(
                athena_client,
                start_calculation_execution=Mock(return_value=test_calculation_execution_id),
                get_calculation_execution=Mock(return_value=test_calculation_execution),
            ):
                with patch.multiple(
                    athena_job_helper, poll_until_session_idle=Mock(return_value="IDLE")
                ):
                    result = athena_job_helper.submit("hello world")
                    assert result == test_calculation_execution["Result"]


class TestSessionStateErrorHandling:
    """
    Self-contained tests for session state error handling in submit().
    These tests don't depend on the class-scoped fixtures from conftest.py.
    """

    @pytest.fixture
    def mock_credentials(self):
        """Create a mock credentials object."""
        credentials = Mock()
        credentials.aws_access_key_id = None
        credentials.aws_secret_access_key = None
        credentials.aws_session_token = None
        credentials.region_name = "us-east-1"
        credentials.aws_profile_name = None
        credentials.spark_work_group = "test-workgroup"
        credentials.poll_interval = 1
        credentials.num_retries = 3
        credentials.effective_num_retries = 3
        return credentials

    @pytest.fixture
    def mock_parsed_model(self):
        """Create a mock parsed model."""
        return {
            "alias": "test_model",
            "relation_name": "test_relation",
            "schema": "test_schema",
            "config": {
                "timeout": 10,
                "polling_interval": 1,
                "engine_config": {
                    "CoordinatorDpuSize": 1,
                    "MaxConcurrentDpus": 2,
                    "DefaultExecutorDpuSize": 1,
                },
            },
        }

    @pytest.mark.parametrize(
        "session_state",
        ["TERMINATED", "TERMINATING", "DEGRADED", "FAILED"],
    )
    def test_submit_handles_terminated_session_states(
        self, mock_credentials, mock_parsed_model, session_state
    ):
        """Test that submit() handles terminated session states by getting a new session."""
        first_session_id = str(uuid.uuid4())
        second_session_id = str(uuid.uuid4())

        # Track session_id calls
        session_id_calls = [0]

        def mock_get_session_id():
            session_id_calls[0] += 1
            if session_id_calls[0] == 1:
                return uuid.UUID(first_session_id)
            return uuid.UUID(second_session_id)

        # First call raises ClientError, second succeeds
        error_response = {
            "Error": {
                "Code": "InvalidRequestException",
                "Message": f"Session is in the {session_state} state",
            }
        }
        client_error = botocore.exceptions.ClientError(error_response, "StartCalculationExecution")

        start_calc_calls = [0]

        def mock_start_calculation(*args, **kwargs):
            start_calc_calls[0] += 1
            if start_calc_calls[0] == 1:
                raise client_error
            return {"CalculationExecutionId": "test_execution_id"}

        with patch(
            "dbt.adapters.athena.python_submissions.AthenaSparkSessionManager"
        ) as MockSessionManager:
            mock_session_manager = Mock()
            mock_session_manager.get_session_id = mock_get_session_id
            mock_session_manager.remove_terminated_session = Mock()
            mock_session_manager.set_spark_session_load = Mock()
            MockSessionManager.return_value = mock_session_manager

            # Create the helper
            helper = AthenaPythonJobHelper(mock_parsed_model, mock_credentials)

            # Mock the athena_client
            mock_athena_client = Mock()
            mock_athena_client.start_calculation_execution = mock_start_calculation
            mock_athena_client.get_calculation_execution = Mock(
                return_value={
                    "Result": {"ResultS3Uri": "test_results_s3_uri"},
                    "Status": {"State": "COMPLETED"},
                }
            )
            helper.__dict__["athena_client"] = mock_athena_client

            result = helper.submit("print('hello')")

            # Verify session was cleaned up
            mock_session_manager.remove_terminated_session.assert_called_once_with(
                first_session_id
            )
            # Verify we got a result (meaning retry worked)
            assert result == {"ResultS3Uri": "test_results_s3_uri"}
            # Verify we made two attempts
            assert start_calc_calls[0] == 2

    def test_submit_raises_for_unknown_client_error(self, mock_credentials, mock_parsed_model):
        """Test that submit() raises DbtRuntimeError for unknown ClientErrors."""
        session_id = uuid.uuid4()

        error_response = {
            "Error": {
                "Code": "UnknownException",
                "Message": "Some unexpected error occurred",
            }
        }
        client_error = botocore.exceptions.ClientError(error_response, "StartCalculationExecution")

        with patch(
            "dbt.adapters.athena.python_submissions.AthenaSparkSessionManager"
        ) as MockSessionManager:
            mock_session_manager = Mock()
            mock_session_manager.get_session_id = Mock(return_value=session_id)
            MockSessionManager.return_value = mock_session_manager

            helper = AthenaPythonJobHelper(mock_parsed_model, mock_credentials)

            mock_athena_client = Mock()
            mock_athena_client.start_calculation_execution = Mock(side_effect=client_error)
            helper.__dict__["athena_client"] = mock_athena_client

            with pytest.raises(DbtRuntimeError) as exc_info:
                helper.submit("print('hello')")

            assert "Unable to start spark python code execution" in str(exc_info.value)
            assert "ClientError" in str(exc_info.value)

    def test_submit_handles_busy_session_state(self, mock_credentials, mock_parsed_model):
        """Test that submit() continues to poll when session is BUSY."""
        session_id = uuid.uuid4()

        # First call raises BUSY error, second succeeds
        error_response = {
            "Error": {
                "Code": "InvalidRequestException",
                "Message": "Session is in the BUSY state; needs to be IDLE to accept Calculations.",
            }
        }
        client_error = botocore.exceptions.ClientError(error_response, "StartCalculationExecution")

        call_count = [0]

        def mock_start_calculation(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise client_error
            return {"CalculationExecutionId": "test_execution_id"}

        with patch(
            "dbt.adapters.athena.python_submissions.AthenaSparkSessionManager"
        ) as MockSessionManager:
            mock_session_manager = Mock()
            mock_session_manager.get_session_id = Mock(return_value=session_id)
            mock_session_manager.get_session_status = Mock(return_value={"State": "IDLE"})
            mock_session_manager.set_spark_session_load = Mock()
            MockSessionManager.return_value = mock_session_manager

            helper = AthenaPythonJobHelper(mock_parsed_model, mock_credentials)

            mock_athena_client = Mock()
            mock_athena_client.start_calculation_execution = mock_start_calculation
            mock_athena_client.get_calculation_execution = Mock(
                return_value={
                    "Result": {"ResultS3Uri": "test_results_s3_uri"},
                    "Status": {"State": "COMPLETED"},
                }
            )
            helper.__dict__["athena_client"] = mock_athena_client

            result = helper.submit("print('hello')")

            # Verify we got a result
            assert result == {"ResultS3Uri": "test_results_s3_uri"}
            # Verify we made two attempts (once failed with BUSY, then succeeded)
            assert call_count[0] == 2


class TestSparkVersionDetection:
    """Tests for Spark version detection in AthenaPythonJobHelper."""

    @patch("dbt.adapters.athena.python_submissions.get_boto3_session_from_credentials")
    def test_detects_spark_35_version(self, mock_get_boto3_session, athena_credentials):
        """Test that Spark 3.5 version is detected from workgroup."""
        mock_athena_client = Mock()
        mock_athena_client.get_work_group.return_value = {
            "WorkGroup": {
                "Configuration": {
                    "EngineVersion": {"EffectiveEngineVersion": "Apache Spark version 3.5"}
                }
            }
        }

        mock_session = Mock()
        mock_session.client.return_value = mock_athena_client
        mock_get_boto3_session.return_value = mock_session

        parsed_model = {"config": {}}
        helper = AthenaPythonJobHelper(parsed_model, athena_credentials)

        # Verify the config has Spark 3.5 version
        assert helper.config.spark_version == "Apache Spark version 3.5"

        # Verify engine config excludes DPU parameters
        engine_config = helper.engine_config
        assert "SparkProperties" in engine_config
        assert "CoordinatorDpuSize" not in engine_config
        assert "MaxConcurrentDpus" not in engine_config
        assert "DefaultExecutorDpuSize" not in engine_config

    @patch("dbt.adapters.athena.python_submissions.get_boto3_session_from_credentials")
    def test_detects_legacy_pyspark_version(self, mock_get_boto3_session, athena_credentials):
        """Test that legacy PySpark 3 version is detected from workgroup."""
        mock_athena_client = Mock()
        mock_athena_client.get_work_group.return_value = {
            "WorkGroup": {
                "Configuration": {
                    "EngineVersion": {"EffectiveEngineVersion": "PySpark engine version 3"}
                }
            }
        }

        mock_session = Mock()
        mock_session.client.return_value = mock_athena_client
        mock_get_boto3_session.return_value = mock_session

        parsed_model = {"config": {}}
        helper = AthenaPythonJobHelper(parsed_model, athena_credentials)

        # Verify the config has PySpark 3 version
        assert helper.config.spark_version == "PySpark engine version 3"

        # Verify engine config includes DPU parameters
        engine_config = helper.engine_config
        assert "CoordinatorDpuSize" in engine_config
        assert "MaxConcurrentDpus" in engine_config
        assert "DefaultExecutorDpuSize" in engine_config
        assert "SparkProperties" in engine_config

    @patch("dbt.adapters.athena.python_submissions.get_boto3_session_from_credentials")
    def test_handles_version_detection_failure(self, mock_get_boto3_session, athena_credentials):
        """Test that version detection failure defaults to legacy configuration."""
        mock_get_boto3_session.side_effect = Exception("API Error")

        parsed_model = {"config": {}}
        helper = AthenaPythonJobHelper(parsed_model, athena_credentials)

        # Should default to None (which triggers legacy behavior)
        assert helper.config.spark_version is None

        # Verify engine config includes DPU parameters (legacy fallback)
        engine_config = helper.engine_config
        assert "CoordinatorDpuSize" in engine_config
        assert "MaxConcurrentDpus" in engine_config
        assert "DefaultExecutorDpuSize" in engine_config

    @patch("dbt.adapters.athena.python_submissions.get_boto3_session_from_credentials")
    def test_handles_missing_engine_version(self, mock_get_boto3_session, athena_credentials):
        """Test handling when workgroup has no engine version configured."""
        mock_athena_client = Mock()
        mock_athena_client.get_work_group.return_value = {"WorkGroup": {"Configuration": {}}}

        mock_session = Mock()
        mock_session.client.return_value = mock_athena_client
        mock_get_boto3_session.return_value = mock_session

        parsed_model = {"config": {}}
        helper = AthenaPythonJobHelper(parsed_model, athena_credentials)

        # Should be None when version not found
        assert helper.config.spark_version is None

        # Verify engine config includes DPU parameters (legacy fallback)
        engine_config = helper.engine_config
        assert "CoordinatorDpuSize" in engine_config

    def test_skips_detection_when_no_workgroup(self):
        """Test that version detection is skipped when no spark workgroup is configured."""
        credentials = AthenaCredentials(
            database="test_db",
            schema="test_schema",
            s3_staging_dir="s3://test-bucket/",
            region_name="us-east-1",
            spark_work_group=None,  # No workgroup
        )

        parsed_model = {"config": {}}
        helper = AthenaPythonJobHelper(parsed_model, credentials)

        # Should be None when no workgroup
        assert helper.config.spark_version is None

        # Verify engine config includes DPU parameters (legacy fallback)
        engine_config = helper.engine_config
        assert "CoordinatorDpuSize" in engine_config
