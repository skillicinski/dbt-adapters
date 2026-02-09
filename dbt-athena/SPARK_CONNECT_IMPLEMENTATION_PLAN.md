# Spark Connect Implementation Plan for dbt-athena

**Status:** Prototype validated ✅
**Date:** 2026-02-09
**Context:** Add Python model support for Apache Spark 3.5 workgroups via Spark Connect

## Executive Summary

AWS Athena Spark 3.5 does not support Calculation APIs (`start_calculation_execution`), which are required for the current Python model implementation. However, Spark 3.5 supports **Spark Connect**, a client-server architecture that enables remote DataFrame operations.

**Prototype Results:** Successfully demonstrated that dbt Python models can execute via Spark Connect, with all DataFrame operations serializing to remote Athena Spark sessions.

## Background

### Current Implementation
- Python models compiled to code strings
- Submitted via `start_calculation_execution(CodeBlock=code)`
- Works with PySpark 3, **blocked in Spark 3.5**

### Spark Connect Alternative
- Session endpoint obtained via `get_session_endpoint` API
- Remote SparkSession created with gRPC connection
- Python code executes locally, Spark operations serialize remotely
- Works with Spark 3.5+

### AWS Documentation
- [Spark Connect support](https://docs.aws.amazon.com/athena/latest/ug/notebooks-spark-connect.html)
- [Release versions](https://docs.aws.amazon.com/athena/latest/ug/notebooks-spark-release-versions.html)
- Calculation APIs explicitly not supported in Spark 3.5

---

## Implementation Phases

### Phase 1: Infrastructure & Dependencies (2-3 days)

#### 1.1 Update Dependencies
**File:** `dbt-athena/pyproject.toml` or `requirements.txt`

Add:
```toml
pyspark = {version = ">=3.5.6", extras = ["connect"]}
```

Update boto3 minimum:
```toml
boto3 = ">=1.42.44"  # Required for get_session_endpoint
```

**Testing:**
- Run `pip install -e .` to verify dependencies resolve
- Verify `pyspark.sql.SparkSession.builder.remote()` is available

#### 1.2 Add Spark Connect Detection
**File:** `dbt-athena/src/dbt/adapters/athena/utils.py`

Add function to detect if Spark Connect should be used:
```python
def should_use_spark_connect(version_string: Optional[str]) -> bool:
    """
    Determine if Spark Connect should be used based on version.

    Spark 3.5+ requires Spark Connect for Python models.
    PySpark 3 uses Calculation APIs.

    Args:
        version_string: Spark version (e.g., "Apache Spark version 3.5")

    Returns:
        True if Spark Connect should be used, False for Calculation APIs
    """
    if not version_string:
        return False

    version_match = re.search(r"(\d+)\.(\d+)", version_string)
    if version_match:
        major, minor = int(version_match.group(1)), int(version_match.group(2))
        # Spark 3.5+ requires Spark Connect
        return major == 3 and minor >= 5

    return False
```

**Testing:**
- Unit tests for version detection
- Test cases: "Apache Spark version 3.5", "PySpark engine version 3", None

#### 1.3 Create Spark Connect Configuration
**File:** `dbt-athena/src/dbt/adapters/athena/spark_connect.py` (new)

Create configuration class:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SparkConnectConfig:
    """Configuration for Spark Connect connection to Athena."""

    session_id: str
    endpoint_url: str
    auth_token: str
    token_expiration: datetime

    @property
    def spark_connect_url(self) -> str:
        """Build Spark Connect URL with authentication."""
        # Convert HTTPS to sc:// protocol
        sc_url = self.endpoint_url.replace("https", "sc")
        sc_url += ":443/;use_ssl=true;"
        sc_url += f"x-aws-proxy-auth={self.auth_token}"
        return sc_url

    @property
    def is_token_expired(self) -> bool:
        """Check if auth token is expired."""
        return datetime.now(timezone.utc) >= self.token_expiration
```

---

### Phase 2: Spark Connect Job Helper (3-4 days)

#### 2.1 Create SparkConnectJobHelper Class
**File:** `dbt-athena/src/dbt/adapters/athena/spark_connect.py`

Implement new job helper:
```python
class AthenaSparkConnectJobHelper(PythonJobHelper):
    """
    Execute Python models using Spark Connect for Spark 3.5+.

    This replaces the Calculation API approach which is not supported
    in Apache Spark 3.5.
    """

    def __init__(self, parsed_model: Dict[Any, Any], credentials: AthenaCredentials):
        # Initialize similar to AthenaPythonJobHelper
        # Store session manager, config, etc.
        self._spark_session = None
        self._spark_connect_config = None

    def _get_spark_connect_config(self) -> SparkConnectConfig:
        """Get Spark Connect endpoint configuration."""
        response = self.athena_client.get_session_endpoint(
            SessionId=self.session_id
        )

        return SparkConnectConfig(
            session_id=self.session_id,
            endpoint_url=response['EndpointUrl'],
            auth_token=response['AuthToken'],
            token_expiration=response['AuthTokenExpirationTime']
        )

    def _create_spark_session(self) -> Any:
        """Create remote Spark Connect session."""
        from pyspark.sql import SparkSession

        if not self._spark_connect_config:
            self._spark_connect_config = self._get_spark_connect_config()

        # Check token expiration
        if self._spark_connect_config.is_token_expired:
            LOGGER.debug("Auth token expired, refreshing...")
            self._spark_connect_config = self._get_spark_connect_config()

        spark = SparkSession.builder \
            .remote(self._spark_connect_config.spark_connect_url) \
            .getOrCreate()

        LOGGER.debug(f"Spark Connect session created: version {spark.version}")
        return spark

    @property
    def spark_session(self) -> Any:
        """Get or create Spark Connect session."""
        if self._spark_session is None:
            self._spark_session = self._create_spark_session()
        return self._spark_session

    def submit(self, compiled_code: str) -> Any:
        """
        Execute Python model code via Spark Connect.

        The code executes locally in the dbt process, but all Spark
        DataFrame operations serialize to the remote Athena session.
        """
        if not compiled_code.strip():
            return self._empty_result()

        try:
            # Create dbt context
            dbt_context = self._create_dbt_context()

            # Execute compiled code with Spark Connect session
            namespace = {
                "dbt": dbt_context,
                "spark": self.spark_session,
                "spark_session": self.spark_session,  # Some models use this name
            }

            exec(compiled_code, namespace)

            # Get the model function and execute it
            model_func = namespace.get('model')
            if not model_func:
                raise DbtRuntimeError("Model function not found in compiled code")

            result_df = model_func(dbt_context, self.spark_session)

            # Write result to target location
            self._write_result(result_df)

            return self._success_result()

        except Exception as e:
            LOGGER.error(f"Spark Connect execution failed: {e}")
            raise DbtRuntimeError(f"Python model execution failed: {e}")

        finally:
            # Cleanup (but keep session for reuse)
            pass

    def _create_dbt_context(self):
        """Create dbt context object for model execution."""
        # Implementation similar to existing dbt context
        # Needs to support: dbt.config(), dbt.ref(), dbt.source(), dbt.this
        pass

    def _write_result(self, result_df):
        """Write result DataFrame to target table."""
        # Get target table location from dbt context
        target_table = f"{self.schema}.{self.table_name}"

        # Write using saveAsTable or write.parquet
        result_df.write.mode("overwrite").saveAsTable(target_table)

        LOGGER.debug(f"Result written to {target_table}")

    def _success_result(self) -> Dict[str, Any]:
        """Return success result matching Calculation API format."""
        return {
            "ResultS3Uri": f"s3://path/to/result",
            "ResultType": "SparkConnect",
            "StdErrorS3Uri": "",
            "StdOutS3Uri": ""
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for blank code."""
        return {
            "ResultS3Uri": "string",
            "ResultType": "string",
            "StdErrorS3Uri": "string",
            "StdOutS3Uri": "string"
        }

    def cleanup(self):
        """Stop Spark Connect session."""
        if self._spark_session:
            try:
                self._spark_session.stop()
                LOGGER.debug("Spark Connect session stopped")
            except Exception as e:
                LOGGER.warning(f"Failed to stop Spark session: {e}")
```

#### 2.2 Update Python Submissions Router
**File:** `dbt-athena/src/dbt/adapters/athena/python_submissions.py`

Update `AthenaPythonJobHelper.__init__` to route to Spark Connect:
```python
def __init__(self, parsed_model: Dict[Any, Any], credentials: AthenaCredentials) -> None:
    # ... existing version detection code ...

    # Determine execution method
    if should_use_spark_connect(spark_version):
        LOGGER.debug(f"Using Spark Connect for {spark_version}")
        # Delegate to Spark Connect helper
        # Note: May need to refactor to factory pattern
    else:
        LOGGER.debug(f"Using Calculation APIs for {spark_version}")
        # Use existing implementation
```

**Alternative:** Create factory pattern:
```python
def create_python_job_helper(
    parsed_model: Dict[Any, Any],
    credentials: AthenaCredentials
) -> PythonJobHelper:
    """Factory to create appropriate Python job helper."""

    spark_version = detect_spark_version(credentials)

    if should_use_spark_connect(spark_version):
        from dbt.adapters.athena.spark_connect import AthenaSparkConnectJobHelper
        return AthenaSparkConnectJobHelper(parsed_model, credentials)
    else:
        return AthenaPythonJobHelper(parsed_model, credentials)
```

---

### Phase 3: dbt Context Integration (2-3 days)

#### 3.1 Implement dbt.ref() for Spark Connect
**Challenge:** `dbt.ref()` needs to load referenced models as DataFrames

**Solution:** Load from Glue catalog using Spark Connect:
```python
class SparkConnectDbtContext:
    def __init__(self, spark_session, model_context):
        self.spark = spark_session
        self.model_context = model_context

    def ref(self, model_name: str):
        """Load referenced model as Spark Connect DataFrame."""
        # Get fully qualified table name from dbt's resolver
        schema = self.model_context.get("schema")
        table = f"{schema}.{model_name}"

        LOGGER.debug(f"Loading ref: {table}")
        return self.spark.table(table)

    def source(self, source_name: str, table_name: str):
        """Load source table as Spark Connect DataFrame."""
        # Resolve source location from dbt metadata
        fqn = self._resolve_source(source_name, table_name)
        return self.spark.table(fqn)

    def config(self, **kwargs):
        """Handle dbt.config() calls."""
        # Store config, may affect write strategy
        self._config.update(kwargs)

    @property
    def this(self):
        """Return target table reference."""
        return self.model_context.get("this")

    @property
    def is_incremental(self) -> bool:
        """Check if running in incremental mode."""
        return self.model_context.get("is_incremental", False)
```

#### 3.2 Integrate with Existing dbt Infrastructure
**File:** `dbt-athena/src/dbt/adapters/athena/impl.py`

Ensure adapter methods support Spark Connect:
- Table creation/updates via Spark DataFrames
- Schema operations remain unchanged (use Athena SQL)
- Incremental strategy handling

---

### Phase 4: Result Handling & Table Writes (1-2 days)

#### 4.1 DataFrame Write Strategy
**Options:**
1. `saveAsTable()` - Writes to Glue catalog directly
2. `write.parquet()` - Writes to S3, then create external table
3. Hybrid - Write parquet, let dbt create table definition

**Recommended:** Use `saveAsTable()` for simplicity

**Requirements:**
- Execution role needs Glue permissions:
  ```json
  {
    "Effect": "Allow",
    "Action": [
      "glue:GetDatabase",
      "glue:CreateDatabase",
      "glue:GetTable",
      "glue:CreateTable",
      "glue:UpdateTable",
      "glue:DeleteTable"
    ],
    "Resource": "*"
  }
  ```

#### 4.2 Handle Materialization Strategies
**Table:**
```python
result_df.write.mode("overwrite").saveAsTable(target_table)
```

**Incremental:**
```python
if is_incremental and table_exists:
    # Read existing data
    existing_df = spark.table(target_table)
    # Merge logic (append, merge, etc.)
    result_df.write.mode("append").saveAsTable(target_table)
else:
    result_df.write.mode("overwrite").saveAsTable(target_table)
```

#### 4.3 Error Handling
Wrap Spark Connect exceptions:
```python
try:
    result_df.write.saveAsTable(target_table)
except Exception as e:
    if "AccessDeniedException" in str(e):
        raise DbtRuntimeError(
            f"IAM permissions error writing to {target_table}. "
            f"Ensure execution role has glue:CreateTable permission. "
            f"Error: {e}"
        )
    else:
        raise DbtRuntimeError(f"Failed to write table {target_table}: {e}")
```

---

### Phase 5: Testing (2-3 days)

#### 5.1 Unit Tests
**File:** `dbt-athena/tests/unit/test_spark_connect.py` (new)

Test coverage:
- [ ] Spark Connect URL construction
- [ ] Auth token handling and expiration
- [ ] Version detection routing (Spark Connect vs Calculation API)
- [ ] dbt context methods (ref, source, config)
- [ ] Error handling and exception wrapping

Example:
```python
def test_spark_connect_url_construction():
    config = SparkConnectConfig(
        session_id="s-test123",
        endpoint_url="https://test.sessions.athena.us-east-1.amazonaws.com",
        auth_token="test-token-abc",
        token_expiration=datetime.now() + timedelta(hours=1)
    )

    url = config.spark_connect_url
    assert url.startswith("sc://test.sessions.athena")
    assert ":443/;use_ssl=true;" in url
    assert "x-aws-proxy-auth=test-token-abc" in url
```

#### 5.2 Integration Tests
**File:** `dbt-athena/tests/functional/adapter/test_python_submissions_spark_connect.py` (new)

Test with real Spark 3.5 workgroup:
- [ ] Basic Python model execution
- [ ] Python model with dbt.ref()
- [ ] Incremental Python models
- [ ] Error scenarios (missing permissions, invalid code)

**Prerequisites:**
- Spark 3.5 workgroup configured (already exists: `spark-test`)
- Execution role with Glue permissions
- Test dbt project with Python models

#### 5.3 Compatibility Testing
Test matrix:
- PySpark 3 workgroup → Calculation APIs (existing behavior)
- Spark 3.5 workgroup → Spark Connect (new behavior)
- No workgroup → N/A (SQL models only)

---

### Phase 6: Documentation & Release (1-2 days)

#### 6.1 Update README
**File:** `dbt-athena/README.md`

Add section:
```markdown
## Python Models with Apache Spark

dbt-athena supports Python models using AWS Athena for Apache Spark.

### Spark Version Support

- **PySpark engine version 3**: Uses Calculation APIs (legacy)
- **Apache Spark version 3.5+**: Uses Spark Connect (recommended)

### Requirements

1. **Spark-enabled workgroup** in AWS Athena
2. **Python runtime**: PySpark 3.5.6+ with Spark Connect
   ```bash
   pip install 'dbt-athena[spark]'  # Includes pyspark[connect]
   ```
3. **IAM permissions**: Execution role needs Glue access
   ```json
   {
     "Effect": "Allow",
     "Action": [
       "glue:GetDatabase",
       "glue:GetTable",
       "glue:CreateTable",
       "glue:UpdateTable"
     ],
     "Resource": "*"
   }
   ```

### Configuration

In `profiles.yml`:
```yaml
my_athena_profile:
  target: dev
  outputs:
    dev:
      type: athena
      # ... standard config ...
      spark_work_group: my-spark-workgroup  # Spark-enabled workgroup
```

### Example Python Model

```python
def model(dbt, spark):
    dbt.config(materialized='table')

    # Reference upstream models
    orders_df = dbt.ref('orders')

    # Transform with PySpark
    from pyspark.sql import functions as F
    result = orders_df.groupBy('customer_id').agg(
        F.sum('amount').alias('total_spent'),
        F.count('*').alias('order_count')
    )

    return result
```

### Spark 3.5 vs PySpark 3

| Feature | PySpark 3 | Spark 3.5+ |
|---------|-----------|------------|
| Execution | Calculation APIs | Spark Connect |
| Configuration | Requires DPU params | Serverless (no DPU) |
| Session Format | UUID | `s-xxx` format |
| boto3 Required | 1.40+ | **1.42.44+** |

### Troubleshooting

**"AccessDeniedException: glue:GetDatabase"**
- Add Glue permissions to Athena execution role

**"'Athena' object has no attribute 'get_session_endpoint'"**
- Upgrade boto3: `pip install --upgrade boto3>=1.42.44`

**"Unable to start spark python code execution"**
- Spark 3.5 requires Spark Connect (automatic)
- Ensure PySpark 3.5.6+ installed with connect extras
```

#### 6.2 Update Changelog
Already created via `changie new`, update with Spark Connect details:
```yaml
kind: Features
body: |
  Add support for Apache Spark 3.5 serverless workgroups:
  - Automatic version detection (PySpark 3 vs Spark 3.5)
  - Python models via Spark Connect for Spark 3.5+
  - Backward compatible with PySpark 3 Calculation APIs
  - Fixed session ID UUID parsing for s-xxx format
```

#### 6.3 Migration Guide
**File:** `dbt-athena/SPARK_35_MIGRATION.md` (new)

Guide for users upgrading:
```markdown
# Migrating to Apache Spark 3.5

## Overview
AWS Athena now offers Apache Spark 3.5 with serverless architecture.
dbt-athena automatically detects and uses the appropriate execution method.

## What Changed
- Spark 3.5 uses Spark Connect instead of Calculation APIs
- No DPU configuration required (serverless)
- Requires boto3 1.42.44+ and pyspark[connect]

## Upgrade Steps
1. Update dependencies:
   ```bash
   pip install --upgrade 'dbt-athena>=1.11.0' 'boto3>=1.42.44'
   ```

2. Add Glue permissions to execution role (see README)

3. Update workgroup to Spark 3.5 (or create new)

4. Test: `dbt run --select tag:python`

## Backward Compatibility
- PySpark 3 workgroups continue to work unchanged
- No changes required to existing Python models
- Automatic routing based on workgroup version

## Troubleshooting
See README.md for common issues and solutions.
```

#### 6.4 PR Description Template
```markdown
## Summary
Add support for Apache Spark 3.5 workgroups in dbt-athena Python models via Spark Connect.

## Problem
AWS Athena Spark 3.5 does not support Calculation APIs (StartCalculationExecution), which are required for the current Python model implementation. This causes Python models to fail with "Not authorized" errors on Spark 3.5 workgroups.

## Solution
Implement Spark Connect client for Python model execution on Spark 3.5+:
- Automatic version detection and routing
- Spark Connect for Spark 3.5+ (new)
- Calculation APIs for PySpark 3 (existing, unchanged)
- Backward compatible with all existing configurations

## Changes
- Add `pyspark[connect]>=3.5.6` dependency
- Update boto3 requirement to `>=1.42.44`
- New `AthenaSparkConnectJobHelper` for Spark 3.5 execution
- Version-aware routing in `python_submissions.py`
- Updated documentation and migration guide

## Testing
- [x] Unit tests for Spark Connect components
- [x] Integration tests with real Spark 3.5 workgroup
- [x] Backward compatibility tests with PySpark 3
- [x] Prototype validation successful

## Checklist
- [x] Signed CLA
- [x] Linked to issue #1607
- [x] Changelog entry created
- [x] Code quality checks pass
- [x] Unit tests pass
- [x] Documentation updated
- [x] No breaking changes

## Additional Notes
- Requires IAM Glue permissions for table writes (documented)
- Auth token expires after 1 hour (automatic refresh)
- Session management unchanged (reuse existing sessions)
```

---

## Technical Challenges & Solutions

### Challenge 1: Auth Token Expiration
**Problem:** Spark Connect auth tokens expire after 1 hour
**Solution:** Check expiration before each operation, refresh if needed

### Challenge 2: Glue Permissions
**Problem:** Table writes require Glue API access
**Solution:** Document required permissions, provide clear error messages

### Challenge 3: dbt Context Complexity
**Problem:** dbt.ref() needs to resolve and load tables
**Solution:** Use Spark's built-in catalog access via `spark.table()`

### Challenge 4: Result Format Compatibility
**Problem:** Existing code expects Calculation API result format
**Solution:** Return compatible dictionary from Spark Connect path

### Challenge 5: Dependency Conflicts
**Problem:** PySpark adds heavy dependencies
**Solution:** Make it optional: `pip install 'dbt-athena[spark]'`

---

## Success Metrics

### Functional Requirements
- [ ] Python models execute successfully on Spark 3.5 workgroups
- [ ] Python models continue working on PySpark 3 workgroups
- [ ] dbt.ref(), dbt.source(), dbt.this work correctly
- [ ] Incremental models function properly
- [ ] Error messages are clear and actionable

### Performance Requirements
- [ ] Session reuse works (no session per model)
- [ ] Auth token refresh is transparent
- [ ] Execution time comparable to Calculation APIs

### Code Quality
- [ ] All unit tests pass (target: 100% coverage of new code)
- [ ] All integration tests pass
- [ ] No mypy/flake8/black errors
- [ ] Documentation complete and accurate

---

## Rollout Plan

### Stage 1: Alpha Testing (Internal)
- Test with development workgroup
- Validate all functionality
- Fix any critical issues

### Stage 2: Beta Testing (Early Users)
- Release as pre-release version
- Gather feedback from community
- Address edge cases

### Stage 3: General Availability
- Release in dbt-athena 1.11.0
- Announce in dbt Community
- Monitor for issues

---

## Open Questions

1. **Optional Spark Connect dependency?**
   - Option A: Required dependency (simpler)
   - Option B: Optional extra `pip install 'dbt-athena[spark]'` (lighter)
   - **Recommendation:** Optional extra, most users don't need it

2. **Session lifecycle management?**
   - Should we stop Spark Connect sessions after each model?
   - Or reuse sessions across multiple models?
   - **Recommendation:** Reuse sessions (better performance)

3. **Table write strategy?**
   - Use `saveAsTable()` (simpler, requires Glue)
   - Or write parquet + create external table (more control)
   - **Recommendation:** Start with saveAsTable, add alternatives later

4. **Error handling granularity?**
   - Wrap all Spark exceptions as DbtRuntimeError?
   - Or preserve original exception types?
   - **Recommendation:** Wrap with context, include original in message

---

## References

### AWS Documentation
- [Spark Connect support](https://docs.aws.amazon.com/athena/latest/ug/notebooks-spark-connect.html)
- [Release versions](https://docs.aws.amazon.com/athena/latest/ug/notebooks-spark-release-versions.html)
- [GetSessionEndpoint API](https://docs.aws.amazon.com/athena/latest/APIReference/API_GetSessionEndpoint.html)

### Prototype
- Location: `/tmp/spark_connect_prototype.py`
- Results: All tests passed ✅
- Session: `s-00g3acf5akqt8i01` (Spark 3.5.6-amzn-1)

### Related Issues
- dbt-athena issue: [#1607](https://github.com/dbt-labs/dbt-adapters/issues/1607)
- PR: TBD

---

## Timeline Estimate

| Phase | Effort | Calendar Time |
|-------|--------|---------------|
| Phase 1: Infrastructure | 2-3 days | Week 1 |
| Phase 2: Spark Connect Helper | 3-4 days | Week 1-2 |
| Phase 3: dbt Context | 2-3 days | Week 2 |
| Phase 4: Result Handling | 1-2 days | Week 2 |
| Phase 5: Testing | 2-3 days | Week 3 |
| Phase 6: Documentation | 1-2 days | Week 3 |
| **Total** | **11-17 days** | **3 weeks** |

*Note: Assumes single developer, part-time work. Could be faster with dedicated time.*

---

## Decision Log

### 2026-02-09: Spark Connect Validated
- **Decision:** Proceed with Spark Connect implementation
- **Rationale:** Prototype successful, all DataFrame operations work
- **Alternative considered:** Wait for AWS to add Calculation APIs back
- **Risk:** Low - fallback to PySpark 3 for users needing Python models immediately

### Next Decision Point: Dependency Strategy
- Make PySpark optional extra? (Week 1)
- Impact on user installation experience
- Need to finalize before Phase 1 complete
