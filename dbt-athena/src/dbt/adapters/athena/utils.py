import json
import re
from enum import Enum
from typing import Any, Generator, List, Optional, TypeVar

from mypy_boto3_athena.type_defs import DataCatalogTypeDef, GetWorkGroupOutputTypeDef

from dbt.adapters.athena.constants import LOGGER


def clean_sql_comment(comment: str) -> str:
    split_and_strip = [line.strip() for line in comment.split("\n")]
    return " ".join(line for line in split_and_strip if line)


def stringify_table_parameter_value(value: Any) -> Optional[str]:
    """Convert any variable to string for Glue Table property."""
    try:
        if isinstance(value, (dict, list)):
            value_str: str = json.dumps(value)
        else:
            value_str = str(value)
        return value_str[:512000]
    except (TypeError, ValueError) as e:
        # Handle non-stringifiable objects and non-serializable objects
        LOGGER.warning(f"Non-stringifiable object. Error: {str(e)}")
        return None


def is_valid_table_parameter_key(key: str) -> bool:
    """Check if key is valid for Glue Table property according to official documentation."""
    # Simplified version of key pattern which works with re
    # Original pattern can be found here https://docs.aws.amazon.com/glue/latest/webapi/API_Table.html
    key_pattern: str = r"^[\u0020-\uD7FF\uE000-\uFFFD\t]*$"
    return len(key) <= 255 and bool(re.match(key_pattern, key))


def get_catalog_id(catalog: Optional[DataCatalogTypeDef]) -> Optional[str]:
    return (
        catalog["Parameters"]["catalog-id"]
        if catalog and catalog["Type"] == AthenaCatalogType.GLUE.value
        else None
    )


class AthenaCatalogType(Enum):
    GLUE = "GLUE"
    LAMBDA = "LAMBDA"
    HIVE = "HIVE"


def get_catalog_type(catalog: Optional[DataCatalogTypeDef]) -> Optional[AthenaCatalogType]:
    return AthenaCatalogType(catalog["Type"]) if catalog else None


T = TypeVar("T")


def get_chunks(lst: List[T], n: int) -> Generator[List[T], None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def ellipsis_comment(s: str, max_len: int = 255) -> str:
    """Ellipsis string if it exceeds max length"""
    return f"{s[:(max_len - 3)]}..." if len(s) > max_len else s


def get_spark_engine_version(work_group_config: GetWorkGroupOutputTypeDef) -> Optional[str]:
    """
    Get Spark engine version from workgroup configuration.

    Args:
        work_group_config: The GetWorkGroup API response containing workgroup configuration

    Returns:
        Version string if found, None otherwise
    """
    try:
        work_group = work_group_config.get("WorkGroup", {})
        configuration = work_group.get("Configuration", {})
        engine_version = configuration.get("EngineVersion", {})

        # Prefer EffectiveEngineVersion, fallback to SelectedEngineVersion
        version = engine_version.get("EffectiveEngineVersion") or engine_version.get(
            "SelectedEngineVersion"
        )

        if version:
            LOGGER.debug(f"Extracted Spark engine version: {version}")
            return version
        else:
            LOGGER.debug("No engine version found in workgroup configuration")
            return None
    except (KeyError, AttributeError, TypeError) as e:
        LOGGER.warning(f"Failed to extract Spark engine version from workgroup config: {e}")
        return None


def is_spark_35(version_string: Optional[str]) -> bool:
    """
    Determine if Spark version is 3.5 (serverless) based on version string.

    Args:
        version_string: Version string from workgroup config, e.g.:
            - "Apache Spark version 3.5" (serverless, returns True)
            - "PySpark engine version 3" (legacy, returns False)
            - None or unknown (returns False for safety/backward compatibility)

    Returns:
        True if version is Spark 3.5 (serverless), otherwise False
    """
    if not version_string:
        LOGGER.debug("No version string provided, defaulting to legacy (DPU) configuration")
        return False

    try:
        # Extract major and minor version numbers using regex
        version_match = re.search(r"(\d+)\.(\d+)", version_string)

        if version_match:
            major = int(version_match.group(1))
            minor = int(version_match.group(2))

            is_35 = major == 3 and minor == 5

            if is_35:
                LOGGER.debug(f"Detected Spark 3.5 serverless version from: {version_string}. ")
            else:
                LOGGER.debug(f"Detected Spark {major}.{minor} from: {version_string}. ")

            return is_35
        else:
            LOGGER.debug(f"Could not parse version numbers from: {version_string}. ")
            return False
    except (ValueError, AttributeError) as e:
        LOGGER.warning(f"Error parsing version string '{version_string}': {e}. ")
        return False
