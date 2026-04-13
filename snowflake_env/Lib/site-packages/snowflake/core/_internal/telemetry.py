import contextlib
import functools
import hashlib
import logging
import os
import platform

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

from typing_extensions import Concatenate, ParamSpec

from snowflake.connector import SnowflakeConnection
from snowflake.connector.telemetry import (
    TelemetryClient,
    TelemetryData,
)
from snowflake.connector.telemetry import (
    TelemetryField as ConnectorTelemetryField,
)
from snowflake.connector.time_util import get_time_millis

from .._common import ObjectCollection, ObjectReferenceMixin
from ..exceptions import SnowflakePythonError
from ..version import __version__ as VERSION
from .utils import TelemetryField, is_running_inside_stored_procedure


if TYPE_CHECKING:
    from ..cortex.chat_service import CortexChatService
    from ..cortex.embed_service import CortexEmbedService
    from ..cortex.inference_service import CortexInferenceService
    from ..cortex.lite_agent_service import CortexAgentService
    from ..task.dagv1 import DAGOperation


logger = logging.getLogger(__name__)

# Constant to decide whether we are running tests
_called_from_test = False

TelemetryEvent = dict[str, Any]


@dataclass
class _TelemetryEventBuilder:
    class_name: str
    func_name: str
    _type: Optional[str] = None
    _data: Optional[dict[str, Any]] = None

    @staticmethod
    def _get_ci_environment_type() -> str:
        if "SF_GITHUB_ACTION" in os.environ:
            return "SF_GITHUB_ACTION"
        if "GITHUB_ACTIONS" in os.environ:
            return "GITHUB_ACTIONS"
        if "GITLAB_CI" in os.environ:
            return "GITLAB_CI"
        if "CIRCLECI" in os.environ:
            return "CIRCLECI"
        if "JENKINS_URL" in os.environ or "HUDSON_URL" in os.environ:
            return "JENKINS"
        if "TF_BUILD" in os.environ:
            return "AZURE_DEVOPS"
        return "UNKNOWN"

    def _build(self) -> dict[str, Any]:
        if self._type is None:
            raise ValueError("event type not set")
        return {
            ConnectorTelemetryField.KEY_SOURCE.value: "snowflake.core",
            TelemetryField.KEY_VERSION.value: VERSION,
            TelemetryField.KEY_PYTHON_VERSION.value: platform.python_version(),
            TelemetryField.KEY_OS.value: platform.system(),
            ConnectorTelemetryField.KEY_TYPE.value: self._type,
            TelemetryField.KEY_CI_ENVIRONMENT_TYPE.value: self._get_ci_environment_type(),
            TelemetryField.KEY_DATA.value: {
                "class_name": self.class_name,
                TelemetryField.KEY_FUNC_NAME.value: self.func_name,
                **(self._data if self._data else {}),
            },
        }

    def usage_event(self) -> TelemetryEvent:
        self._type = "python_api"
        return self._build()

    def exception_event(self, exception: Exception) -> TelemetryEvent:
        from snowflake.core.exceptions import APIError

        self._type = "python_api_exception"
        self._data = {
            "exception_type": type(exception).__name__,
            "exception_sha256": hashlib.sha256(str(exception).encode()).hexdigest(),
            "is_python_api_error": isinstance(exception, SnowflakePythonError),
        }
        if isinstance(exception, APIError):
            request_info = exception.get_request_info()
            self._data["http_code"] = exception.status
            self._data["request_id"] = request_info["request_id"]
            self._data["error_code"] = request_info["error_code"]
        return self._build()


class ApiTelemetryClient:
    def __init__(self, conn: SnowflakeConnection) -> None:
        self.telemetry: Optional[TelemetryClient] = None if is_running_inside_stored_procedure() else conn._telemetry
        logger.info("telemetry client created for %r, telemetry enabled: %s", conn, bool(self.telemetry))

    def _send(self, msg: TelemetryEvent, timestamp: Optional[int] = None) -> None:
        if not self.telemetry:
            return
        if not timestamp:
            timestamp = get_time_millis()
        telemetry_data = TelemetryData(message=msg, timestamp=timestamp)
        self.telemetry.try_add_log_to_batch(telemetry_data)

    def safe_send(
        self,
        data: TelemetryEvent,
    ) -> None:
        with contextlib.suppress(Exception):
            if not self.telemetry:
                return
            self._send(data)


P = ParamSpec("P")
R = TypeVar("R")


def api_telemetry(func: Callable[Concatenate[Any, P], R]) -> Callable[Concatenate[Any, P], R]:
    @functools.wraps(func)
    def wrap(
        self: Union[
            ObjectReferenceMixin[Any],
            ObjectCollection[Any],
            "DAGOperation",
            "CortexInferenceService",
            "CortexChatService",
            "CortexEmbedService",
            "CortexAgentService",
        ],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        from ..cortex.chat_service import CortexChatService
        from ..cortex.embed_service import CortexEmbedService
        from ..cortex.inference_service import CortexInferenceService
        from ..cortex.lite_agent_service import CortexAgentService
        from ..task.dagv1 import DAGOperation

        if isinstance(self, (ObjectReferenceMixin, ObjectCollection)):
            telemetry_client = self.root._telemetry_client  # type: ignore
        elif isinstance(self, DAGOperation):
            telemetry_client = self.schema.root._telemetry_client
        elif isinstance(self, (CortexChatService, CortexInferenceService, CortexEmbedService, CortexAgentService)):
            telemetry_client = self._api._root._telemetry_client
        else:
            raise TypeError(f"unknown type {type(self)}")
        api = None
        if hasattr(self, "_api"):
            api = self._api
        elif hasattr(self, "collection") and hasattr(self.collection, "_api"):
            api = self.collection._api
        elif _called_from_test and not isinstance(self, DAGOperation):
            # DAGOperation will not be reported when the API object cannot be extracted
            #  from them. This is okay because this class will call other APIs
            #  downstream.
            raise Exception(f"cannot determine API for {self=}")
        if api is not None:
            # Cause resolution of api client, if not done beforehand
            api.api_client  # noqa: B018
        class_name = self.__class__.__name__
        func_name = func.__name__
        logger.debug(
            "calling method %s on class %s after submitting telemetry if enabled",
            func_name,
            class_name,
        )

        event_builder = _TelemetryEventBuilder(class_name=class_name, func_name=func_name)

        telemetry_client.safe_send(event_builder.usage_event())
        try:
            r = func(self, *args, **kwargs)
            return r
        except Exception as err:
            try:
                telemetry_client.safe_send(event_builder.exception_event(err))
            except Exception as telemetry_err:
                logging.debug("Failed to send telemetry: %s", telemetry_err)
            raise

    return wrap
