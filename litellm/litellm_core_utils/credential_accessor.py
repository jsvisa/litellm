"""Utils for accessing credentials."""

from collections.abc import Mapping
from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import holds_secret_pointer
from litellm.types.utils import CredentialItem

_ENV_REF_PREFIX: Final = "os.environ/"


def _resolved_credential_value(credential_name: str, key: str, value: object) -> object:
    """The environment variable's value for an ``os.environ/<VAR>`` reference, or the value itself.

    An unresolvable reference is kept and logged rather than dropped, so a provider with optional
    params cannot silently fall back to another account's ambient credentials. ``*_ref`` federation
    fields are pointers their identity source dereferences at use time, so they stay as written.
    """
    if not isinstance(value, str) or not value.startswith(_ENV_REF_PREFIX):
        return value
    if holds_secret_pointer(key):
        return value
    resolved: Final = get_secret_str(value)
    if resolved is not None:
        return resolved
    verbose_logger.warning(
        "credential '%s' value '%s' references unset environment variable %s; keeping the reference",
        credential_name,
        key,
        value,
    )
    return value


def expand_credential_env_refs(
    credential_name: str, values: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: CredentialItem.credential_values is a mutable dict by contract
    """Resolve ``os.environ/<VAR>`` values against the environment.

    Mirrors the config loader: references resolve once, when the credential enters the process, so
    runtime reads stay plain and a rotated env var needs the credential to be re-saved or the proxy
    restarted.
    """
    return {key: _resolved_credential_value(credential_name, key, value) for key, value in values.items()}


class CredentialAccessor:
    @staticmethod
    def find_credential(credential_name: str) -> CredentialItem | None:
        return next(
            (credential for credential in litellm.credential_list if credential.credential_name == credential_name),
            None,
        )

    @staticmethod
    def get_credential_values(
        credential_name: str,
    ) -> dict[str, object]:  # mutable-ok: callers merge the returned credential map
        """Safe accessor for credentials."""

        credential: Final = CredentialAccessor.find_credential(credential_name)
        return {} if credential is None else credential.credential_values.copy()

    @staticmethod
    def upsert_credentials(credentials: list[CredentialItem]):
        """Add a credential to the list of credentials."""

        for credential in credentials:
            credential.credential_values = expand_credential_env_refs(
                credential.credential_name, credential.credential_values
            )

        credential_names: Final = [cred.credential_name for cred in litellm.credential_list]

        for credential in credentials:
            if credential.credential_name in credential_names:
                # Find and replace the existing credential in the list
                for i, existing_cred in enumerate(litellm.credential_list):
                    if existing_cred.credential_name == credential.credential_name:
                        litellm.credential_list[i] = credential
                        break
            else:
                litellm.credential_list.append(credential)
