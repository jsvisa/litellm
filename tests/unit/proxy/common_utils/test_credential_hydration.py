from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy.common_utils.credential_hydration import hydrate_named_credential_authoritative
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper


@pytest.mark.asyncio
async def test_authoritative_hydrate_resolves_env_refs_in_stored_values(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-hydration-test-salt")
    monkeypatch.setenv("BEDROCK_HYDRATION_TEST_KEY", "ak-account-b")
    monkeypatch.delenv("BEDROCK_HYDRATION_TEST_MISSING", raising=False)
    row = {
        "credential_name": "bedrock-account-b",
        "credential_values": {
            "aws_access_key_id": encrypt_value_helper("os.environ/BEDROCK_HYDRATION_TEST_KEY"),
            "aws_secret_access_key": encrypt_value_helper("literal-secret"),
            "aws_session_token": encrypt_value_helper("os.environ/BEDROCK_HYDRATION_TEST_MISSING"),
            "anthropic_issuer_signing_key_ref": encrypt_value_helper("os.environ/BEDROCK_HYDRATION_TEST_KEY"),
        },
        "credential_info": {"custom_llm_provider": "bedrock"},
    }
    prisma = MagicMock()
    prisma.db.litellm_credentialstable.find_unique = AsyncMock(return_value=row)

    with patch.object(litellm, "credential_list", []):  # test-quality-ok: the row under test must win over memory
        resolved = await hydrate_named_credential_authoritative("bedrock-account-b", prisma)

    assert resolved is not None
    assert resolved.credential_values == {
        "aws_access_key_id": "ak-account-b",
        "aws_secret_access_key": "literal-secret",
        "aws_session_token": "os.environ/BEDROCK_HYDRATION_TEST_MISSING",
        "anthropic_issuer_signing_key_ref": "os.environ/BEDROCK_HYDRATION_TEST_KEY",
    }


@pytest.mark.asyncio
async def test_authoritative_hydrate_returns_an_encrypted_empty_value_as_empty(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-hydration-test-salt")
    row = {
        "credential_name": "openai-wif",
        "credential_values": {
            "api_base": encrypt_value_helper(""),
            "openai_service_account_id": encrypt_value_helper("user-1"),
        },
        "credential_info": {"custom_llm_provider": "openai"},
    }
    prisma = MagicMock()
    prisma.db.litellm_credentialstable.find_unique = AsyncMock(return_value=row)

    with patch.object(litellm, "credential_list", []):  # test-quality-ok: the row under test must win over memory
        resolved = await hydrate_named_credential_authoritative("openai-wif", prisma)

    assert resolved is not None
    assert resolved.credential_values == {"api_base": "", "openai_service_account_id": "user-1"}
