import inspect

from hot_and_cold_memory.storage.metadata_store.base import BaseMetadataStore


def test_profile_methods_exist():
    methods = [
        "create_profile_fact",
        "get_current_profile_fact",
        "expire_profile_fact",
        "update_profile_fact_confidence",
        "list_profile_facts",
        "get_profile",
        "upsert_profile",
    ]
    for name in methods:
        assert hasattr(BaseMetadataStore, name)
        assert inspect.isfunction(getattr(BaseMetadataStore, name))
