def test_update_metadata_replaces_without_rechunking() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY, metadata={"kind": "policy", "tenant": "acme"})
    original_chunks = ingester.chunks
    updated = ingester.update_metadata("  policy-1  ", {"kind": "handbook", "tenant": "beta"})
    assert updated.source_id == "policy-1"
    assert updated.metadata == {"kind": "handbook", "tenant": "beta"}
    assert updated.text == POLICY
    # Chunks unchanged
    assert ingester.chunks == original_chunks
    # Metadata mutation on return does not affect store
    updated.metadata["kind"] = "mutated"
    again = ingester.get("policy-1")
    assert again.metadata == {"kind": "handbook", "tenant": "beta"}
    with pytest.raises(KeyError):
        ingester.update_metadata("missing", {"kind": "x"})
    with pytest.raises(ValueError):
        ingester.update_metadata("   ", {"kind": "x"})
    with pytest.raises(ValueError):
        ingester.update_metadata("policy-1", {"": : "x"})
    with pytest.raises(TypeError):
        ingester.update_metadata("policy-1", {"kind": 1})  # type: ignore[arg-type]
