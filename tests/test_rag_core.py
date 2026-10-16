"""Unit tests for the in-memory RAG core."""
from ekap.rag import Citation, Ingester, Retriever


def test_ingest_two_documents() -> None:
    ingester = Ingester()
    ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    ingester.add(
        "handbook-1",
        "Vacation accrues at 1.5 days per month.",
        metadata={"kind": "handbook"},
    )
    assert len(ingester) == 2
    assert "policy-1" in ingester
    assert ingester.chunk_count == 2


def test_retrieve_and_cite() -> None:
    ingester = Ingester()
    ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    citations = Retriever(ingester).query_with_citations(
        "Who must approve expense reports over 500?",
        top_k=1,
    )
    assert len(citations) == 1
    assert isinstance(citations[0], Citation)
    assert citations[0].source_id == "policy-1"
    assert "manager approval" in citations[0].snippet


def test_observability_properties_empty() -> None:
    ingester = Ingester()
    assert ingester.chunk_mean_chars == 0.0
    assert ingester.chunk_max_chars == 0
    assert ingester.chunk_skewness_chars == 0.0
    assert ingester.chunk_kurtosis_chars == 0.0
    assert ingester.chunk_excess_kurtosis_chars == 0.0
    assert ingester.chunk_geometric_mean_chars == 0.0
    assert ingester.chunk_harmonic_mean_chars == 0.0
    assert ingester.chunk_quadratic_mean_chars == 0.0
    assert ingester.chunk_cubic_mean_chars == 0.0
    assert ingester.chunk_quartic_mean_chars == 0.0
    assert ingester.chunk_quintic_mean_chars == 0.0
    assert ingester.chunk_sextic_mean_chars == 0.0
    assert ingester.chunk_septic_mean_chars == 0.0
    assert ingester.chunk_octic_mean_chars == 0.0
    assert ingester.chunk_nonic_mean_chars == 0.0
    assert ingester.chunk_decic_mean_chars == 0.0
    assert ingester.chunk_undecic_mean_chars == 0.0
    assert ingester.chunk_length_histogram == {}


def test_remove_and_clear() -> None:
    ingester = Ingester()
    doc = ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    assert len(ingester) == 1
    assert ingester.chunk_count == 1
    removed = ingester.remove("policy-1")
    assert removed is doc
    assert len(ingester) == 0
    assert ingester.chunk_count == 0
    assert "policy-1" not in ingester
    ingester.add("handbook-1", "Vacation accrues at 1.5 days per month.")
    ingester.clear()
    assert len(ingester) == 0
    assert ingester.chunk_count == 0
    assert ingester.source_ids == ()


def test_chunk_kurtosis_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_kurtosis_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    ingester.add("c", "zzz")
    assert ingester.chunk_kurtosis_chars == 0.0  # fewer than 4 chunks
    ingester.add("d", "wwww")
    # lengths 1, 2, 3, 4 → raw population kurtosis = 1.64
    assert abs(ingester.chunk_kurtosis_chars - 1.64) < 1e-9
    # Zero variance (all equal) returns 0.0
    ingester.clear()
    for i in range(4):
        ingester.add(f"eq-{i}", "same")
    assert ingester.chunk_kurtosis_chars == 0.0


def test_chunk_excess_kurtosis_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_excess_kurtosis_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    ingester.add("c", "zzz")
    assert ingester.chunk_excess_kurtosis_chars == 0.0  # fewer than 4 chunks
    ingester.add("d", "wwww")
    # lengths 1, 2, 3, 4 → excess kurtosis = 1.64 - 3 = -1.36
    assert abs(ingester.chunk_excess_kurtosis_chars - (-1.36)) < 1e-9
    # Zero variance (all equal) returns 0.0
    ingester.clear()
    for i in range(4):
        ingester.add(f"eq-{i}", "same")
    assert ingester.chunk_excess_kurtosis_chars == 0.0


def test_chunk_geometric_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_geometric_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → geometric mean = sqrt(2) ≈ 1.41421356237
    assert abs(ingester.chunk_geometric_mean_chars - (2 ** 0.5)) < 1e-9
    # Zero length returns 0.0
    ingester.add("c", "")
    assert ingester.chunk_geometric_mean_chars == 0.0
    # Single positive length equals itself (within floating-point tolerance)
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_geometric_mean_chars - 5.0) < 1e-9


def test_chunk_harmonic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_harmonic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → harmonic mean = 2 / (1/1 + 1/2) = 2 / 1.5 = 4/3 ≈ 1.33333333333
    assert abs(ingester.chunk_harmonic_mean_chars - (4 / 3)) < 1e-9
    # Zero length returns 0.0
    ingester.add("c", "")
    assert ingester.chunk_harmonic_mean_chars == 0.0
    # Single positive length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_harmonic_mean_chars - 5.0) < 1e-9


def test_chunk_quadratic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_quadratic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → quadratic mean = sqrt((1**2 + 2**2)/2) = sqrt(2.5) ≈ 1.58113883008
    assert abs(ingester.chunk_quadratic_mean_chars - (2.5 ** 0.5)) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → sqrt((1+4+0)/3) = sqrt(5/3) ≈ 1.29099444873
    assert abs(ingester.chunk_quadratic_mean_chars - ((5 / 3) ** 0.5)) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_quadratic_mean_chars - 5.0) < 1e-9


def test_chunk_cubic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_cubic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → cubic mean = ((1**3 + 2**3)/2)**(1/3) = (9/2)**(1/3) ≈ 1.65096362445
    assert abs(ingester.chunk_cubic_mean_chars - ((9 / 2) ** (1.0 / 3.0))) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 8 + 0)/3)**(1/3) = 3**(1/3) ≈ 1.4422495703
    assert abs(ingester.chunk_cubic_mean_chars - (3.0 ** (1.0 / 3.0))) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_cubic_mean_chars - 5.0) < 1e-9


def test_chunk_quartic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_quartic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → quartic mean = ((1**4 + 2**4)/2)**0.25 = (17/2)**0.25 ≈ 1.7104
    assert abs(ingester.chunk_quartic_mean_chars - ((17 / 2) ** 0.25)) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 16 + 0)/3)**0.25 = (17/3)**0.25
    assert abs(ingester.chunk_quartic_mean_chars - ((17 / 3) ** 0.25)) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_quartic_mean_chars - 5.0) < 1e-9


def test_chunk_quintic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_quintic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → quintic mean = ((1**5 + 2**5)/2)**0.2 = (33/2)**0.2
    assert abs(ingester.chunk_quintic_mean_chars - ((33 / 2) ** 0.2)) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 32 + 0)/3)**0.2 = (33/3)**0.2 = 11**0.2
    assert abs(ingester.chunk_quintic_mean_chars - (11.0 ** 0.2)) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_quintic_mean_chars - 5.0) < 1e-9


def test_chunk_sextic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_sextic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → sextic mean = ((1**6 + 2**6)/2)**(1/6) = (65/2)**(1/6)
    assert abs(ingester.chunk_sextic_mean_chars - ((65 / 2) ** (1.0 / 6.0))) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 64 + 0)/3)**(1/6) = (65/3)**(1/6)
    assert abs(ingester.chunk_sextic_mean_chars - ((65 / 3) ** (1.0 / 6.0))) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_sextic_mean_chars - 5.0) < 1e-9


def test_chunk_septic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_septic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → septic mean = ((1**7 + 2**7)/2)**(1/7) = (129/2)**(1/7)
    assert abs(ingester.chunk_septic_mean_chars - ((129 / 2) ** (1.0 / 7.0))) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 128 + 0)/3)**(1/7) = (129/3)**(1/7) = 43**(1/7)
    assert abs(ingester.chunk_septic_mean_chars - (43.0 ** (1.0 / 7.0))) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_septic_mean_chars - 5.0) < 1e-9


def test_chunk_octic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_octic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → octic mean = ((1**8 + 2**8)/2)**(1/8) = (257/2)**(1/8)
    assert abs(ingester.chunk_octic_mean_chars - ((257 / 2) ** (1.0 / 8.0))) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 256 + 0)/3)**(1/8) = (257/3)**(1/8)
    assert abs(ingester.chunk_octic_mean_chars - ((257 / 3) ** (1.0 / 8.0))) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_octic_mean_chars - 5.0) < 1e-9


def test_chunk_nonic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_nonic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → nonic mean = ((1**9 + 2**9)/2)**(1/9) = (513/2)**(1/9)
    assert abs(ingester.chunk_nonic_mean_chars - ((513 / 2) ** (1.0 / 9.0))) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 512 + 0)/3)**(1/9) = (513/3)**(1/9) = 171**(1/9)
    assert abs(ingester.chunk_nonic_mean_chars - (171.0 ** (1.0 / 9.0))) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_nonic_mean_chars - 5.0) < 1e-9


def test_chunk_decic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_decic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → decic mean = ((1**10 + 2**10)/2)**(1/10) = (1025/2)**(1/10)
    assert abs(ingester.chunk_decic_mean_chars - ((1025 / 2) ** (1.0 / 10.0))) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 1024 + 0)/3)**(1/10) = (1025/3)**(1/10)
    assert abs(ingester.chunk_decic_mean_chars - ((1025 / 3) ** (1.0 / 10.0))) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_decic_mean_chars - 5.0) < 1e-9


def test_chunk_undecic_mean_chars() -> None:
    ingester = Ingester()
    assert ingester.chunk_undecic_mean_chars == 0.0
    ingester.add("a", "x")
    ingester.add("b", "yy")
    # lengths 1, 2 → undecic mean = ((1**11 + 2**11)/2)**(1/11) = (2049/2)**(1/11)
    assert abs(ingester.chunk_undecic_mean_chars - ((2049 / 2) ** (1.0 / 11.0))) < 1e-9
    # Zero length is allowed and included
    ingester.add("c", "")
    # lengths 1, 2, 0 → ((1 + 2048 + 0)/3)**(1/11) = (2049/3)**(1/11) = 683**(1/11)
    assert abs(ingester.chunk_undecic_mean_chars - (683.0 ** (1.0 / 11.0))) < 1e-9
    # Single length equals itself
    ingester.clear()
    ingester.add("solo", "hello")
    assert abs(ingester.chunk_undecic_mean_chars - 5.0) < 1e-9
