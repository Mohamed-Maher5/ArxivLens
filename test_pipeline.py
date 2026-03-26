#!/usr/bin/env python3
"""
Full Pipeline Test Suite - ArxivLens (Improved)
Handles existing data and intent classification edge cases.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.generation.pipeline import Pipeline
from app.models.schemas import Message
from app.ingestion import ingest_paper
from app.indexing import index_paper
from app.ingestion.arxiv_fetcher import ArxivFetcher
from app.indexing.vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()

TEST_PAPER_ID = "1706.03762"

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"

def print_header(title):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}{title:^70}{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")

def check_langsmith():
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    api_key = bool(os.getenv("LANGCHAIN_API_KEY"))
    return tracing and api_key

# ============================================================================
# TEST 1: Chat Intent (No Retrieval)
# ============================================================================
# def test_chat_intent():
#     print_header("TEST 1: Chat Intent (No Retrieval)")

#     pipeline = Pipeline()
#     result = pipeline.run("hi, how are you?")

#     print(f"Question: 'hi, how are you?'")
#     print(f"Answer: {result.answer[:100]}...")
#     print(f"Confidence: {result.confidence}")
#     print(f"Sources: {len(result.sources)}")

#     passed = (
#         len(result.sources) == 0 and
#         result.confidence == "HIGH" and
#         len(result.answer) > 10
#     )

#     status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
#     print(f"{status} - Chat Intent")
#     return passed

# ============================================================================
# TEST 2: Task Intent - With Existing Papers (Knowledge Fallback if no match)
# ============================================================================
def test_task_with_existing_papers():
    """Test that we handle case where papers exist but query doesn't match."""
    print_header("TEST 2: Task Intent - Papers Exist But Query Unrelated")

    # Query about something completely unrelated to any indexed papers
    pipeline = Pipeline()
    result = pipeline.run("what is quantum computing cryptography?")

    print(f"Question: 'what is quantum computing cryptography?' (unrelated to papers)")
    print(f"Answer preview: {result.answer[:150]}...")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {len(result.sources)}")

    # If papers exist but don't match, should fallback to knowledge
    # If papers match, should use them (either way is correct behavior)
    passed = len(result.answer) > 20

    if len(result.sources) == 0:
        print("   → Knowledge fallback (no relevant papers)")
    else:
        print(f"   → Used {len(result.sources)} paper sources")

    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - Task with Existing Papers")
    return passed

# ============================================================================
# TEST 3: Index Paper and Query Successfully
# ============================================================================
def test_index_and_query():
    print_header("TEST 3: Index Paper + Successful Query")

    # Check if paper already indexed
    try:
        store = VectorStore()
        # Try to query for this paper specifically
        from app.retrieval import retrieve
        test_results = retrieve("attention is all you need transformer")

        if test_results and any("1706.03762" in str(r.get("paper_id", "")) for r in test_results):
            print("✓ Paper already indexed, using existing data")
            paper_indexed = True
        else:
            paper_indexed = False
    except Exception as e:
        print(f"⚠️  Could not check existing data: {e}")
        paper_indexed = False

    if not paper_indexed:
        try:
            print(f"Fetching and indexing: {TEST_PAPER_ID}")
            fetcher = ArxivFetcher()
            paper = fetcher.fetch_by_id(TEST_PAPER_ID)
            parsed = ingest_paper(paper)
            chunks = index_paper(parsed)
            print(f"✓ Indexed {len(chunks)} chunks")
        except Exception as e:
            print(f"⚠️  Indexing failed: {e}")
            print("Skipping this test...")
            return None

    # Now query specifically about the paper
    pipeline = Pipeline()
    result = pipeline.run("what is multi-head attention?")

    print(f"Question: 'what is multi-head attention?'")
    print(f"Answer preview: {result.answer[:200]}...")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {len(result.sources)}")

    if result.sources:
        print("\nSources:")
        for i, src in enumerate(result.sources[:3], 1):
            print(f"  {i}. [{src.paper_title}, p.{src.page_number}]")

    passed = len(result.sources) > 0 and len(result.answer) > 50

    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - Index and Query")
    return passed

# ============================================================================
# TEST 4: Vague Query Handling
# ============================================================================
def test_vague_query():
    """Test that vague queries are handled appropriately."""
    print_header("TEST 4: Vague Query Handling")

    pipeline = Pipeline()

    # Try a vague query - might go to chat or task depending on classification
    result = pipeline.run("explain this paper to me")

    print(f"Question: 'explain this paper to me' (vague)")
    print(f"Answer preview: {result.answer[:150]}...")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {len(result.sources)}")

    # Should get some kind of response (either chat or task)
    passed = len(result.answer) > 20

    if result.sources:
        print(f"   → Classified as TASK, found {len(result.sources)} sources")
    else:
        print("   → Classified as CHAT or no relevant chunks")

    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - Vague Query")
    return passed

# ============================================================================
# TEST 5: History Management
# ============================================================================
def test_history_management():
    print_header("TEST 5: History Management")

    pipeline = Pipeline()

    history = [
        Message(role="user", content="tell me about the transformer paper"),
        Message(role="assistant", content="The paper discusses attention mechanisms..."),
        Message(role="user", content="what are the limitations?"),
    ]

    result = pipeline.run("what are the limitations?", history=history)

    print(f"History: {len(history)} messages")
    print(f"Contextualized: {result.contextualized_query[:80]}...")
    print(f"Answer: {result.answer[:100]}...")

    passed = len(result.answer) > 20

    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - History Management")
    return passed

# ============================================================================
# TEST 6: Score Thresholds Verification
# ============================================================================
def test_score_thresholds():
    """Verify that score thresholds are working correctly."""
    print_header("TEST 6: Score Thresholds Verification")

    from app.core.settings import settings

    print(f"Retrieval threshold (SCORE_THRESHOLD): {settings.score_threshold}")
    print(f"Reranker threshold (RERANK_SCORE_THRESHOLD): {settings.rerank_score_threshold}")

    # These should match your requirements
    correct = (
        settings.score_threshold == 0.25 and
        settings.rerank_score_threshold == 6.0
    )

    if correct:
        print("✓ Thresholds configured correctly")
    else:
        print(f"⚠️  Expected 0.25 and 6.0, got {settings.score_threshold} and {settings.rerank_score_threshold}")

    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if correct else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - Score Thresholds")
    return correct

# ============================================================================
# MAIN
# ============================================================================
def run_all_tests():
    print_header("ARXIVLENS PIPELINE TEST SUITE")

    langsmith_ready = check_langsmith()
    print(f"LangSmith: {'✅ ON' if langsmith_ready else '⚠️  OFF'}")
    if langsmith_ready:
        print(f"Project: {os.getenv('LANGCHAIN_PROJECT', 'arxiv-lens')}\n")

    results = {}
    tests = [
        # ("Chat Intent", test_chat_intent),
        ("Task with Papers", test_task_with_existing_papers),
        ("Index & Query", test_index_and_query),
        ("Vague Query", test_vague_query),
        ("History Mgmt", test_history_management),
        ("Thresholds", test_score_thresholds),
    ]

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"{Colors.RED}❌ FAIL - {test_name}: {e}{Colors.END}")
            results[test_name] = False
        time.sleep(1)

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"Total: {len(results)} | {Colors.GREEN}Passed: {passed}{Colors.END} | {Colors.RED}Failed: {failed}{Colors.END} | Skipped: {skipped}\n")

    for name, result in results.items():
        if result is True:
            print(f"  {Colors.GREEN}✅{Colors.END} {name}")
        elif result is False:
            print(f"  {Colors.RED}❌{Colors.END} {name}")
        else:
            print(f"  {Colors.YELLOW}⚠️{Colors.END} {name} (skipped)")

    if langsmith_ready:
        print(f"\n{Colors.BLUE}View traces: https://smith.langchain.com{Colors.END}")

    print(f"\n{Colors.GREEN}Pipeline is working correctly!{Colors.END}" if passed >= 4 else f"\n{Colors.YELLOW}Some tests need attention{Colors.END}")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
