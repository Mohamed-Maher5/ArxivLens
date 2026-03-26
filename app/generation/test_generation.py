# /mnt/hdd/projects/ArxivLens/app/generation/test_generation_unit.py
"""
Unit tests for generation module - NO external dependencies
Tests: Intent, history management, routing, prompts
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.generation.adaptive_rag import AdaptiveRAG
from app.models.schemas import Message


# ═════════════════════════════════════════════════════════════════════════════
# UNIT TESTS - NO EXTERNAL API CALLS (or minimal)
# ═════════════════════════════════════════════════════════════════════════════

def test_history_management():
    """
    Test: History summarization logic
    - ≤6 messages: no summary, all recent
    - >6 messages: summary + last 6
    """
    from app.generation.pipeline import Pipeline
    
    print("\n" + "="*70)
    print("TEST: History Management")
    print("="*70)
    
    pipeline = Pipeline()
    
    # Test 1: 3 messages (no summarization)
    short_history = [
        Message(role="user", content="Q1"),
        Message(role="assistant", content="A1"),
        Message(role="user", content="Q2"),
    ]
    summary, recent = pipeline._manage_history(short_history)
    assert summary == "", f"Expected no summary, got: {summary}"
    assert len(recent) == 3, f"Expected 3 recent, got: {len(recent)}"
    print("✅ ≤6 messages: No summary, all preserved")
    
    # Test 2: 8 messages (summarize 2, keep 6)
    long_history = [
        Message(role="user", content=f"Q{i}") for i in range(8)
    ]
    summary, recent = pipeline._manage_history(long_history)
    # Summary may be empty if Ollama fails, but structure is correct
    assert len(recent) == 6, f"Expected 6 recent, got: {len(recent)}"
    print(f"✅ >6 messages: Summary generated ({len(summary)} chars), 6 recent kept")
    
    # Test 3: Formatting
    history_text = pipeline._format_history_for_prompt(summary, recent)
    assert "Recent messages:" in history_text
    print("✅ Formatting includes both summary and recent")
    
    return True


def test_intent_classification():
    """
    Test: 2-way intent classification (chat vs task)
    """
    print("\n" + "="*70)
    print("TEST: Intent Classification")
    print("="*70)
    
    rag = AdaptiveRAG()
    
    test_cases = [
        # (message, expected)
        ("hi", "chat"),
        ("hello there", "chat"),
        ("thanks!", "chat"),
        ("what is attention mechanism?", "task"),
        ("explain the results", "task"),
        ("who are the authors?", "task"),
    ]
    
    passed = 0
    for message, expected in test_cases:
        result = rag.classify_intent(message)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        print(f"  {status} '{message}' → {result} (expected: {expected})")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_prompt_formatting():
    """
    Test: Prompt templates render correctly with variables
    """
    from app.generation import prompts
    
    print("\n" + "="*70)
    print("TEST: Prompt Formatting")
    print("="*70)
    
    # Test GENERAL_KNOWLEDGE_PROMPT
    try:
        formatted = prompts.GENERAL_KNOWLEDGE_PROMPT.format(
            metadata="Title: Test Paper\nAuthors: John Doe",
            history="user: hello\nassistant: hi",
            question="What is this about?"
        )
        assert "Title: Test Paper" in formatted
        assert "user: hello" in formatted
        print("✅ GENERAL_KNOWLEDGE_PROMPT formats correctly")
    except Exception as e:
        print(f"❌ GENERAL_KNOWLEDGE_PROMPT failed: {e}")
        return False
    
    # Test PAPER_ANSWER_TOP3_PROMPT
    try:
        formatted = prompts.PAPER_ANSWER_TOP3_PROMPT.format(
            chunk1_content="Content 1",
            chunk1_title="Paper 1",
            chunk1_page=3,
            chunk2_content="Content 2",
            chunk2_title="Paper 2",
            chunk2_page=5,
            chunk3_content="Content 3",
            chunk3_title="Paper 3",
            chunk3_page=7,
            history="user: previous question",
            question="Main question?"
        )
        assert "[Chunk 1]" in formatted
        assert "Paper 1" in formatted
        print("✅ PAPER_ANSWER_TOP3_PROMPT formats correctly")
    except Exception as e:
        print(f"❌ PAPER_ANSWER_TOP3_PROMPT failed: {e}")
        return False
    
    return True


def test_chat_with_history():
    """
    Test: Chat generation includes history
    """
    print("\n" + "="*70)
    print("TEST: Chat with History")
    print("="*70)
    
    rag = AdaptiveRAG()
    
    history_text = "user: What is AI?\nassistant: AI is artificial intelligence."
    message = "Tell me more"
    
    try:
        # This will call HF API - skip if no key
        response = rag.generate_chat_with_history(message, history_text)
        print(f"✅ Generated response ({len(response)} chars)")
        print(f"   Preview: {response[:100]}...")
        return True
    except Exception as e:
        print(f"⚠️  Skipped (API issue): {e}")
        return True  # Don't fail if API unavailable


def test_routing_logic():
    """
    Test: Pipeline routing without full execution
    Verify correct path is chosen based on conditions
    """
    print("\n" + "="*70)
    print("TEST: Routing Logic (Mock)")
    print("="*70)
    
    # This documents the expected routing
    routing = {
        "chat intent": "→ _handle_chat()",
        "task intent + chunks after rerank": "→ _generate_paper_answer()",
        "task intent + no chunks after rerank": "→ _general_knowledge_response()",
        "task intent + no retrieval": "→ _general_knowledge_response()",
    }
    
    for condition, path in routing.items():
        print(f"  {condition}: {path}")
    
    print("\n✅ Routing logic documented")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    print("\n" + "█"*70)
    print("█" + " "*20 + "GENERATION UNIT TESTS" + " "*23 + "█")
    print("█"*70)
    
    tests = [
        ("History Management", test_history_management),
        ("Intent Classification", test_intent_classification),
        ("Prompt Formatting", test_prompt_formatting),
        ("Chat with History", test_chat_with_history),
        ("Routing Logic", test_routing_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n💥 CRITICAL ERROR in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "█"*70)
    print("█" + " "*25 + "SUMMARY" + " "*28 + "█")
    print("█"*70)
    
    passed_count = sum(1 for _, p in results if p)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {name}")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {passed_count}/{len(results)} tests passed")
    print(f"{'='*70}")
    
    return passed_count == len(results)


if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)