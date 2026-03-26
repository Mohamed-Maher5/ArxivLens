# /mnt/hdd/projects/ArxivLens/test_complete_pipeline.py
"""
Complete End-to-End ArxivLens Test
Uses public APIs from each module's __init__
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Public APIs from each module's __init__
from app.ingestion import ArxivFetcher, ingest_paper
from app.indexing import Chunker, Embedder, VectorStore
from app.generation import run_pipeline
from app.models.schemas import Message
from app.core.logger import logger


def search_papers_interactive():
    """Step 1: User enters query, show max 5 papers."""
    print("\n" + "="*70)
    print("🔍 STEP 1: Search ArXiv Papers")
    print("="*70)
    
    query = input("\nEnter your search query (e.g., 'attention mechanism'): ").strip()
    if not query:
        print("❌ Empty query. Exiting.")
        return None
    
    print(f"\nSearching for: '{query}'...")
    
    fetcher = ArxivFetcher()
    papers = fetcher.search_papers(query, max_results=5)
    
    if not papers:
        print("❌ No papers found. Try a different query.")
        return None
    
    print(f"\n📚 Found {len(papers)} papers:\n")
    
    for i, paper in enumerate(papers, 1):
        print(f"  [{i}] {paper.title}")
        print(f"      ID: {paper.paper_id}")
        print(f"      Authors: {', '.join(paper.authors[:2])}{' et al.' if len(paper.authors) > 2 else ''}")
        print(f"      Published: {paper.published}")
        print()
    
    return papers


def select_paper(papers):
    """Step 2: User selects paper 1-5."""
    print("="*70)
    print("📋 STEP 2: Select Paper")
    print("="*70)
    
    while True:
        choice = input(f"Select paper [1-{len(papers)}] or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(papers):
                selected = papers[idx]
                print(f"\n✅ Selected: {selected.title}")
                return selected
            else:
                print(f"❌ Invalid choice. Enter 1-{len(papers)}.")
        except ValueError:
            print("❌ Invalid input. Enter a number.")


def ingest_and_index(paper):
    """Step 3: Full ingestion and indexing pipeline."""
    print("\n" + "="*70)
    print("⚙️  STEP 3: Ingest & Index Paper")
    print("="*70)
    
    try:
        # 3.1: Ingest (fetch + parse)
        print(f"\n⬇️  Fetching and parsing paper {paper.paper_id}...")
        parsed_result = ingest_paper(paper)
        print(f"   ✅ Parsed: {len(parsed_result.get('pages', []))} pages")
        
        # 3.2: Chunk
        print("\n✂️  Chunking paper...")
        chunker = Chunker()
        chunks = chunker.chunk(parsed_result)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # 3.3: Embed
        print("\n🔢 Embedding chunks...")
        embedder = Embedder()
        embedded_chunks = embedder.embed_chunks(chunks)
        print(f"   ✅ Embedded {len(embedded_chunks)} chunks")
        
        # 3.4: Store in Qdrant
        print("\n💾 Storing in Qdrant...")
        store = VectorStore(parsed_result['paper_id'])
        store.store(embedded_chunks)
        print(f"   ✅ Indexed to collection: paper_{paper.paper_id}")
        
        return parsed_result['paper_id']
        
    except Exception as e:
        logger.error(f"Ingest/index failed: {e}")
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def chat_loop(paper_id):
    """Step 4: Interactive chat with the paper."""
    print("\n" + "="*70)
    print("💬 STEP 4: Chat with Paper")
    print("="*70)
    print(f"Paper ID: {paper_id}")
    print("Type your questions or 'exit' to finish.\n")
    
    history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        try:
            # Run pipeline with history
            result = run_pipeline(user_input, history, paper_id)
            
            # Display response
            print(f"\nArxivLens: {result['answer']}")
            print(f"   [Confidence: {result['confidence']}]")
            
            if result['sources']:
                print(f"   [Sources: {len(result['sources'])} chunks]")
                for i, src in enumerate(result['sources'][:2], 1):
                    print(f"      - {src['paper_title']}, p.{src['page_number']}")
            
            # Update history
            history.append(Message(role="user", content=user_input))
            history.append(Message(role="assistant", content=result['answer']))
            
            # Keep history manageable (last 12 = 6 exchanges, will be summarized internally)
            if len(history) > 12:
                history = history[-12:]
            
            print()
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            print(f"\n❌ Error: {e}")


def main():
    """Complete pipeline orchestration."""
    print("\n" + "█"*70)
    print("█" + " "*15 + "ARXIVLENS COMPLETE PIPELINE TEST" + " "*16 + "█")
    print("█"*70)
    
    # Step 1: Search
    papers = search_papers_interactive()
    if not papers:
        sys.exit(1)
    
    # Step 2: Select
    selected = select_paper(papers)
    if not selected:
        print("Cancelled.")
        sys.exit(0)
    
    # Step 3: Ingest & Index
    paper_id = ingest_and_index(selected)
    if not paper_id:
        print("❌ Failed to process paper.")
        sys.exit(1)
    
    # Step 4: Chat
    chat_loop(paper_id)
    
    print("\n" + "█"*70)
    print("█" + " "*20 + "TEST COMPLETED" + " "*26 + "█")
    print("█"*70)


if __name__ == "__main__":
    main()