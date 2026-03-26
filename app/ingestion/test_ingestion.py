from app.ingestion.arxiv_fetcher import ArxivFetcher
from app.ingestion import ingest_paper


def main():
    query = "attention is all you need"  # you can change this

    fetcher = ArxivFetcher()

    print(f"\n🔍 Searching for papers: {query}")
    papers = fetcher.search_papers(query, max_results=1)

    if not papers:
        print("❌ No papers found")
        return

    paper = papers[0]

    print("\n📄 Selected Paper:")
    print(f"ID: {paper.paper_id}")
    print(f"Title: {paper.title}")

    print("\n⬇️ Starting ingestion pipeline...\n")

    result = ingest_paper(paper)

    print("\n✅ Ingestion completed successfully!")

    print("\n📊 Summary:")
    print(f"- Pages: {len(result.get('pages', []))}")
    print(f"- Images: {len(result.get('images', []))}")
    print(f"- Tables: {len(result.get('tables', []))}")
    print(f"- References found: {'Yes' if result.get('references') else 'No'}")

    print("\n💾 Saved to:")
    print(f"data/processed/{paper.paper_id}.json")


if __name__ == "__main__":
    main()