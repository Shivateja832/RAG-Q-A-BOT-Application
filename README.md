# RAG-Q-A-BOT-Application
The Q&amp;A Bot is a RAG system that ingests 4-5 documents, chunks them with overlap, embeds them in batches, and stores vectors in a persistent database. Users ask questions via CLI or web UI, the top-k relevant chunks are retrieved, and an LLM generates grounded answers with source citations from filenames and page numbers.
