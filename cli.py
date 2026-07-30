"""
CLI for the library-docs RAG system.

Usage:
    python cli.py ingest requests
    python cli.py ingest requests --force
    python cli.py ask requests "write a function that does a GET with retries"
    python cli.py chat requests          # interactive loop for one library
"""

import argparse
import sys

from pipeline import ingest_library, generate_code


def main():
    parser = argparse.ArgumentParser(description="Library docs RAG (Chroma + Ollama)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Fetch and index a library's docs")
    p_ingest.add_argument("library")
    p_ingest.add_argument("--force", action="store_true", help="Re-fetch even if cached")

    p_ask = sub.add_parser("ask", help="Generate code for one request")
    p_ask.add_argument("library")
    p_ask.add_argument("request")

    p_chat = sub.add_parser("chat", help="Interactive loop for a single library")
    p_chat.add_argument("library")

    args = parser.parse_args()

    if args.command == "ingest":
        result = ingest_library(args.library, force=args.force)
        print(f"\nLibrary: {result['library']}")
        print(f"Status:  {result['status']}")
        if result["status"] == "ok":
            print(f"Pages fetched: {result['pages']}")
            print(f"Chunks stored: {result['chunks']}")
            print("Sources:")
            for s in result["sources"]:
                print(f"  - {s}")
        else:
            print("Could not find documentation for this library.")
            sys.exit(1)

    elif args.command == "ask":
        print(generate_code(args.library, args.request))

    elif args.command == "chat":
        library = args.library
        print(f"Chatting about '{library}'. Type 'exit' to quit.\n")
        while True:
            try:
                request = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if request.lower() in ("exit", "quit"):
                break
            if not request:
                continue
            print()
            print(generate_code(library, request))
            print()


if __name__ == "__main__":
    main()
