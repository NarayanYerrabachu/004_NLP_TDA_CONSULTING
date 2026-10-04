from __future__ import annotations

import argparse

import uvicorn

from nlp_tda.config import settings
from nlp_tda.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP_TDA local consulting master-data MVP")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run ingest→embed→TDA→extract pipeline")
    run_p.add_argument("--source", type=str, default=str(settings.fixtures_dir))
    run_p.add_argument("--hash-embeddings", action="store_true", help="Skip ST model download")

    serve_p = sub.add_parser("serve", help="Start review API + UI")
    serve_p.add_argument("--host", default=settings.host)
    serve_p.add_argument("--port", type=int, default=settings.port)

    args = parser.parse_args()
    if args.cmd == "run":
        result = run_pipeline(args.source, force_hash_embeddings=True if args.hash_embeddings else None)
        print(result)
    elif args.cmd == "serve":
        uvicorn.run("nlp_tda.app:app", host=args.host, port=args.port, reload=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
