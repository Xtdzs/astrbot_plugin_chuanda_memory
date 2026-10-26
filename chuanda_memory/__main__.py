import argparse
import json

from .store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="串大独立记忆服务与离线管理")
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--data-dir", required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--delay", type=float, default=300)
    server.add_argument("--batch-size", type=int, default=20)
    server.add_argument("--min-batch", type=int, default=3)
    server.add_argument("--max-wait", type=float, default=1800)
    backup = sub.add_parser("backup")
    backup.add_argument("--data-dir", required=True)
    export = sub.add_parser("export", help="Rebuild local memory.md and memory.json from SQLite")
    export.add_argument("--data-dir", required=True)
    profile = sub.add_parser("profile")
    for name in ("data-dir", "namespace", "scope", "person-id"):
        profile.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        from .server import serve
        serve(args.data_dir, args.host, args.port, args.delay, args.batch_size, args.min_batch, args.max_wait)
    elif args.command == "backup":
        print(MemoryStore(args.data_dir).backup())
    elif args.command == "export":
        for path in MemoryStore(args.data_dir).export_readable():
            print(path)
    elif args.command == "profile":
        print(json.dumps(MemoryStore(args.data_dir).profile(args.namespace, args.scope, args.person_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
