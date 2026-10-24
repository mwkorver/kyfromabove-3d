#!/usr/bin/env python3
"""A simple static HTTP server to test the S3 compare viewer locally."""

import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

DIR = Path(__file__).resolve().parent

class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    parser = argparse.ArgumentParser(description="Serve the standalone S3 compare demo locally")
    parser.add_argument("--port", type=int, default=8790, help="Port to listen on (default: 8790)")
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), StaticHandler)
    print(f"Local S3 Demo Server running at: http://127.0.0.1:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
