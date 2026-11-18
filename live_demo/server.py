"""Local inference demo. Run with the project virtualenv; no training or external API."""
from pathlib import Path
import sys, json, re, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import torch
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'charexp'
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT))
from eval_punct_vs_best import FrozenRestorer, TOK

class Restorer:
    def __init__(self):
        torch.set_num_threads(2)
        self.engine = FrozenRestorer(
            'punct', ROOT / 'punct_four_epochs_output_v2/punct_four_epochs_latest.pt.best.pt',
            ROOT / 'punct_tensors_audit_v2/train')
        self.lock = threading.Lock()
        self.params = sum(p.numel() for p in self.engine.model.parameters())

    def restore(self, text):
        if len(list(TOK.finditer(text))) > 256:
            raise ValueError('Please use at most 256 words per request.')
        with self.lock:
            result, _ = self.engine.predict(text)
        return result

class Handler(BaseHTTPRequestHandler):
    def send(self, status, payload, content_type='application/json; charset=utf-8'):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data))); self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == '/': self.send(200, (Path(__file__).parent/'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/health': self.send(200, {'ready': True, 'model': 'Punctuation-aware CRF schema2', 'parameters': RESTORER.params, 'checkpoint': 'punct_four_epochs_latest.pt.best.pt', 'max_words': 256})
        else: self.send(404, {'error': 'Not found'})
    def do_POST(self):
        if self.path != '/restore': return self.send(404, {'error': 'Not found'})
        try:
            n = int(self.headers.get('Content-Length', 0))
            if not 0 < n <= 32768: raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(n)); text = body.get('text')
            if not isinstance(text, str) or len(text) > 12000: raise ValueError('Invalid text')
            self.send(200, {'text': RESTORER.restore(text)})
        except (ValueError, json.JSONDecodeError) as exc: self.send(400, {'error': str(exc)})
        except Exception:
            import traceback; traceback.print_exc(); self.send(500, {'error': 'Inference failed; your input has not been changed.'})

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8790); ap.add_argument('--host', default='127.0.0.1'); args = ap.parse_args()
    RESTORER = Restorer()
    print(f'Demo ready at http://{args.host}:{args.port}', flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
