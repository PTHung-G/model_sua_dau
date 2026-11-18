"""Portable CPU-only local demo. No training dependencies or remote inference."""
import json,sys,threading,webbrowser,os
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from inference import FrozenRestorer,load_gap_mapping
import sylmodel_punct as sm
import build_punct_tensors_v1 as builder

class PortableRestorer(FrozenRestorer):
    def __init__(self):
        self.kind='punct'
        ck=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=True)
        assert ck['pipeline_schema']==2
        state=ck['model'];cfg=ck['config']
        self.vocab=json.loads((ROOT/'vocab.json').read_text(encoding='utf8'))
        bases=json.loads((ROOT/'bases.json').read_text(encoding='utf8'))
        self.bid={b:i+1 for i,b in enumerate(bases)}
        self.c2i=json.loads((ROOT/'chars.json').read_text(encoding='utf8'));self.unk_char=1
        self.gap_map=load_gap_mapping(ROOT/'punct.json')
        self.sm=sm;self.builder=builder
        self.model=sm.PunctSylCRF(state['emb.weight'].shape[0],state['base_ch'],state['cand_ch'],state['cand_ids'],state['gap_emb.weight'].shape[0],**{k:cfg[k] for k in ('d','hidden','r','layers')}).eval()
        self.model.load_state_dict(state,strict=True)
        self.lock=threading.Lock()
    def restore(self,text):
        if len(list(builder.TOK.finditer(text)))>256:raise ValueError('Please use at most 256 words.')
        with self.lock:return self.predict(text)[0]

class Handler(BaseHTTPRequestHandler):
    def send(self,status,payload,content_type='application/json; charset=utf-8'):
        data=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path=='/':self.send(200,(ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        elif self.path=='/health':self.send(200,{'ready':True,'model':'Punctuation-aware CRF schema2','local':True})
        else:self.send(404,{'error':'Not found'})
    def do_POST(self):
        if self.path!='/restore':return self.send(404,{'error':'Not found'})
        # Do not permit cross-origin websites to send restoration requests.
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://127.0.0.1:{self.server.server_port}':return self.send(403,{'error':'Invalid origin'})
        try:
            n=int(self.headers.get('Content-Length',0))
            if not 0<n<=32768:raise ValueError('Invalid request size')
            text=json.loads(self.rfile.read(n)).get('text')
            if not isinstance(text,str) or len(text)>12000:raise ValueError('Invalid text')
            self.send(200,{'text':RESTORER.restore(text)})
        except (ValueError,TypeError,AttributeError) as e:self.send(400,{'error':str(e)})
        except Exception:
            import traceback;traceback.print_exc();self.send(500,{'error':'Inference failed; input unchanged.'})

if __name__=='__main__':
    torch.set_num_threads(min(2,os.cpu_count() or 1))
    print('Loading model (please wait)...',flush=True)
    RESTORER=PortableRestorer()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'Ready: {url}\nKeep this window open. Ctrl+C to stop.',flush=True)
    if '--no-browser' not in sys.argv:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
