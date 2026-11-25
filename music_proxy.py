"""本 机 出 歌 转 发 层 — 对 远端机 开 一 个 端 口, 收 到 出 歌 请 求 就 打 本 机 ACE (127.0.0.1:7860).
不 碰 老 板 的 ACE 进 程, 只 做 转 发 + 落 盘 file://."""
import json, os, time, base64, threading, argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.request
ACE = "http://127.0.0.1:7860"
def gen(prompt, sec=30, seed=None, steps=None):
    body={"model":"ace-step","messages":[{"role":"user","content":prompt}],"task_type":"text-to-music","audio_config":{"duration":int(sec)}}
    if seed is not None: body["seed"]=int(seed)
    if steps is not None: body["inference_steps"]=int(steps)
    r=urllib.request.Request(ACE+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    au=d["choices"][0]["message"].get("audio",[{}])[0].get("audio_url","")
    p=f"/tmp/mx_music_{int(time.time())}.wav"
    if "base64" in au:
        b64=au.split(",",1)[1]
        open(p,"wb").write(base64.b64decode(b64))
        return f"file://{p}"
    return au
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        p=urlparse(self.path).path; q=parse_qs(urlparse(self.path).query)
        def j(o,c=200):
            b=json.dumps(o).encode(); self.send_response(c); self.send_header("Content-Type","application/json"); self.end_headers(); self.wfile.write(b)
        if p=="/health": j({"ok":True,"ace":ACE})
        elif p=="/music":
            t0=time.time()
            try: url=gen(q.get("prompt",[""])[0], q.get("sec",["30"])[0]); j({"url":url,"sec":round(time.time()-t0,1)})
            except Exception as e: j({"error":f"出 歌 失 败: {e}"},502)
        else: j({"error":"unknown"},404)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--port",type=int,default=11260); a=ap.parse_args()
    print(f"[music_proxy] 本 机 出 歌 转 发 :{a.port} -> ACE {ACE}",flush=True)
    HTTPServer(("0.0.0.0",a.port),H).serve_forever()
if __name__=="__main__": main()
