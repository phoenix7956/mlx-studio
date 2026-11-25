import os, sys, time, json, urllib.parse, urllib.request
BASE="http://127.0.0.1:11250"
def get(path):
    r=urllib.request.urlopen(BASE+path, timeout=2000)
    return json.loads(r.read())
def main():
    sid=get("/session?kind=video")["session"]
    print("sid=",sid,flush=True)
    zshots=[l.split("] ",1)[1].strip() for l in open("/tmp/mv_zh_shots.txt").read().split("\n") if l]
    enshots=[l.split("] ",1)[1].strip() for l in open("/tmp/mv_en_shots.txt").read().split("\n") if l]
    n=int(os.environ.get("BATCH_N","3"))
    for i in range(n):
        p=zshots[i]
        print(f"[zh {i}] {p[:40]}",flush=True)
        q=urllib.parse.urlencode({"session":sid,"prompt":p,"sec":5,"steps":12,"seed":i})
        d=get("/video?"+q)
        print("  ->",d.get("url",d),flush=True)
    for i in range(n):
        p=enshots[i]
        print(f"[en {i}] {p[:40]}",flush=True)
        q=urllib.parse.urlencode({"session":sid,"prompt":p,"sec":5,"steps":12,"seed":i+3})
        d=get("/video?"+q)
        print("  ->",d.get("url",d),flush=True)
    print(get(f"/session/close?sid={sid}"))
if __name__=="__main__": main()
