import os, sys, time, json, urllib.parse, urllib.request, urllib.error
# 远端机 的 内核 现在 跑 在 11251 (11250 那 条 老 plist 已经 不用 了) —— 
# 端口 写 死 11250 的 时候, 跑批 直接 "Connection refused", 一 片 都 出 不 来.
BASE = os.environ.get("MLX_BASE") or "http://127.0.0.1:11251"
DONE=os.path.expanduser("~/.ours/mlx_server/done.json")
def load_done():
    try:
        d=json.load(open(DONE))
        return d if isinstance(d,dict) else {}
    except Exception: return {}
def save_done(d):
    json.dump(d,open(DONE,"w"))
def get(path, timeout=7200, retries=4):
    for a in range(retries):
        try:
            return json.loads(urllib.request.urlopen(BASE+path, timeout=timeout).read())
        except urllib.error.HTTPError as e:
            if e.code==409:
                print("  409, 等 30s 重试",file=sys.stderr,flush=True); time.sleep(30); continue
            raise
        except Exception as e:
            print("  重试:",e,file=sys.stderr,flush=True); time.sleep(10)
    return {"error":"超时/失败"}
def read_params():
    import os
    p=os.path.expanduser("~/.ours/mlx_server/batch_params.json")
    d={"shots":["/tmp/mv_zh_shots.txt","/tmp/mv_en_shots.txt"],"sec":5,"steps":12,"resume":True}
    if os.path.exists(p):
        try:
            import json as _j
            d.update(_j.load(open(p)))
            print("[run_full] 读 到 batch_params:",d,flush=True)
        except Exception as e:
            print("[run_full] batch_params 解 析 失 败, 用 默 认:",e,flush=True)
    return d
def parse_shots(path):
    """镜头 脚 本 行 格式: [N] prompt. 无 前 缀 的 裸 prompt 也 收."""
    out=[]
    for l in open(path).read().split("\n"):
        l=l.strip()
        if not l: continue
        if "] " in l and l[0]=="[":
            out.append(l.split("] ",1)[1].strip())
        else:
            out.append(l)
    return out
def main():
    P=read_params()
    shots=P.get("shots",["/tmp/mv_zh_shots.txt","/tmp/mv_en_shots.txt"])
    # 每 个 脚 本 自 成 一 组, 组 名 取 文 件 名 干 (去 掉 _shots.txt 尾 缀)
    groups=[]
    for sp in shots:
        import os
        gname=os.path.splitext(os.path.basename(sp))[0].replace("_shots","") or "g"
        ps=parse_shots(sp)
        groups.append((gname,ps))
    sec=float(P.get("sec",5)); steps=int(P.get("steps",12)); resume=bool(P.get("resume",True))
    jobs=[]
    for gname,ps in groups:
        for i,p in enumerate(ps):
            jobs.append((gname,p,i))
    done=load_done()
    if resume:
        jobs=[j for j in jobs if f"{j[0]}-{j[2]:02d}" not in done]
    else:
        done={}
    if not jobs:
        print("全量已完成, 无剩余",flush=True); return
    print(f"全量 {len(jobs)+len(done)} 片, 已完成 {len(done)}, 剩余 {len(jobs)}",flush=True)
    sid=get("/session?kind=video")["session"]; print("sid=",sid,flush=True)
    t0=time.time()
    for idx,(lang,prompt,pi) in enumerate(jobs):
        q=urllib.parse.urlencode({"session":sid,"prompt":prompt,"sec":sec,"steps":steps,"seed":pi%50})
        d=get("/video?"+q, retries=6)
        if "url" in d:
            print(f"[{lang} {pi:02d}] -> {d['url']}  ({idx+1}/{len(jobs)}, {int((time.time()-t0)//60)}min)",flush=True)
            done[f"{lang}-{pi:02d}"]={"url":d["url"],"ts":time.time()}
            save_done(done)
        else:
            print(f"[{lang} {pi:02d}] 失败 {d} — 保留 sid, 不 close, 退出 (重启续跑)",flush=True)
            break
    else:
        print(get(f"/session/close?sid={sid}"),flush=True)
        print("全量完成",int((time.time()-t0)//60),"min",flush=True)
if __name__=="__main__": main()
