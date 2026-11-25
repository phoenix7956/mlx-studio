import urllib.request, json
for q in ["qwen3","minimax","ace","flux","krea"]:
    found=False
    for path in [f"/api/v1/models?search={q}&PageSize=5", f"/api/v1/models?keyword={q}"]:
        try:
            url="https://modelscope.cn"+path
            req=urllib.request.Request(url,headers={"User-Agent":"curl"})
            d=json.loads(urllib.request.urlopen(req,timeout=10).read())
            arr=d.get("data") or d.get("Data") or []
            if isinstance(arr,dict): arr=arr.get("models") or arr.get("Models") or []
            names=[(m.get("path") or m.get("Path",""))+"/"+(m.get("name") or m.get("Name","")) for m in arr if isinstance(m,dict)]
            if names:
                print(q,"->",names[:5]); found=True; break
        except Exception:
            pass
    if not found: print(q,"未找到")
