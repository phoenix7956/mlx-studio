import subprocess, sys
# 远端机 上探测魔塔下载通道
checks = {
  "modelscope_pkg": "/Users/<you>/.venvs/h3m3/bin/python -c \"from modelscope import snapshot_download; print('ok')\"",
  "hf_hub_pkg": "/Users/<you>/.venvs/h3m3/bin/python -c \"import huggingface_hub; print('ok')\"",
}
for k,v in checks.items():
    try:
        out=subprocess.check_output(shlex:=None)
    except Exception:
        pass
    r=subprocess.run(["bash","-c",v],capture_output=True,text=True,timeout=30)
    print(k, "->", (r.stdout.strip() or r.stderr.strip())[:80])
# 魔塔上搜 H3 / Qwen MLX 模型
import urllib.request
for q in ["qwen3", "minimax", "ace-step", "flux"]:
    try:
        url=f"https://modelscope.cn/api/v1/models?Name={q}&PageSize=5"
        d=urllib.request.urlopen(url,timeout=10).read()
        import json
        arr=json.loads(d)
        names=[m.get("Path","?")+"/"+m.get("Name","?") for m in arr.get("Data",{}).get("Models",[])] if isinstance(arr,dict) else []
        print(q, "->", names[:5])
    except Exception as e:
        print(q, "ERR", str(e)[:80])
