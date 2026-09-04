"""RunPod 控制小工具（全量协议对齐上云用；key 从 ~/.runpod/config.toml 读，永不打印）。

    python scripts/pa_pod/pod_ctl.py list                 # 账号下 pods / 网络卷
    python scripts/pa_pod/pod_ctl.py resume <pod_id>      # podResume（on-demand，卷保留）
    python scripts/pa_pod/pod_ctl.py status <pod_id>      # desiredStatus / runtime / SSH 端口
    python scripts/pa_pod/pod_ctl.py stop <pod_id>        # podStop
    python scripts/pa_pod/pod_ctl.py confirm-stopped <pod_id> [n]   # 连续 n 次查 EXITED（死规定）
    python scripts/pa_pod/pod_ctl.py deploy <name> "<gpuTypeId>" <networkVolumeId>   # resume 无卡时新开同卷 pod

API 事实（memory runpod-pods-image-volumes）：GraphQL https://api.runpod.io/graphql，
`Authorization: Bearer <key>` + 自定义 User-Agent 才 200；REST https://rest.runpod.io/v1 亦可。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

GQL = "https://api.runpod.io/graphql"
UA = "autocls-pod-ctl/1.0"


def _key() -> str:
    cfg = Path.home() / ".runpod" / "config.toml"
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("apikey"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("~/.runpod/config.toml 无 apikey")


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(GQL, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {_key()}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read().decode())
    if out.get("errors"):
        raise SystemExit(f"GraphQL errors: {out['errors']}")
    return out["data"]


POD_FIELDS = """id name desiredStatus costPerHr gpuCount machineId
  machine { gpuDisplayName podHostId }
  runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } gpus { id gpuUtilPercent memoryUtilPercent } }
  networkVolumeId imageName containerDiskInGb volumeInGb"""


def cmd_list() -> None:
    d = gql("{ myself { pods { %s } networkVolumes { id name size dataCenterId } } }" % POD_FIELDS)
    for p in d["myself"]["pods"]:
        m = p.get("machine") or {}
        print(f"pod {p['id']:16s} {p['name']:28s} {p['desiredStatus']:8s} "
              f"{m.get('gpuDisplayName')} ${p['costPerHr']}/h vol={p.get('networkVolumeId')} img={p.get('imageName')}")
    for v in d["myself"]["networkVolumes"]:
        print(f"vol {v['id']:12s} {v['name']:28s} {v['size']} GB {v['dataCenterId']}")


def cmd_status(pod_id: str) -> dict:
    d = gql("query($id:String!){ pod(input:{podId:$id}) { %s } }" % POD_FIELDS, {"id": pod_id})
    p = d["pod"]
    rt = p.get("runtime") or {}
    ssh = [f"{x['ip']}:{x['publicPort']}" for x in (rt.get("ports") or [])
           if x.get("privatePort") == 22 and x.get("isIpPublic")]
    print(f"{p['id']} {p['name']} status={p['desiredStatus']} gpu={(p.get('machine') or {}).get('gpuDisplayName')} "
          f"${p['costPerHr']}/h uptime={rt.get('uptimeInSeconds')} ssh={ssh}")
    return p


def cmd_resume(pod_id: str) -> None:
    d = gql("mutation($id:String!){ podResume(input:{podId:$id, gpuCount:1}) { id desiredStatus costPerHr } }",
            {"id": pod_id})
    print("podResume →", d["podResume"])


def cmd_stop(pod_id: str) -> None:
    d = gql("mutation($id:String!){ podStop(input:{podId:$id}) { id desiredStatus } }", {"id": pod_id})
    print("podStop →", d["podStop"])


def cmd_confirm_stopped(pod_id: str, n: int = 3) -> None:
    ok = 0
    for i in range(n):
        p = cmd_status(pod_id)
        if p["desiredStatus"] == "EXITED" and not (p.get("runtime") or {}).get("uptimeInSeconds"):
            ok += 1
        time.sleep(5)
    print(f"confirm-stopped: {ok}/{n} 次 EXITED")
    if ok != n:
        raise SystemExit(2)


def cmd_deploy(name: str, gpu: str, volume_id: str, image: str = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404") -> None:
    """podFindAndDeployOnDemand：在网络卷所在数据中心新开一台 on-demand pod 并挂卷（resume 无卡时的备选）。"""
    pub = (Path.home() / ".ssh" / "id_ed25519.pub").read_text(encoding="utf-8").strip()
    q = """mutation($in: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $in) { id desiredStatus costPerHr imageName machine { podHostId gpuDisplayName } }
    }"""
    inp = {"cloudType": "ALL", "gpuCount": 1, "volumeInGb": 0, "containerDiskInGb": 30,
           "minVcpuCount": 2, "minMemoryInGb": 15, "gpuTypeId": gpu, "name": name, "imageName": image,
           "dockerArgs": "", "ports": "22/tcp,8888/http", "volumeMountPath": "/workspace",
           "networkVolumeId": volume_id, "env": [{"key": "PUBLIC_KEY", "value": pub}]}
    d = gql(q, {"in": inp})
    print("deploy →", d["podFindAndDeployOnDemand"])


def main() -> None:
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    cmd = a[0]
    if cmd == "list":
        cmd_list()
    elif cmd == "status":
        cmd_status(a[1])
    elif cmd == "resume":
        cmd_resume(a[1])
    elif cmd == "stop":
        cmd_stop(a[1])
    elif cmd == "confirm-stopped":
        cmd_confirm_stopped(a[1], int(a[2]) if len(a) > 2 else 3)
    elif cmd == "deploy":       # deploy <name> <gpuTypeId> <networkVolumeId>
        cmd_deploy(a[1], a[2], a[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
