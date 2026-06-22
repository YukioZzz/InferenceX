import os, time, json
from mooncake.store import MooncakeDistributedStore
store = MooncakeDistributedStore()
_MASTER = os.environ.get("MASTER","10.24.112.182:50051")
cfg = {
  "local_hostname": os.environ.get("LH","10.24.112.181"),
  "metadata_server": os.environ.get("META","http://10.24.112.182:8080/metadata"),
  # Mooncake setup(dict) reads "master_server_addr" (NOT "master_server_address");
  # pass both so it works regardless of the build's expected key.
  "master_server_addr": _MASTER,
  "master_server_address": _MASTER,
  "protocol": os.environ.get("PROTO","rdma"),
  "device_name": "",
  "rdma_devices": "",
  "global_segment_size": int(os.environ.get("SEG","1073741824")),
  "local_buffer_size": int(os.environ.get("BUF","1073741824")),
}
print("[xnode] setup cfg:", json.dumps(cfg))
try:
    rc = store.setup(cfg) if False else store.setup(cfg)
except Exception as e:
    print("[xnode] setup raised:", e)
print("[xnode] setup returned")
key = "xnode_%d" % int(time.time())
val = b"hello-cross-node-" * 64
try:
    pr = store.put(key, val)
    print("[xnode] put rc:", pr)
    g = store.get(key)
    ok = (g == val)
    print("[xnode] get len:", (len(g) if g is not None else None), "match:", ok)
    print("XNODE_OK" if ok else "XNODE_GETMISMATCH")
except Exception as e:
    print("[xnode] put/get exc:", repr(e))
    print("XNODE_FAIL")
