cd "$1/tensat"
export LD_LIBRARY_PATH=$PWD/../taso/build_gpu:/opt/conda/lib:$LD_LIBRARY_PATH
R=../NNs/candidate_models/cifar10_resnet2021/onnx/resnet_2b.taso
COMMON="-r converted.txt -t converted_multi.txt -u -s none --model_file $R --n_iter 12 --iter_multi 5 --n_sec 120 --n_nodes 500000 --node_multi 500000 --no_cycle --no_runtime_report"
echo "=== n_random jitter (20) ==="
./target/debug/tensat $COMMON --n_random 20 --random_mode jitter --export_model tmp/rs2b_jit >/dev/null 2>&1
echo "=== n_random uniform (20) ==="
./target/debug/tensat $COMMON --n_random 20 --random_mode uniform --export_model tmp/rs2b_uni >/dev/null 2>&1
echo "=== n_diverse (20) ==="
./target/debug/tensat $COMMON --n_diverse 20 --export_model tmp/rs2b_div >/dev/null 2>&1
echo "=== DONE. distinct structures per mode (op-histogram + concat/split axes) ==="
/opt/conda/bin/python3 - <<'PY'
import glob, collections
def sig(p):
    L=open(p).read().splitlines(); ops=collections.Counter(); axes=[]
    for i in range(0,len(L),4):
        op=int(L[i+1]); pr=[int(x) for x in L[i+3].split(",")]; ops[op]+=1
        if op in (12,13) and pr: axes.append((op,pr[0]))
    return (tuple(sorted(ops.items())),tuple(sorted(axes)))
for mode,pat in [("input",[ "$R".replace("../","") ]),("jitter","tmp/rs2b_jit_random*.model"),("uniform","tmp/rs2b_uni_random*.model"),("diverse","tmp/rs2b_div_diverse*.model")]:
    files=glob.glob(pat) if isinstance(pat,str) else pat
    sigs=collections.Counter(sig(f) for f in files if __import__('os').path.exists(f))
    print(f"{mode}: {len([f for f in files if __import__('os').path.exists(f)])} samples -> {len(sigs)} distinct structures")
    for s,c in sigs.items():
        print(f"    x{c}: nodes={sum(n for _,n in s[0])} ops={dict(s[0])} axes={s[1]}")
PY
