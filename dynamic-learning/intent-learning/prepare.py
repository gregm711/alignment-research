"""Prepare public text and sentence embeddings in the local research cache."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np

CACHE=Path.home()/".cache"/"alignment-research"/"clinc-minilm-v1"
DATA="https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/"
MODEL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/"
DIGESTS={
    "data_full.json":"36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    "domains.json":"b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a",
    "tokenizer.json":"be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "model_qint8_arm64.onnx":"4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474",
}


def fetch(base,name):
    path=CACHE/name.split("/")[-1]
    if not path.exists():
        with urlopen(base+name,timeout=45) as f:
            data=f.read()
        tmp=path.with_suffix(".download")
        tmp.write_bytes(data)
        tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=DIGESTS[path.name]:
        raise ValueError(f"Cached source differs from its pinned digest: {path.name}")
    return path


def main():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    CACHE.mkdir(parents=True,exist_ok=True)
    data=fetch(DATA,"data_full.json")
    domains=fetch(DATA,"domains.json")
    tok=fetch(MODEL,"tokenizer.json")
    model=fetch(MODEL,"onnx/model_qint8_arm64.onnx")
    tokenizer=Tokenizer.from_file(str(tok))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(pad_id=0,pad_token="[PAD]")
    options=ort.SessionOptions()
    options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=["CPUExecutionProvider"])
    inputs={x.name for x in session.get_inputs()}
    corpus=json.loads(data.read_text())
    known={c for cats in json.loads(domains.read_text()).values() for c in cats}
    text_keys=set()
    result=[]
    duplicates={}
    # Reserve official test wording first and remove normalized cross-split duplicates.
    for split in ("test","val","train"):
        duplicates[split]=0
        for text,label in corpus[split]:
            if label not in known:
                continue
            key=" ".join(text.casefold().split())
            if key in text_keys:
                duplicates[split]+=1
                continue
            text_keys.add(key)
            result.append((split,label,text))
    vectors=[]
    for start in range(0,len(result),32):
        chunk=tokenizer.encode_batch([row[2] for row in result[start:start+32]])
        ids=np.array([item.ids for item in chunk],dtype=np.int64)
        mask=np.array([item.attention_mask for item in chunk],dtype=np.int64)
        values={"input_ids":ids,"attention_mask":mask,"token_type_ids":np.zeros_like(ids)}
        output=session.run(None,{k:values[k] for k in inputs})[0]
        pooled=(output*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        vectors.append(pooled/np.linalg.norm(pooled,axis=1,keepdims=True))
    z=np.vstack(vectors).astype(np.float32)
    assert z.shape==(len(result),384) and np.isfinite(z).all()
    np.savez_compressed(CACHE/"embedded.npz",features=z,split=[r[0] for r in result],label=[r[1] for r in result])
    manifest={"source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (data,domains,tok,model)},
              "embedded_sha256":hashlib.sha256(z.tobytes()).hexdigest(),"rows":len(result),
              "omitted_duplicates":duplicates,"onnxruntime":ort.__version__}
    (CACHE/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest),flush=True)


if __name__=="__main__":
    main()
