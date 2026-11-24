"""Cache exact pretrained MLM logits and decoder inputs; model weights stay outside Git."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import numpy as np
import torch
import transformers
from transformers import BertForMaskedLM, BertTokenizer
from world import COLORS, NAMES, phrases

HERE = Path(__file__).resolve().parent
REV = '30b0a37ccaaa32f332884b96992754e246e48c5f'
CACHE = Path.home()/'.cache'/'teaching-cost'/REV


def main():
    torch.set_num_threads(1)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in ('config.json','vocab.txt','model.safetensors'):
        path = CACHE/name
        if not path.exists():
            with urlopen(f'https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/{REV}/{name}', timeout=60) as response:
                data = response.read()
            tmp = path.with_suffix('.part')
            tmp.write_bytes(data)
            os.replace(tmp,path)
    model = BertForMaskedLM.from_pretrained(str(CACHE), local_files_only=True).eval()
    tok = BertTokenizer.from_pretrained(str(CACHE), local_files_only=True)
    ids = tok.convert_tokens_to_ids(COLORS)
    assert len(set(ids))==4 and tok.unk_token_id not in ids
    archive, timings = {}, {}
    with torch.inference_mode():
        decoder = model.cls.predictions.decoder
        for key in NAMES:
            rows = []
            scores = []
            tokens = []
            start = time.process_time()
            for sentence in phrases(key):
                batch = tok(sentence,return_tensors='pt')
                idx = (batch.input_ids[0] == tok.mask_token_id).nonzero().item()
                hidden = model.bert(**batch).last_hidden_state[0,idx]
                h = model.cls.predictions.transform(hidden)
                logits = decoder.weight[ids] @ h + decoder.bias[ids]
                rows.append(h.numpy())
                scores.append(logits.numpy())
                tokens.append(batch.input_ids.shape[1])
            timings[key] = time.process_time()-start
            official = model(**batch).logits[0,idx,ids]
            assert torch.allclose(official,logits,atol=0.0001)
            archive[key+'_h'] = np.array(rows)
            archive[key+'_logits'] = np.array(scores)
            archive[key+'_tokens'] = np.array(tokens)
    dest = HERE/'results'/'encoder.npz'
    np.savez_compressed(dest,**archive)
    metadata = dict(model='google/bert_uncased_L-2_H-128_A-2',revision=REV,
                    weights_sha256=hashlib.sha256((CACHE/'model.safetensors').read_bytes()).hexdigest(),
                    torch=torch.__version__, transformers=transformers.__version__,
                    python=sys.version.split()[0], parameters=sum(p.numel() for p in model.parameters()),
                    output_ids=ids, cpu_seconds=timings, phrases={k:phrases(k) for k in NAMES})
    (dest.parent/'encoder.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    main()
