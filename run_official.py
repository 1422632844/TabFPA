"""Official full-pool URL/WiDS evaluation entry; defaults are scenario-specific."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'method'))
from official_defaults import resolve,profile_id


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['wids','url'],required=True)
    p.add_argument('--models',nargs='+',default=['tabtransformer_default','torchrln_default','vime_default','stg_default','tabnet_default'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    p.add_argument('--out',type=Path)
    p.add_argument('--parameters',type=Path,help='Shared parameter JSON, or a frozen record containing parameters')
    p.add_argument('--show-defaults',action='store_true')
    a=p.parse_args()
    profiles={m:dict(profile=profile_id(a.dataset,m),parameters=resolve(a.dataset,m),attack_query_cap=200,initial_clean_queries=1) for m in a.models}
    if a.parameters:
        data=json.loads(a.parameters.read_text())
        overrides=data.get('parameters',data)
        for profile in profiles.values():
            assert set(overrides)<=set(profile['parameters']),'Unknown parameter name'
            profile['parameters'].update(overrides)
            digest=hashlib.sha256(json.dumps(profile['parameters'],sort_keys=True).encode()).hexdigest()[:12]
            profile['profile']='custom_'+digest
    if a.show_defaults:
        print(json.dumps(profiles,indent=2)); return
    from official_evaluation_query200 import Evaluation,torch,save,fga
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.dataset=='url':
        # LOCAL WRAPPER / AUDIT: reuse the existing URL official loader.
        # Its VictimAdapter consumes raw features; the generic TabNet probe
        # compares logits with probabilities and can choose double scaling.
        import run_experiments as original
        def url_components(dataset,name):
            ds=original.get_dataset(dataset)
            probe,indices,meta,scaler,cons=original.dataset_data(dataset)
            cpu,adapter,judge=original.victims(dataset,name,meta,scaler,cons,probe)
            return ds,meta,scaler,cpu,cons,adapter,judge
        fga.old.components=url_components
    for m in a.models:
        root=a.out or ROOT/'runs'/('official_'+a.dataset+'_'+profiles[m]['profile']+'_q200')
        evaluation_class=Evaluation
        if a.dataset=='url' and m.split('_',1)[1]=='madry':
            from official_evaluation_temperature import Evaluation as evaluation_class
        e=evaluation_class(a.dataset,m,root/m)
        if a.parameters:
            path=a.parameters.resolve()
            e.sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest=json.loads((e.out/'manifest.json').read_text())
            manifest['source_sha256']=e.sources
            save(e.out/'manifest.json',manifest)
        rows=[e.run(s,profiles[m]['parameters']) for s in a.seeds]
        save(root/m/'complete.json',dict(dataset=a.dataset,model=m,seeds=a.seeds,
             parameters=profiles[m]['parameters'],profile=profiles[m]['profile'],attack_query_cap=200,initial_clean_queries=1,
             results=[{k:r[k] for k in ('seed','D','N','S','RA','attack_queries_max')} for r in rows]))
        del e;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
