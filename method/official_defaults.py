"""Official parameter selection only; the attack implementation is unchanged."""
import json
from pathlib import Path


def resolve(dataset, model):
    catalog = json.loads(Path(__file__).with_suffix('.json').read_text())
    training = model.split('_', 1)[1]
    profile = catalog['profiles'].get(f'{dataset}/{training}', {})
    return dict(catalog['base'], **profile.get('parameters', {}))


def profile_id(dataset, model):
    catalog = json.loads(Path(__file__).with_suffix('.json').read_text())
    training = model.split('_', 1)[1]
    return catalog['profiles'].get(f'{dataset}/{training}', {}).get('id', 'released_wreinit')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    print(json.dumps(dict(profile=profile_id(a.dataset, a.model),
                         parameters=resolve(a.dataset, a.model)), indent=2))
