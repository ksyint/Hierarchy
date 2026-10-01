"""Released language checkpoints and their reproducible download locations."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Backbone:
    repo: str
    revision: str
    attention: str = 'sdpa'
    remote_code: bool = False
    multimodal: bool = False


BACKBONES = {
    'qwen3-1.7b': Backbone('Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
    'qwen3-4b': Backbone('Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
    'kanana-2.1b': Backbone('kakaocorp/kanana-nano-2.1b-base', '3d1ace1214dbebf52b9d2cd6f18b78857194d72b'),
    'exaone-2.4b': Backbone('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct', 'ccce25bd39c141fe053e0bc75818a8f5fe962802', 'eager', True),
    'gemma3-4b': Backbone('google/gemma-3-4b-it', '093f9f388b31de276ce2de164bdc2081324b9767', multimodal=True),
    'teacher-strong': Backbone('Qwen/Qwen3-32B', '9216db5781bf21249d130ec9da846c4624c16137'),
    'teacher-medium': Backbone('Qwen/Qwen3-14B', '40c069824f4251a91eefaf281ebe4c544efd3e18'),
    'teacher-weak': Backbone('Qwen/Qwen3-8B-AWQ', '4da05a8edb55c6046cce958586c33b61da07bb79'),
}


def resolve_backbone(name='qwen3-4b'):
    name = name or 'qwen3-4b'
    if name in BACKBONES:
        return BACKBONES[name]
    for spec in BACKBONES.values():
        if spec.repo == name:
            return spec
    return Backbone(name, 'main')


def download(name, cache_dir='.cache/huggingface', destination=None, offline=False):
    from huggingface_hub import snapshot_download
    spec = resolve_backbone(name)
    try:
        return snapshot_download(spec.repo, revision=spec.revision, cache_dir=cache_dir,
                                 local_dir=destination, local_files_only=offline,
                                 allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt',
                                                 '*.jinja', '*.py', 'tokenizer.*'])
    except OSError as error:
        folder = destination or f'checkpoints/{name}'
        raise RuntimeError(
            f'Get the complete checkpoint from https://huggingface.co/{spec.repo}/tree/{spec.revision} '
            f'into {folder}, then pass --local-model {folder}. '
            'For gated repositories accept the model terms and run hf auth login first.'
        ) from error


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--destination')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.list:
        print(json.dumps({key: asdict(value) for key, value in BACKBONES.items()}, indent=2))
        return
    print(download(args.model, args.cache_dir, args.destination, args.offline))


if __name__ == '__main__':
    main()
