import torch

checkpoints = {
    '7zhen/NO_error':   r'DJSCC/7zhen/NO_error_epoch_99.pth',
    '7zhen/MI':         r'DJSCC/7zhen/MI_epoch_99.pth',
    '14zhen/NO_error':  r'DJSCC/14zhen/NO_error_epoch_99.pth',
    'model/NO_error':   r'DJSCC/model/NO_error_epoch_99.pth',
    'NoError/NO_error': r'DJSCC/NoError/NO_error_epoch_99.pth',
}

for name, path in checkpoints.items():
    print(f'\n========== {name} ==========')
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        print(f'  Top-level keys: {list(ckpt.keys())}')
        for k, v in ckpt.items():
            if isinstance(v, dict):
                sub_keys = list(v.keys())
                modules = sorted(set(sk.split('.')[0] for sk in sub_keys))
                print(f'  [{k}] -> dict with {len(sub_keys)} keys')
                print(f'         Sub-modules: {modules}')
                print(f'         First key: {sub_keys[0]}')
                print(f'         Last key:  {sub_keys[-1]}')
            elif isinstance(v, torch.Tensor):
                print(f'  [{k}] -> Tensor {tuple(v.shape)}')
            else:
                print(f'  [{k}] -> {type(v).__name__}: {v}')
    else:
        print(f'  Type: {type(ckpt).__name__}')
