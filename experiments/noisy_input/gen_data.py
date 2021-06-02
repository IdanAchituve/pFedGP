import argparse
import random

import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from imagecorruptions import corrupt

from utils import set_seed, set_logger
from experiments.heterogeneous_class_dist.dataset import gen_random_loaders

SEVERITIES = [3, 4, 5]
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]


def corrupt_loader(loader, client_id, split, corruption, severity, out_dict, tqdm_iterator=None):
    # NOTE: assume batch size is 1
    out_dict[client_id]['corruption_params'] = {'severity': severity, 'corruption_name': corruption}
    for x, y in loader:
        if tqdm_iterator:
            tqdm_iterator.set_description(
                desc=f'client {client_id} corruption {corruption} (severity {severity}), {split} data'
            )
        out_dict[client_id][split]['label'].append(y.numpy())
        out_dict[client_id][split]['original_data'].append((x * 255).numpy().astype(np.uint8))
        out_dict[client_id][split]['data'].append(
            corrupt(
                (x.squeeze(0) * 255).permute(1, 2, 0).numpy().astype(np.uint8),
                severity=severity,
                corruption_name=corruption
            ).transpose((2, 0, 1)).reshape((1, 3, 32, 32))
        )

    # concat and to numpy array
    out_dict[client_id][split]['label'] = np.concatenate(out_dict[client_id][split]['label'])
    out_dict[client_id][split]['data'] = np.concatenate(out_dict[client_id][split]['data'])
    out_dict[client_id][split]['original_data'] = np.concatenate(out_dict[client_id][split]['original_data'])

    return out_dict


def generate_data(dataset_dictionary, args):
    train_loaders, val_loaders, test_loaders = gen_random_loaders(
        data_name=args.data_name,
        data_path=args.data_path,
        num_users=args.num_users,
        bz=1,
        classes_per_user=args.classes_per_user,
        normalize=False
    )

    # iterate over train/val/test
    iterator = tqdm(enumerate(zip(train_loaders, val_loaders, test_loaders)))
    for i, (train, val, test) in iterator:
        # sample severity and corruption
        severity, corruption = random.sample(SEVERITIES, 1)[0], random.sample(CORRUPTIONS, 1)[0]

        dataset_dictionary = corrupt_loader(
            train, client_id=i, split='train', corruption=corruption, severity=severity,
            out_dict=dataset_dictionary, tqdm_iterator=iterator
        )
        dataset_dictionary = corrupt_loader(
            val, client_id=i, split='val', corruption=corruption, severity=severity,
            out_dict=dataset_dictionary, tqdm_iterator=iterator
        )
        dataset_dictionary = corrupt_loader(
            test, client_id=i, split='test', corruption=corruption, severity=severity,
            out_dict=dataset_dictionary, tqdm_iterator=iterator
        )

    return dataset_dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate noisy data")
    parser.add_argument('--data-path', type=str, default="../datafolder")
    parser.add_argument('--num-users', type=int, default=100)
    # parser.add_argument('--max-severity', type=int, default=2)
    parser.add_argument('--classes-per-user', type=int, default=10)
    parser.add_argument(
        "--out-path", type=str,
        default="../datafolder",
        help="dir path for output file"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    args = parser.parse_args()
    set_logger()
    set_seed(args.seed)

    save_path = Path(args.out_path)
    save_path.mkdir(parents=True, exist_ok=True)

    dataset_dictionary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dataset_dictionary = generate_data(dataset_dictionary, args)

    out_dict = dict()
    for k, v in dataset_dictionary.items():
        out_dict[k] = {
            'corruption_params': v['corruption_params'],
            'train': dict(v['train']),
            'val': dict(v['val']),
            'test': dict(v['test'])
        }
    with open(save_path / "data_dictionary.pkl", "wb") as f:
        pickle.dump(out_dict, f)
