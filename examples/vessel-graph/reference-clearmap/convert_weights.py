# SPDX-License-Identifier: MIT
# Original work for this crate.
"""Convert the reference's vessel-filling network weights (a PyTorch state
dict, `cnn_binary_vessel_filling.pth`) to safetensors, which
`vessel-graph segment --deep-fill-weights` reads.

The weights are not part of this repository; see the README ("Deep vessel
filling") for where they come from and why they are loaded at run time.

    python convert_weights.py IN.pth OUT.safetensors

Needs torch (the `clearmap-gt2` env). The output holds every tensor of the
state dict as little-endian float32 under its state-dict name, and the script
checks that it reads back equal to the input.
"""
import argparse
import json
import struct
import sys

import numpy as np
import torch


def write_safetensors(tensors, path):
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        data = np.ascontiguousarray(tensors[name], dtype='<f4')
        raw = data.tobytes()
        header[name] = {'dtype': 'F32', 'shape': list(data.shape),
                        'data_offsets': [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    text = json.dumps(header, separators=(',', ':')).encode()
    text += b' ' * (-len(text) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(text)))
        f.write(text)
        for raw in blobs:
            f.write(raw)


def read_safetensors(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        body = f.read()
    out = {}
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        assert entry['dtype'] == 'F32', (name, entry['dtype'])
        lo, hi = entry['data_offsets']
        out[name] = np.frombuffer(body[lo:hi], dtype='<f4').reshape(entry['shape'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('pth')
    parser.add_argument('out')
    args = parser.parse_args()
    state = torch.load(args.pth, map_location='cpu', weights_only=True)
    tensors = {k: v.detach().float().numpy() for k, v in state.items()}
    write_safetensors(tensors, args.out)
    back = read_safetensors(args.out)
    assert sorted(back) == sorted(tensors)
    for name in tensors:
        assert np.array_equal(back[name], tensors[name]), name
    total = sum(v.size for v in tensors.values())
    for name in sorted(tensors):
        print(f'{name:32s} {tuple(tensors[name].shape)}')
    print(f'{len(tensors)} tensors, {total} parameters -> {args.out}')


if __name__ == '__main__':
    sys.exit(main())
