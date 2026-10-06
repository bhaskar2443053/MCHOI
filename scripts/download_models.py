"""Download the model files the pipeline needs into the models folder (MCHOI_MODELS, default <repo>/models).

  wilor/detector.pt, wilor/wilor_final.ckpt   WiLoR hand detector and reconstruction model (original authors)
  yolo11x-pose.pt                             Ultralytics YOLO11x-pose body keypoints (shoulder, elbow)
  objects/<name>/nontextured.stl              YCB object meshes (GRASPA-benchmark copies and YCB google_16k)

Every file is checked against the SHA-256 of the file used for MCHOI v1.0; existing files are verified, not
downloaded again. MANO cannot be downloaded automatically: register at https://mano.is.tue.mpg.de and place
MANO_RIGHT.pkl in <models>/mano (or set MCHOI_MANO).

Usage: python scripts/download_models.py [--only wilor yolo objects]
"""
import argparse
import hashlib
import io
import json
import tarfile
import urllib.request

from mchoi.paths import MANO_DIR, MODELS

WILOR = 'https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/'
GRASPA = 'https://raw.githubusercontent.com/hsp-iit/GRASPA-benchmark/master/data/objects/YCB/'
YCB = 'http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/'
# group, path in the models folder, source URL, member of a .tgz source, SHA-256
FILES = [
    ('wilor', 'wilor/detector.pt', WILOR + 'detector.pt', None, '5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033'),
    ('wilor', 'wilor/wilor_final.ckpt', WILOR + 'wilor_final.ckpt', None, '3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2'),
    ('yolo', 'yolo11x-pose.pt', 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt', None,
     '013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8'),
    ('objects', 'objects/mustard_bottle/nontextured.stl', GRASPA + 'mustard_bottle/nontextured.stl', None,
     '86b4580d7926cf6b0623a759f90cefb369a22f39237e7995f450073a14fded90'),
    ('objects', 'objects/cracker_box/nontextured.stl', GRASPA + 'cracker_box/nontextured.stl', None,
     '8adb7f1a774da83e4a1331e072dbf6b46855bc2422fafc05e5c70cd3b451663b'),
    ('objects', 'objects/pudding_box/nontextured.stl', YCB + '008_pudding_box_google_16k.tgz', '008_pudding_box/google_16k/nontextured.stl',
     'c6310bd9dfc3d0d052d6a6f7b104725467d4deb33cf9c874a7520f5a8af4deda'),
    ('objects', 'objects/bowl/nontextured.stl', YCB + '024_bowl_google_16k.tgz', '024_bowl/google_16k/nontextured.stl',
     'b369df0fa110ba3156358ce48efc51142e07527aa55e359462770699252f76e1'),
    ('objects', 'objects/mug/nontextured.stl', YCB + '025_mug_google_16k.tgz', '025_mug/google_16k/nontextured.stl',
     '0372292ea626695a7a1cfe9b98ebd1fac89dc2ac8542eeb839d97d7e928a1082'),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 23), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(rel, url, member, expected):
    target = MODELS / rel
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + '.part')
        print(f'downloading {rel} <- {url}', flush=True)
        if member:
            with urllib.request.urlopen(url) as response, tarfile.open(fileobj=io.BytesIO(response.read()), mode='r:gz') as tar:
                part.write_bytes(tar.extractfile(member).read())
        else:
            urllib.request.urlretrieve(url, part)
        part.replace(target)
    actual = sha256(target)
    status = 'ok' if actual == expected else f'SHA-256 MISMATCH (got {actual})'
    print(f'{status:>4}  {rel}', flush=True)
    return dict(file=rel, source=url + (f'#{member}' if member else ''), bytes=target.stat().st_size, sha256=actual,
                matches_release=actual == expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='+', choices=('wilor', 'yolo', 'objects'))
    args = parser.parse_args()
    print(f'models folder: {MODELS}')
    manifest = [fetch(rel, url, member, sha) for group, rel, url, member, sha in FILES if not args.only or group in args.only]
    (MODELS / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if not (MANO_DIR / 'MANO_RIGHT.pkl').is_file():
        print(f'\nMANO missing: register at https://mano.is.tue.mpg.de, download "Models & Code" and copy MANO_RIGHT.pkl to {MANO_DIR}')
    if not all(m['matches_release'] for m in manifest):
        raise SystemExit('Some files differ from the ones used for the release (see above).')


if __name__ == '__main__':
    main()
