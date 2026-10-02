"""A resumed ZIP must still match the verified package, including every member."""
import hashlib
import zipfile
import os
from isaacmin.io import sha256_file,read_json,atomic_json


def verify_archive(path,package,manifest,preset):
    entries=manifest['files']+[dict(path='package.json',sha256=sha256_file(package/'package.json'))]
    expected={'IsaacMin-'+preset+'/'+entry['path']:entry for entry in entries}
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(expected):
            raise ValueError('Cached archive inventory differs from its verified package')
        for name,entry in expected.items():
            digest=hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=entry['sha256']:
                raise ValueError('Cached archive bytes differ from its verified package')


def prepare_archive(root,package,scene,ident):
    manifest=read_json(package/'package.json');directory=root/'artifacts/demo/downloads'
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/(ident+'_'+manifest['package_sha256'][:16]+'.zip')
    if target.is_file():verify_archive(target,package,manifest,ident)
    else:
        temporary=target.with_suffix('.partial')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for entry in manifest['files']+[dict(path='package.json',sha256=sha256_file(package/'package.json'))]:
                path=package/entry['path']
                if path.is_symlink() or not path.resolve().is_relative_to(package) or sha256_file(path)!=entry['sha256']:
                    raise ValueError('Package changed during archive creation')
                archive.write(path,arcname='IsaacMin-'+ident+'/'+entry['path'])
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,target)
        fd=os.open(directory,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    delivery=dict(archive=str(target),sha256=sha256_file(target),bytes=target.stat().st_size,
        scene_sha256=sha256_file(scene),package=str(package),qualification='unqualified_development_artifact')
    atomic_json(directory/(ident+'.json'),delivery)
    return delivery
