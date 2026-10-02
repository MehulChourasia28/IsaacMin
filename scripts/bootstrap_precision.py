"""Build isolated deterministic subdivision and authoritative OBJ workers."""
from pathlib import Path
import argparse,subprocess,shutil,json
from isaacmin.io import atomic_json,sha256_file
from isaacmin.security import worker_environment

def bootstrap(root):
    root=Path(root).resolve();out=root/'.tools/native_precision';out.mkdir(parents=True,exist_ok=True)
    compiler=Path(shutil.which('g++-13')).resolve();include=root/'.tools/native/usr/include';lib=root/'.tools/native/usr/lib/aarch64-linux-gnu';reports={}
    for name in ('subdivide_double','write_native_obj'):
        source=root/'scripts/native'/f'{name}.cpp';binary=out/name;dep=out/f'{name}.d'
        command=[str(compiler),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-MMD','-MF',str(dep),str(source),'-o',str(binary)]
        if name=='subdivide_double':command.extend(['-I',str(include),'-L',str(lib),'-Wl,-rpath,'+str(lib),'-losdCPU'])
        with (out/f'{name}.compile.log').open('w') as log:
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=worker_environment(),cwd=root)
        if p.returncode:raise RuntimeError('Precision worker compiler failed: '+name)
        headers=dep.read_text().replace('\\\n',' ').split(':',1)[1].split()
        dependencies=sorted({Path(p).resolve() for p in headers if Path(p).is_file()}|{compiler,binary,source})
        listing=subprocess.check_output(['ldd',str(binary)],text=True,env=worker_environment())
        for line in listing.splitlines():
            for token in line.split():
                if token.startswith('/') and Path(token).is_file():dependencies.append(Path(token).resolve())
        reports[name]={'command':command,'files':[{'path':str(p),'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(set(dependencies))]}
    result={'status':'compiled_not_qualified','method':'pinnedinstalledOpenSubdiv3.5doublebilinear and17digitnativeOBJ','compiler':str(compiler),'workers':reports,
            'build_script_sha256':sha256_file(Path(__file__)),'qualification_required':'actualnativefixture and actualsourceglobalproof'}
    atomic_json(out/'build_manifest.json',result);return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='.');args=parser.parse_args();bootstrap(Path(args.root));print('Privateprecisionworkerscompiled')
