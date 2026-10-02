#!/usr/bin/env python3
"""Reproducible, resumable ARM64 user-space native builds; no sudo/system mutation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request
import zipfile
import stat

ROOT=Path(__file__).resolve().parents[1]
LOGS=ROOT/'artifacts/bootstrap';LOGS.mkdir(parents=True,exist_ok=True)
NATIVE=ROOT/'.tools/native/usr'
PREFIX=[ROOT/'.tools/oiio',ROOT/'.tools/ocio',ROOT/'.tools/minizip',ROOT/'.tools/tbb',NATIVE,ROOT/'.tools/usd']
env={k:os.environ[k] for k in ('PATH','HOME','USER','LANG') if k in os.environ}
env.update(PYTHONNOUSERSITE='1',QT_QPA_PLATFORM='offscreen',
           UV_CACHE_DIR=str(ROOT/'.tools/uv-cache'),
           LD_LIBRARY_PATH=':'.join(str(p) for p in [ROOT/'.tools/tbb/lib',ROOT/'.tools/oiio/lib',
               ROOT/'.tools/ocio/lib',ROOT/'.tools/minizip/lib',NATIVE/'lib/aarch64-linux-gnu',NATIVE/'lib',ROOT/'.tools/usd/lib']))


def run(name,argv,timeout=14400):
    print(name,flush=True)
    with (LOGS/f'{name}.log').open('w') as log:
        subprocess.run([str(x) for x in argv],cwd=ROOT,env=env,stdout=log,
                       stderr=subprocess.STDOUT,check=True,timeout=timeout)


def configure(name,extra):
    source={'usd':'OpenUSD','tbb':'oneTBB','ocio':'OpenColorIO',
            'oiio':'OpenImageIO','minizip':'minizip-ng','blender':'blender'}[name]
    run(f'{name}-configure',['cmake','-S',ROOT/'.tools/src'/source,'-B',ROOT/'.tools/build'/name,
          '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=gcc-13','-DCMAKE_CXX_COMPILER=g++-13',
          f'-DCMAKE_INSTALL_PREFIX={ROOT}/.tools/{name}',
          f'-DCMAKE_PREFIX_PATH={";".join(map(str,PREFIX))}',
          f'-DCMAKE_LIBRARY_PATH={NATIVE}/lib/aarch64-linux-gnu',
          '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON',*extra])
    run(f'{name}-build',['cmake','--build',ROOT/'.tools/build'/name,'-j','8'])
    run(f'{name}-install',['cmake','--install',ROOT/'.tools/build'/name])


def packages(lock):
    for item in lock['packages']:
        target=ROOT/'.tools/debs-locked'/Path(item['url']).name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
            part=target.with_suffix('.part')
            with urllib.request.urlopen(item['url'],timeout=60) as response,part.open('wb') as out:
                shutil.copyfileobj(response,out)
            if hashlib.sha256(part.read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError(f'Package checksum mismatch {item["name"]}')
            part.replace(target)
        run('extract-'+item['name'],['dpkg-deb','-x',target,ROOT/'.tools/native'])
    header=NATIVE/'include/jconfig.h'
    if not header.exists():header.symlink_to('aarch64-linux-gnu/jconfig.h')


def sources(lock):
    for name,item in lock['sources'].items():
        path=ROOT/'.tools/src'/name
        if not path.exists():
            run(name+'-clone',['git','clone','--no-checkout',item['url'],path])
            run(name+'-checkout',['git','-C',path,'checkout','--detach',item['commit']])
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=item['commit']:
            raise RuntimeError(f'Existing {name} revision differs; preserve it and choose a separate build prefix')
        if name=='HighMap':
            run(name+'-submodules',['git','-C',path,'submodule','update','--init','--recursive'])


def isaac_runtime(lock):
    runtime=lock['isaac_runtime'];target=ROOT/runtime['prefix']
    marker=target/'.isaacmin-extraction-complete.json'
    if (target/'VERSION').is_file() and (target/'VERSION').read_text().strip()!=runtime['version']:
        raise RuntimeError('Existing private runtime prefix has another version; preserve it and choose a new prefix')
    if marker.is_file() and json.loads(marker.read_text()).get('archive_sha256')==runtime['archive_sha256'] and (target/'VERSION').read_text().strip()==runtime['version']:
        print('Pinned Isaac runtime already extracted; capability probes remain required.',flush=True)
        return
    archive=ROOT/'.tools/downloads'/Path(runtime['url']).name
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.is_file():
        run('isaac-runtime-download',['curl','--fail','--location','--continue-at','-',
             '--output',archive,runtime['url']],timeout=3600)
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=runtime['archive_sha256']:raise RuntimeError('Isaac archive checksum mismatch')
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for info in package.infolist():
            path=target/info.filename
            if not path.resolve().is_relative_to(target.resolve()):raise RuntimeError('Unsafe archive member')
        package.extractall(target)
        for info in package.infolist():
            mode=info.external_attr>>16;path=target/info.filename
            if stat.S_ISLNK(mode):
                destination=package.read(info).decode()
                if not (path.parent/destination).resolve().is_relative_to(target.resolve()):
                    raise RuntimeError('Unsafe archive symlink')
                path.unlink();path.symlink_to(destination)
            elif path.exists() and mode:path.chmod(mode&0o777)
    if (target/'VERSION').read_text().strip()!=runtime['version']:raise RuntimeError('Isaac version mismatch')
    marker.write_text(json.dumps({'archive_sha256':runtime['archive_sha256'],'version':runtime['version'],'status':'extraction_complete'},indent=2)+'\n')
    # post_install.sh only creates examples link and installs a desktop icon.
    # Neither is required for this worker; preserve existing desktop configuration.
    print('Isolated runtime extracted; no EULA override or desktop configuration changes applied.',flush=True)


def geometry_runtime(lock):
    prefix=ROOT/'.tools/manifold-py'
    if not (prefix/'bin/python').is_file():
        run('geometry-python-env',['uv','venv',prefix,'--python','/usr/bin/python3.12'])
    wheels=[]
    for item in lock['geometry_python_packages']:
        path=ROOT/'.tools/geometry-wheels'/item['filename'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            part=path.with_suffix('.part')
            with urllib.request.urlopen(item['url'],timeout=60) as source,part.open('wb') as out:
                shutil.copyfileobj(source,out)
            if hashlib.sha256(part.read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError('Geometry wheel checksum mismatch: '+item['name'])
            part.replace(path)
        wheels.append(path)
    run('geometry-python-install',['uv','pip','install','--python',prefix/'bin/python','--no-deps',*wheels])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['packages','sources','terrain','blender','isaac','geometry','precision','all'],default='all')
    args=parser.parse_args()
    if platform.machine()!='aarch64':raise RuntimeError('This lock targets native Linux ARM64')
    if not shutil.which('g++-13'):raise RuntimeError('Pinned host compiler GCC13 unavailable; do not change the system compiler default')
    lock=json.loads((LOGS/'native-build-lock.json').read_text())
    if args.phase in ('geometry','all'):geometry_runtime(lock)
    if args.phase in ('packages','all'):packages(lock)
    if args.phase in ('sources','all'):sources(lock)
    if args.phase in ('terrain','all'):
        patch=ROOT/'scripts/native/patches/highmap-optional-mixbox-color.patch'
        check=subprocess.run(['git','-C',str(ROOT/'.tools/src/HighMap'),'apply','--reverse','--check',str(patch)],
                             env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if check.returncode:
            run('highmap-optional-mixbox-patch',['git','-C',ROOT/'.tools/src/HighMap','apply',patch])
        run('native-configure',['cmake','-S',ROOT/'scripts/native','-B',ROOT/'.tools/build/native-gcc13',
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=gcc-13','-DCMAKE_CXX_COMPILER=g++-13',
            f'-DCMAKE_PREFIX_PATH={NATIVE}',f'-DCMAKE_LIBRARY_PATH={NATIVE}/lib/aarch64-linux-gnu',
            f'-DCMAKE_INCLUDE_PATH={NATIVE}/include','-DOpenCL_LIBRARY=/usr/lib/aarch64-linux-gnu/libOpenCL.so.1',
            '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON',
            f'-DCMAKE_EXE_LINKER_FLAGS=-L{NATIVE}/lib/aarch64-linux-gnu -L{NATIVE}/lib -Wl,-rpath-link,{NATIVE}/lib'])
        run('highmap-build',['cmake','--build',ROOT/'.tools/build/native-gcc13','--target','isaacmin_highmap','-j','8'])
        run('openvdb-build',['g++-13','-std=c++17','-O2','-I',NATIVE/'include','-I',NATIVE/'include/Imath',
            ROOT/'scripts/native/openvdb_worker.cpp','-L',NATIVE/'lib/aarch64-linux-gnu',
            f'-Wl,-rpath,{NATIVE}/lib/aarch64-linux-gnu','-lopenvdb','-ltbb','-lImath-3_1',
            '-o',ROOT/'.tools/openvdb_worker'])
    if args.phase in ('blender','all'):
        patch=ROOT/'scripts/native/patches/blender-oiio-include-order.patch'
        check=subprocess.run(['git','-C',str(ROOT/'.tools/src/blender'),'apply','--reverse','--check',str(patch)],
                             env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if check.returncode:
            run('blender-local-header-order-patch',['git','-C',ROOT/'.tools/src/blender','apply',patch])
        configure('tbb',['-DTBB_TEST=OFF'])
        if not (ROOT/'.tools/blender-py/bin/python').is_file():
            run('blender-python-env',['uv','venv',ROOT/'.tools/blender-py','--python','/usr/bin/python3.12'])
        run('blender-python-deps',['uv','pip','install','--python',ROOT/'.tools/blender-py/bin/python',
                                   'numpy==1.26.4','Jinja2==3.1.6','MarkupSafe==3.0.3'])
        configure('usd',['-DPXR_BUILD_MONOLITHIC=ON','-DPXR_ENABLE_PYTHON_SUPPORT=ON',
            f'-DPython3_EXECUTABLE={ROOT}/.tools/blender-py/bin/python','-DPython3_FIND_VIRTUALENV=ONLY',
            '-DPXR_BUILD_IMAGING=ON','-DPXR_BUILD_USD_IMAGING=ON','-DPXR_BUILD_USDVIEW=OFF',
            f'-DOpenSubdiv_DIR={ROOT}/scripts/native/cmake/OpenSubdiv','-DPXR_BUILD_TESTS=OFF',
            '-DPXR_BUILD_EXAMPLES=OFF','-DPXR_BUILD_TUTORIALS=OFF','-DPXR_BUILD_USD_TOOLS=OFF',
            '-DPXR_ENABLE_GL_SUPPORT=OFF','-DPXR_BUILD_DOCUMENTATION=OFF'])
        configure('minizip',['-DMZ_BUILD_TESTS=OFF','-DMZ_BUILD_UNIT_TESTS=OFF','-DMZ_OPENSSL=OFF',
            '-DMZ_LIBBSD=OFF','-DMZ_LZMA=OFF','-DMZ_ZSTD=OFF','-DMZ_BZIP2=OFF','-DMZ_ICONV=OFF','-DBUILD_SHARED_LIBS=ON'])
        configure('ocio',['-DOCIO_BUILD_APPS=OFF','-DOCIO_BUILD_TESTS=OFF','-DOCIO_BUILD_GPU_TESTS=OFF',
            '-DOCIO_BUILD_PYTHON=OFF','-DOCIO_BUILD_DOCS=OFF','-DOCIO_INSTALL_EXT_PACKAGES=NONE',
            '-DOCIO_USE_SSE=OFF','-DOCIO_USE_AVX=OFF'])
        configure('oiio',['-DUSE_PYTHON=OFF','-DOIIO_BUILD_TESTS=OFF','-DOIIO_BUILD_TOOLS=OFF',
            '-DUSE_OPENGL=OFF','-DUSE_QT=OFF','-DUSE_FFMPEG=OFF','-DUSE_OPENVDB=OFF',
            '-DUSE_DCMTK=OFF','-DUSE_LIBRAW=OFF','-DUSE_WEBP=OFF','-DSTOP_ON_WARNING=OFF',
            '-DOpenImageIO_BUILD_MISSING_DEPS=none','-DUSE_OpenCV=OFF',
            f'-DTIFF_LIBRARY_RELEASE={NATIVE}/lib/aarch64-linux-gnu/libtiff.so',
            f'-DJPEG_LIBRARY_RELEASE={NATIVE}/lib/aarch64-linux-gnu/libjpeg.so',
            f'-DPNG_LIBRARY_RELEASE={NATIVE}/lib/aarch64-linux-gnu/libpng.so',
            f'-DCMAKE_CXX_FLAGS=-I{NATIVE}/include',f'-DCMAKE_C_FLAGS=-I{NATIVE}/include'])
        configure('blender',['-C',ROOT/'.tools/src/blender/build_files/cmake/config/blender_lite.cmake',
            '-C',ROOT/'.tools/src/blender/build_files/cmake/config/blender_headless.cmake',
            '-DPYTHON_VERSION=3.12','-DWITH_BOOST=ON','-DWITH_TBB=ON','-DWITH_USD=ON',
            '-DWITH_OPENVDB=ON','-DWITH_IMAGE_OPENEXR=ON','-DWITH_OPENCOLORIO=ON','-DWITH_IO_WAVEFRONT_OBJ=ON',
            '-DWITH_BUILDINFO=ON','-DWITH_OPENSUBDIV=ON',f'-DOPENSUBDIV_ROOT_DIR={NATIVE}','-DWITH_LIBS_PRECOMPILED=OFF','-DWITH_VULKAN_BACKEND=OFF',
            '-DWITH_GHOST_WAYLAND=OFF','-DWITH_PYTHON_INSTALL=OFF','-DWITH_PYTHON_INSTALL_NUMPY=OFF',
            '-DWITH_PYTHON_INSTALL_REQUESTS=OFF','-DUSD_PYTHON_SUPPORT=ON',
            f'-DUSD_ROOT_DIR={ROOT}/.tools/usd',f'-DTBB_DIR={ROOT}/.tools/tbb/lib/cmake/TBB',
            f'-DOPENIMAGEIO_ROOT_DIR={ROOT}/.tools/oiio',
            f'-DCMAKE_INCLUDE_PATH={NATIVE}/include',
            f'-DCMAKE_EXE_LINKER_FLAGS=-L{NATIVE}/lib/aarch64-linux-gnu -L{NATIVE}/lib -Wl,-rpath-link,{NATIVE}/lib'])
    if args.phase in ('isaac','all'):isaac_runtime(lock)
    if args.phase in ('precision','all'):
        # Uses the locked private OpenSubdiv package, existing compiler and
        # content-bound C++ sources. This does not qualify generated geometry.
        run('native-precision-build',[ROOT/'.venv/bin/python',ROOT/'scripts/bootstrap_precision.py','--root',ROOT])
    print('Build complete. Native capability probes remain required before qualification.')


if __name__=='__main__':main()
