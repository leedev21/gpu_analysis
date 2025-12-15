from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.install import install as _install
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
import os
import shutil
import subprocess
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_date() -> str:
    try:
        return (subprocess.check_output(
            ["git", "show", "-s", "--date=format:'%Y%m%d'", "--format=%cd"],
            cwd=BASE_DIR)
                .decode("ascii").strip()).replace("'","")
    except Exception:
        return '0'

class CustomInstall(_install):
    def run(self):

        # 执行默认的安装过程
        _install.run(self)

        # 解压 .so 文件到特定目录
        install_lib_dir = self.install_lib
        target_dir = os.path.join(install_lib_dir, 'torchtrace', 'optrace')

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 找到 .so 文件并将其移动到目标目录
        source_dir = os.path.join(os.path.dirname(__file__), 'cmake_build', 'optrace')
        for file_name in os.listdir(source_dir):
            full_name = os.path.join(source_dir, file_name)
            if os.path.isfile(full_name):
                if ".so" in full_name:
                    shutil.copy(full_name, target_dir)

        source_dir = os.path.join(os.path.dirname(__file__), 'cmake_build', 'cudahook', 'bin')
        if os.path.exists(source_dir):        # install check
            # 解压 .so 文件到特定目录
            install_lib_dir = self.install_lib
            target_dir = os.path.join(install_lib_dir, 'torchtrace', 'cudahook', 'bin')

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # 找到 .so 文件并将其移动到目标目录
            for file_name in os.listdir(source_dir):
                full_name = os.path.join(source_dir, file_name)
                if os.path.isfile(full_name):
                    shutil.copy(full_name, target_dir)


setup(
    name='torchtrace',
    version='0.1.' + get_date(),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'robotpy-cppheaderparser>=5.0.16',
        'setuptools>=74.1.2',
    ],
    package_data={
        '': ['cmake_build/cudahook/bin/*'],  
        '': ['cmake_build/optrace/*'],  
    },
    entry_points={
        'vllm.general_plugins': [
            'torchtrace = torchtrace.vllm_plugin:initialize_torchtrace',
        ],
    },
    cmdclass={'install': CustomInstall},
)


