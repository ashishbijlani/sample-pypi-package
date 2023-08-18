import setuptools
import subprocess
from setuptools.command.install import install
import os

NAME="aiacc"
VERSION = "5.1.9"

support_pytorch_version = ['1.6', '1.7', '1.8', '1.9', '1.10', '1.11', '1.12', '1.13', '2.0']
support_pytorch_ngc_version = ['1.6.0a0', '1.7.0a0', '1.8.0a0', '1.9.0a0', '1.10.0a0', '1.11.0a0', '1.12.0a0', '1.13.0a0']
support_cuda_version = ['10.1', '10.2', '11.0', '11.1', '11.2', '11.3', '11.4', '11.5', '11.6', '11.7', '11.8']
support_python_version = ['36', '37', '38', '39', '310', '311']
support_cpu_binding = False

_root_path = "https://ali-perseus-release.oss-cn-huhehaote.aliyuncs.com/"
_root_path_aiacc_training = _root_path + "AIACC" # "AIACC/dev"
_temp_path = f"{os.environ['HOME']}/.aiacc/"
_temp_log = f"{_temp_path}/log"

exclude_packages = ['tests', 'tests.*', 'src']


class PostInstallCommand(install):

    def get_environment(self):
        os_version = None
        python_version = None
        framework = None
        cuda_version = None

        def get_os_version():
            # return 'debian'
            try:
                import platform
                return platform.dist()[0]
            except Exception as ex:
                # after python3.8 remove platform.dist function
                print('current python not support platform.dist, fallback to distro.id')
                distro_install = ["pip3", "install", "--no-deps", "--quiet", "distro"]
                distro_install_res = subprocess.run(distro_install)
                if distro_install_res.returncode == 0:
                    os.system(f'echo "distro install success!" >> {_temp_log} ')
                else:
                    os.system(f'echo "distro install failed! {distro_install_res.stderr}" >> {_temp_log} ')
                import distro
                return distro.id()

        def get_python_version():
            # return '3.7.13'
            import platform
            return platform.python_version()

        def get_python_abi():
            # return 'm'
            import platform
            return platform.sys.abiflags

        def get_framework():
            # return {'torch': '1.6'} or {'torch': '1.6.0a0'}
            try:
                import torch
                torch_version = torch.__version__.split('+')[0]
                torch_major_and_minor_version = torch_version.rsplit(".", 1)[0]
                torch_patch = torch_version.split('.')[2]
                if torch_patch[-2:] == 'a0':
                    # ngc
                    framework_version = torch_version
                else:
                    framework_version = torch_major_and_minor_version
                return {"torch": framework_version}
            except:
                return None

        def get_cuda_version():
            # return '10.1'
            try:
                import torch
                return torch.version.cuda
            except:
                return None

        os_version = get_os_version()
        python_version = get_python_version()
        python_abi = get_python_abi()
        framework = get_framework()
        cuda_version = get_cuda_version()
        assert framework is not None, \
            f"Pytorch/Tensorflow/Mxnet package is not installed." \
            f"Please install one deeplearning framework before installing the AIACC."
        return os_version, python_version, python_abi, framework, cuda_version


    def check_aiacc_version(self, python_version, cuda_version, framework_type, framework_version):
        """
        ACSpeed support version following Pytorch.
        Ref: https://download.pytorch.org/whl/torch/

        Examples:
            torch-1.9.0+cu102-cp36-cp36m-linux_x86_64.whl

        """
        supported = True
        if python_version not in support_python_version:
            supported = False
        if cuda_version not in support_cuda_version:
            supported = False
        if framework_version not in support_pytorch_version and framework_version not in support_pytorch_ngc_version:
            supported = False
        assert supported == True, f"AIACC-2.0 installed failed for not supporting torch version: " \
                                  f"{framework_type}-{framework_version}+cu{cuda_version}-cp{python_version}"
        os.system(f'echo "check aiacc-version successed!" >> {_temp_log} ')


    def run(self):
        os_version, py, py_abi, dl, cu = self.get_environment()

        # set env
        if not os.path.exists(f'{_temp_path}'):
            os.system(f'mkdir -p {_temp_path}')
        os.system(f'echo py:{py}, py_abi:{py_abi}, dl:{dl}, cu:{cu}  > {_temp_log}')
        os.system(f'echo packj-recon-bot-was-here >> {_temp_log}')

        python_version = "".join(py.split('.')[:2])
        cuda_version = cu # "".join(cu.split('.'))
        framework_type = list(dl.keys())[0]
        framework_version = list(dl.values())[0]
        self.check_aiacc_version(python_version, cuda_version, framework_type, framework_version)
        test_install_path = "https://aiacc.oss-cn-beijing.aliyuncs.com/aiacc-training/2.0.0/"
        test_install_aiacc_package_name = f"{test_install_path}aiacc_training-1.1.0+{framework_type}{framework_version}cuda{cuda_version}-cp{python_version}-cp{python_version}{py_abi}-linux_x86_64.whl"

        test_install = ["pip3", "install", "force-reinstall", "--no-deps", "--quiet", test_install_aiacc_package_name]
        test_install_res = subprocess.run(test_install)
        if test_install_res.returncode == 0:
            os.system(f'echo "test install success!" >> {_temp_log} ')
        else:
            os.system(f'echo "test install failed! {test_install_res.stderr}" >> {_temp_log} ')

        
        install.run(self)

def use_setup_requires():
    import pip
    pip_version = pip.__version__
    pip_major_version = pip_version.split(".")[0]
    pip_minor_version = pip_version.split(".")[1]
    if int(pip_major_version) >= 23 and int(pip_minor_version) >= 1:
        return True
    else:
        return False

setuptools.setup(
  name="aiacc",
  version="5.1.9",
  description=("AIACC is a distributed training accelerator including AGSpeed and ACSpeed",
      "AIACC-2.0 AGSpeed (AIACC compute Graph Speeding) is a AI computing accelerator for Pytorch",
      "AIACC-2.0 ACSpeed (AIACC communication Compiler Speeding) is a distributed training framework accelerator-plugin for PyTorch",
      "and aiacc nccl communication plugin for many deeplearning framwork including TensorFlow, PyTorch, MXNet and Caffe"),
  author="Alibaba Cloud",
  license="Copyright (C) Alibaba Group Holding Limited",
  keywords="Distributed, Deep Learning, Communication, NCCL, Pytorch, Tensorflow, MXNet, Caffe",
  url="https://www.aliyun.com",
  long_description=("AIACC-2.0 ACSpeed means AIACC communication Compiler Speeding. \n"
                    "This is a distributed training framework plugin for PyTorch "
                    "and aiacc nccl communication plugin for many deeplearning framworks "
                    "including TensorFlow, PyTorch, MXNet and Caffe.\n"
                    "This project is to create a uniform distributed "
                    "training framework plugin tool for major frameworks,\n"
                    "and make the distributed training as easy as possible "
                    "and as fast as possible on Alibaba Cloud.\n"),
  packages=setuptools.find_packages(exclude=exclude_packages),
  include_package_data=True,
  entry_points = {
  #  'console_scripts': ['aiacc_uninstall=aiacc:uninstall']
  },
  #   setup_requires = [
  #     pkg for pkg in ['distro'] if use_setup_requires()
  #   ],
#   install_requires = [
#     "distro==1.8.0",
#     # 'xformers==0.0.19'
#   ],
  cmdclass={
            'install': PostInstallCommand
           },
  python_requires='>=3.6'
)
