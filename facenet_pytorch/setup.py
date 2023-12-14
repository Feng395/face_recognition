import setuptools, os

PACKAGE_NAME = 'facenet-pytorch'
VERSION = '2.5.2'
AUTHOR = 'Tim Esler'
EMAIL = 'tim.esler@gmail.com'
DESCRIPTION = 'Pretrained Pytorch face detection and recognition models'
GITHUB_URL = 'https://github.com/timesler/facenet-pytorch'

parent_dir = os.path.dirname(os.path.realpath(__file__))
import_name = os.path.basename(parent_dir)

with open('{}/README.md'.format(parent_dir), 'r') as f:
    long_description = f.read()

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=GITHUB_URL,
    packages=[
        'facenet_pytorch',
        'facenet_pytorch.models',
        'facenet_pytorch.models.utils',
        'facenet_pytorch.data',
    ],
    package_dir={'facenet_pytorch':'.'},
    package_data={'': ['*net.pt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'requests',
        'torchvision',
        'pillow',
    ],
)

'''
这段代码是用于设置Python包的安装和配置信息，以便将facenet-pytorch包发布到PyPI（Python Package Index）供其他人使用。

具体解释如下：

1. 引入必要的模块和库：
   - `setuptools`：用于构建和分发Python包的工具。
   - `os`：提供与操作系统交互的功能。

2. 定义一些常量和变量：
   - `PACKAGE_NAME`：包的名称，即'facenet-pytorch'。
   - `VERSION`：包的版本号，即'2.5.2'。
   - `AUTHOR`：包的作者，即'Tim Esler'。
   - `EMAIL`：作者的电子邮件地址，即'tim.esler@gmail.com'。
   - `DESCRIPTION`：包的简要描述，即'Pretrained Pytorch face detection and recognition models'。
   - `GITHUB_URL`：包的GitHub仓库的URL，即'https://github.com/timesler/facenet-pytorch'。
   - `parent_dir`：当前文件的父目录的路径。
   - `import_name`：父目录的名称，也就是包的名称。
   - `long_description`：从README.md文件中读取的长描述信息。

3. 使用`setuptools.setup()`函数设置包的相关信息：
   - `name`：包的名称。
   - `version`：包的版本号。
   - `author`：包的作者。
   - `author_email`：作者的电子邮件地址。
   - `description`：包的简要描述。
   - `long_description`：包的详细描述，通常是从README.md文件中读取的内容。
   - `long_description_content_type`：长描述的内容类型，这里是Markdown格式。
   - `url`：包的URL，即GitHub仓库的URL。
   - `packages`：包含的子包列表。
   - `package_dir`：包的目录结构。
   - `package_data`：包含的数据文件。
   - `classifiers`：包的分类器列表，用于指定包的属性。
   - `install_requires`：包的依赖项列表，指定安装包所需的其他Python库。

通过运行该脚本，可以生成用于安装和分发facenet-pytorch包的配置文件。
'''