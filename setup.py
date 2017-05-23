import os
from setuptools import setup, find_packages

PACKAGE = 'shambrain'
PACKAGE_ABSPATH = os.path.abspath(PACKAGE)
VERSION_FILE = PACKAGE + '/version.py'

# get content from README file
with open('README.md') as file:
    README = file.read()

setup(
    name=PACKAGE,
    # TODO: define version, but how? See duecredit setup.py
    version=__version__,
    packages=list(find_packages([PACKAGE_ABSPATH], PACKAGE)),
    scripts=[],
    install_requires=['pymvpa2', 'nipype'],
    # TODO: specify after test scripts are written
    extras_require={
        'tests': [
            'pytest',
            'vcrpy', 'contextlib2'
        ]
    },
    include_package_data=False,
    provides=[PACKAGE],
    # TODO: check out how to do this
    entry_points={
        'console_scripts': [
             'duecredit=duecredit.cmdline.main:main',
        ],
    },
    author='Oliver Contier, Yaroslav Halchenko',
    author_email='o.contier@gmail.com',
    description='fmri data simulator',
    #TODO: description
    long_description="""\
    
    INSERT DESCRIPTION HERE
    
    """

"""SHABLONA FILE CONTENT"""
# PACKAGES = find_packages()
# Get version and release info, which is all stored in shambrain/version.py
# ver_file = os.path.join('shambrain', 'version.py')
# with open(ver_file) as f:
#     exec(f.read())
#
# opts = dict(name=NAME,
#             maintainer=MAINTAINER,
#             maintainer_email=MAINTAINER_EMAIL,
#             description=DESCRIPTION,
#             long_description=LONG_DESCRIPTION,
#             url=URL,
#             download_url=DOWNLOAD_URL,
#             license=LICENSE,
#             classifiers=CLASSIFIERS,
#             author=AUTHOR,
#             author_email=AUTHOR_EMAIL,
#             platforms=PLATFORMS,
#             version=VERSION,
#             packages=PACKAGES,
#             package_data=PACKAGE_DATA,
#             install_requires=REQUIRES,
#             requires=REQUIRES)
#
#
# if __name__ == '__main__':
#     setup(**opts)
