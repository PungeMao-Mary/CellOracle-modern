# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from os import path
import re

with open("requirements.txt") as f:
    required = [line for line in f.read().splitlines()
                if line.strip() and not line.strip().startswith("#")]

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

root_dir = path.abspath(path.dirname(__file__))
package_name = "celloracle"

with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    license = re.search(r'__license__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author = re.search(r'__author__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author_email = re.search(r'__author_email__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    url = re.search(r'__url__\s*=\s*[\'\"](.+?)[\'"]', init_text).group(1)


with open(path.join(root_dir, package_name, 'version.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'"]', init_text).group(1)


# Optional dependency groups.  MLX is only meaningful on Apple silicon, so it is
# requested conditionally to keep `pip install celloracle[gpu]` a no-op on
# Intel macs, Linux and Windows.
extras_require = {
    # ATAC-seq / motif analysis (gimmemotifs + genomepy stack)
    "motif": [
        "genomepy>=0.14",
        "gimmemotifs>=0.17",
        "pybedtools",
    ],
    # Interactive helpers used inside Jupyter notebooks
    "notebook": [
        "ipython",
        "ipywidgets",
    ],
    # Apple Metal GPU acceleration through MLX
    "gpu": [
        "mlx>=0.20; platform_system == 'Darwin' and platform_machine == 'arm64'",
    ],
    "test": [
        "pytest",
        "build",
    ],
}
extras_require["all"] = sorted({pkg for group in extras_require.values() for pkg in group})
extras_require["dev"] = extras_require["all"] + extras_require["test"]


# Modified for CellOracle-modern: package metadata uses the full restricted
# license designation declared in celloracle.__init__.
# Start install process
setup(
    name=package_name,
    version=version,
    description='in silico gene perturbation analysis and GRN analysis with single cell data',
    long_description=readme,
    long_description_content_type="text/x-rst",
    keywords='scRNA-seq, GRN, simulation, gene perturbation',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    install_requires=required,
    extras_require=extras_require,
    author=author,
    author_email=author_email,
    url=url,
    license=license,
    zip_safe=False,
    include_package_data=True,
    package_data={"celloracle": ["go_analysis/data/*.txt", "go_analysis/data/*.obo",
                                 "data_conversion/*.R",
                                 "motif_analysis/tss_ref_data/*.bed",
                                 "network_analysis/rscripts_for_network_analysis/*.R",
                                 "utility/requirements.txt"]},
    packages=find_packages(),
    entry_points={'console_scripts': [
        'seuratToAnndata = celloracle.data_conversion.process_seurat_object:main']}
)
