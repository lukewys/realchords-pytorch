"""Setup script for realjam package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read the long description from README if it exists
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = (
        "Lightweight real-time music accompaniment system based on ReaLchords"
    )

setup(
    name="realjam",
    version="1.0.1",
    author="Yusong Wu",
    author_email="wuyusongwys@gmail.com",
    description="Lightweight real-time music accompaniment system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukewys/realchords",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "realjam-start-server=realjam.cli:start_server",
            "realjam-download-weights=realjam.cli:download_weights",
        ],
    },
)
