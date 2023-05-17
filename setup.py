from setuptools import setup


setup(
    # meta
    name="BirdCLEF 2023",
    version="0.1.0",
    author="William Santos",
    author_email="contact@williamsantos.me",
    description="Identify bird calls in soundscapes",
    url="https://github.com/0xc0392b/birdclef-2023",

    # publicly accessible packages
    packages=["birdclef"],

    # dependencies
    install_requires=[
        "numpy==1.24.2",
        "librosa==0.10.0.post2"
    ]
)
