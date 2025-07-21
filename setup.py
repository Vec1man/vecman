from setuptools import setup, find_packages

setup(
    name="vecman",
    version="2.7.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.17",
        "torch>=2.0.0",
        "sentence-transformers>=4.0.0",
        "datasets>=2.20.0",
        "google-generativeai>=0.8.0",
        "tqdm>=4.65.0"
    ],
    author="Loaii abdalslam",
    author_email="loaiabdalslam@gmail.com",
    description="VECMAN - A VQ-VAE based vector database for text embeddings and retrieval",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Vec1man/vecman",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)