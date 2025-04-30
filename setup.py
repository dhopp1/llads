import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="llads",
    version="0.0.6",
    author="Daniel Hopp",
    author_email="daniel.hopp@un.org",
    description="LLM insights to data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhopp1/llads",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
