import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name = 'transtab',
    version = '0.0.1',
    author = 'Zifeng Wang',
    author_email = 'zifengw2@illinois.edu',
    description = 'A flexible tabular prediction model that handles variable input tables.',
    url = 'https://github.com/RyanWangZf/transtab',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['test']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)

