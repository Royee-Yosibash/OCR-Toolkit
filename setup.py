import json
from setuptools import setup, find_packages

release_data = json.load(open('release_data.json'))

setup(
    name="ocr-highlight",
    version=release_data['version'],
    description="OCR enhancment package that allows evaluation for specific use cases",
    python_requires=">=3.12",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "datasets>=4.5.0",
        "easyocr>=1.7.2",
        "numpy>=2.4.2",
        "opencv-python-headless>=4.13.0",
        "pillow>=12.1.0",
    ],
)
