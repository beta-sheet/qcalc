from setuptools import setup, find_packages

setup(
    # Self-descriptive entries which should always be present
    name='qcalc',
    author='Alzbeta Kubincova',
    author_email='alzbetak@ethz.ch',
    description="something",
    long_description="something longer",
    long_description_content_type="text/markdown",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
)