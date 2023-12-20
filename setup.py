import setuptools

setuptools.setup(
    name="alr_simulation_tools",
    version="0.0.1",
    author="Moritz Hesche",
    author_email="mo.hesche@gmail.com",
    # description="",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="",
    classifiers=["Programming Language :: Python :: 3"],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "nptyping",
        "trimesh",
        "scipy",
    ],
    extras_require={"dev": ["black", "pylint", "jupyter", "ipykernel", "rospkg"]},
    include_package_data=True,
    use_scm_version=True,
)
