from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_desc = fh.read()

dist = setup(
    name="ode_nn",
    descrption="ODE models of C19",
    long_description=long_desc,
    url="https://github.com/EbTech/AutoODE-DSL",
    packags=["ode_nn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
