from setuptools import setup


setup(
    name="kak_tools",
    version="0.1",
    description="KAK decomposition tools",
    author="People",
    author_email="emails",
    packages=["kak_tools"],
    install_requires=[
        "numpy>=2.0",
        "scipy>=1.12",
        "networkx>=3.2",
        "matplotlib>=3.8",
        "pennylane>=0.45.1,<0.46",
        "paulie>=0.2.2",
    ],
    extras_require={
        # PennyLane 0.45.1 documents this JAX pair. Newer JAX versions can
        # remove APIs used by PennyLane even when pip sees no version conflict.
        # https://docs.pennylane.ai/en/stable/introduction/interfaces/jax.html
        "test": ["pytest>=8.4,<10", "jax==0.7.1", "jaxlib==0.7.1"],
    },
    # PauLie requires Python >= 3.12; JAX 0.7.1 also provides Python 3.13 wheels.
    python_requires=">=3.12",
)
