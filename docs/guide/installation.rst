Installation
============

CPU installation
----------------

Use Python 3.12 or newer and Rust 1.88 or newer. A source build needs the
platform linker: Xcode Command Line Tools on macOS, a C toolchain on Linux,
or Visual Studio Build Tools on Windows. No external SAT solver is required.

.. code-block:: sh

   git clone https://github.com/CarloNicolini/BSP.rs.git
   cd BSP.rs
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install .

On Windows, activate with ``.venv\Scripts\Activate.ps1`` in PowerShell.
The distribution is named ``pybsp-rust`` and the import is ``pybsp_rust``.
The default is portable, scalar ``f64`` execution. Prebuilt wheel artifacts are
available from repository CI; this project is not yet published on PyPI.

.. code-block:: python

   import pybsp_rust
   print(pybsp_rust.__version__)

Optional GPU builds
-------------------

Activate the virtual environment and install ``maturin`` and NumPy. Build just
the feature for your operating system. A default CPU wheel does not contain GPU
support merely because the machine has a GPU.

.. code-block:: sh

   python -m pip install maturin numpy

On Apple silicon, install Xcode Command Line Tools and build Metal support:

.. code-block:: sh

   xcode-select --install
   maturin develop --release --features extension-module,metal

The backend is named ``metal`` and uses custom Metal compute kernels directly.
It does not require PyTorch or the MPS framework. Full Xcode is unnecessary for
normal execution. Device pipelines compile from the embedded source at runtime.

On Linux/NVIDIA, install a compatible driver and toolkit providing ``nvcc``,
runtime headers, and ``libcudart``. The tested Ubuntu host uses
``sudo apt install nvidia-cuda-toolkit`` and CUDA 12.0:

.. code-block:: sh

   nvidia-smi
   nvcc --version
   python -m pip install patchelf
   BSP_CUDA_ARCH=61 maturin develop --release --features extension-module,cuda

``61`` targets the GTX 1060's compute capability 6.1; set the digits for your
device, such as ``86`` for 8.6. ``NVCC`` selects a compiler; ``CUDA_HOME`` adds a
nonstandard toolkit library location. Pascal GPUs need CUDA 12.9 or earlier
for compilation. The driver capability printed by ``nvidia-smi`` is not the
installed compiler version; see
`NVIDIA's architecture guidance <https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/>`_.

An explicitly selected unavailable backend fails during solver construction.
See :doc:`hardware` for precision, fallback, and concurrency behavior.

Documentation dependencies
--------------------------

From the checkout, install ``python -m pip install -e '.[docs]'``. Build HTML
with ``python -m sphinx -W --keep-going -b html docs docs/_build/html``.
The diagrams are checked-in SVGs generated from TikZ, and MathJax is bundled
locally: neither a TeX installation nor a CDN is needed to read or build HTML.
See :doc:`../building` for diagram regeneration and executable examples.
