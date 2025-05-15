#!/usr/bin/env python

import os

import setuptools

# --- CUDA Extension Setup ---
# This list will hold your Extension objects
ext_modules = []

# Environment variable to explicitly skip CUDA build
skip_cuda_build = os.environ.get("SKIP_CUDA_BUILD", "0") == "1"

if skip_cuda_build:
    print("---------------------------------------------------------------------")
    print("SKIP_CUDA_BUILD is set to 1. Skipping CUDA extension compilation.")
    print("---------------------------------------------------------------------")
else:
    # Attempt to build CUDA extensions if not explicitly skipped
    cuda_home = os.environ.get("CUDA_HOME")
    # You might want a more robust check for nvcc or other CUDA components
    if cuda_home:
        print(f"---------------------------------------------------------------------")
        print(f"CUDA_HOME found at {cuda_home}. Attempting to build CUDA extensions.")
        print(
            f"If you want to skip this, set the SKIP_CUDA_BUILD=1 environment variable."
        )
        print(f"---------------------------------------------------------------------")

        # ==========================================================================
        # TODO: DEFINE YOUR CUDA EXTENSIONS HERE
        # This is where you would define your setuptools.Extension objects for
        # your CUDA code.
        #
        # Example (replace with your actual extension details):
        #
        # my_cuda_extension = Extension(
        #     name="mergenetic.cuda_ops",  # How it will be imported, e.g., from mergenetic import cuda_ops
        #     sources=[
        #         "src/mergenetic/cuda_ops/ops_interface.cpp", # C++/Pybind11 wrapper
        #         "src/mergenetic/cuda_ops/kernels.cu",      # Your CUDA kernel file
        #     ],
        #     include_dirs=[
        #         os.path.join(cuda_home, "include"),
        #         # any other include directories for your C++/CUDA code
        #     ],
        #     library_dirs=[os.path.join(cuda_home, "lib64")], # Adjust lib/lib64 as per your CUDA setup
        #     libraries=["cudart"], # CUDA runtime library, add others if needed (e.g., cublas, cufft)
        #     runtime_library_dirs=[os.path.join(cuda_home, "lib64")],
        #     # Extra compile args for nvcc (via C++ compiler) or direct C++ flags
        #     extra_compile_args={
        #         'nvcc': [ # For .cu files if setuptools uses NVCC directly or via a custom build step
        #             '-O3',
        #             # Add other NVCC flags, e.g., -gencode arch=compute_XX,code=sm_XX
        #         ],
        #         'cxx': [ # For .cpp files
        #             '-O3',
        #             # Add other C++ compiler flags
        #         ]
        #     },
        #     language="c++" # Or "cuda" if your setuptools/build system handles it directly
        # )
        # ext_modules.append(my_cuda_extension)
        #
        # If you have multiple CUDA extensions, define and append each one.
        # ==========================================================================

        if (
            not ext_modules
        ):  # If no extensions were defined (e.g., placeholder not filled)
            print(
                "---------------------------------------------------------------------"
            )
            print(
                "WARNING: CUDA detected, but no CUDA extensions were defined in setup.py."
            )
            print("The package will be built without CUDA-specific compiled modules.")
            print(
                "Please fill in the 'TODO' section in setup.py if CUDA modules are expected."
            )
            print(
                "---------------------------------------------------------------------"
            )

    else:
        print("---------------------------------------------------------------------")
        print("CUDA_HOME environment variable not found.")
        print("Skipping CUDA extension compilation.")
        print(
            "If you have CUDA installed and want to build extensions, ensure CUDA_HOME is set."
        )
        print(
            "Alternatively, set SKIP_CUDA_BUILD=1 to suppress this message and build without CUDA."
        )
        print("---------------------------------------------------------------------")

if __name__ == "__main__":
    setuptools.setup(
        ext_modules=ext_modules  # Add the (possibly empty) list of extensions here
    )
