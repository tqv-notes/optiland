.. _configurable_backend:

Configurable Backend
====================

Optiland’s core computations are routed through a unified backend abstraction layer, allowing seamless use of either NumPy or PyTorch. All numerical operations, including array creation, linear algebra, trigonometric functions, and more, are automatically dispatched to the selected backend without changing your code. This design ensures:

- Consistent API across backends
- Easy GPU acceleration via PyTorch
- Full compatibility with ML/DL workflows
- Simplified maintenance and extensibility

Importing and Selecting a Backend
----------------------------------

Begin by importing the backend module instead of `numpy` or `torch` directly:

.. code:: python

   import optiland.backend as be

By default, Optiland uses the NumPy backend. To switch to PyTorch:

.. code:: python

   # for torch with CUDA support, it must be installed manually, e.g.:
   #   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   be.set_backend("torch")

To explicitly revert to NumPy at any time:

.. code:: python

   be.set_backend("numpy")

It is generally recommended to restart the Python kernel after switching backends to avoid unexpected behavior.

NumPy Backend
-------------

The NumPy backend provides pure-CPU execution and leverages Numba JIT compilation for hotspots. Common functions (e.g., `sum`, `cos`, `sin`, `dot`, etc.) are automatically proxied from NumPy into `optiland.backend`:

.. code:: python

   # under the hood, be.sum == numpy.sum, etc.
   x = be.arange(0.0, 1.0, 0.1)    # array creation
   y = be.sin(x)                   # trigonometric
   total = be.sum(y)               # reduction

Because of automatic patching, you do not need to re‑implement any basic routines in the `NumpyBackend` class.

Threads of the compiled trace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In its domain, the NumPy backend traces real rays with a compiled kernel
(see :ref:`performance_internals`). The kernel uses one thread by default.
To trace large bundles on several threads:

.. code:: python

   be.set_trace_threads(8)    # or -1 for the number of logical CPUs
   be.get_trace_threads()     # 8

The results are the same, bit for bit, as with one thread. The limits:

- Only bundles of 8192 rays or more use several threads. For a smaller
  bundle, the threads cost about as much time as they save.
- The setting has no effect outside the compiled trace: the generic trace,
  the traces in parts, the ray generation and the PyTorch backend use one
  thread.
- For a medium that absorbs (k > 0), the step that applies the absorption
  to the intensity uses at most 4 threads.
- ``-1`` sets the number of logical CPUs: ``os.process_cpu_count()`` on
  Python 3.13 and later, and ``os.cpu_count()`` before. ``os.cpu_count()``
  counts all CPUs of the system, also those that the CPU affinity mask of
  the process excludes. The number is fixed when ``set_trace_threads`` is
  called.
- The threads are a private pool of Python threads. The setting does not
  change, and does not depend on, ``numba.set_num_threads``,
  ``NUMBA_NUM_THREADS``, ``torch.set_num_threads`` or BLAS variables such as
  ``OMP_NUM_THREADS``.
- If the program already traces in parallel processes, for example an
  optimizer with several workers, keep one thread in each process.
- ``set_trace_threads`` changes global state and is not thread-safe, like
  ``set_backend``. Call it before other threads start to trace.

PyTorch Backend
---------------

The PyTorch backend routes operations through `torch.*` and adds configurable device, precision, and gradient settings:

- **Device management**  
  Select CPU or CUDA:

  .. code:: python

     be.set_device("cuda")   # or "cpu"

- **Precision control**  
  Choose between single- and double‑precision floats:

  .. code:: python

     be.set_precision("float32")   # or "float64"

- **Global gradient mode**  
  Enable or disable autograd across all PyTorch operations:

  .. code:: python

     be.grad_mode.enable()    # all ops will track gradients
     be.grad_mode.disable()   # disable gradient tracking

Example usage:

.. code:: python

   import optiland.backend as be

   be.set_backend("torch")
   be.set_device("cuda")
   be.set_precision("float64")
   be.grad_mode.enable()

   # now every be.* call (e.g. be.matmul, be.exp) uses torch.cuda.FloatTensor,
   # with gradient support enabled.

.. tip::
   On VRAM-limited GPUs (e.g. 4-8 GB laptop/consumer cards), throughput is
   **not monotonic** with ray count. As allocations approach the physical
   VRAM limit, the CUDA driver starts spilling/evicting memory, and trace
   speed can collapse by 5-10x well before you get an actual out-of-memory
   error -- which makes the underlying cause easy to misdiagnose. If a
   ``trace()`` call is much slower than expected, check
   ``torch.cuda.max_memory_allocated()`` first: if it is close to your
   card's total VRAM, reduce ``num_rays`` (or split the trace into smaller
   batches) or switch to ``be.set_precision("float32")`` rather than
   assuming there is a bug. When only the returned rays are needed,
   ``trace(..., record=False)`` skips the per-surface ray snapshots and
   roughly triples the ray count that fits on the card.

   .. figure:: ../images/gpu_trace_before_after.png
      :width: 85%

      Throughput vs ray count on a 12 GB card (Cooke triplet, float64,
      single wavelength). Traces stay fast until allocations reach physical
      VRAM; skipping per-surface snapshots moves that point out by roughly
      3x.

JAX Backend
-----------

.. note::
   The JAX backend is experimental. It implements the whole
   ``AbstractBackend`` contract and traces rays through sequential systems,
   but parts of Optiland still assign into arrays in place, which JAX does
   not allow, or have a branch for PyTorch only. Such code raises on JAX.
   Non-sequential tracing is not supported and raises
   ``BackendCapabilityError``.

The JAX backend routes operations through ``jax.numpy``. Install JAX with
``pip install optiland[jax]`` and select the backend:

.. code:: python

   import optiland.backend as be

   be.set_backend("jax")

It differs from the PyTorch backend in these points:

- **Precision.** The default is ``float64``, as on the NumPy backend. JAX
  computes in 32 bits unless its ``jax_enable_x64`` flag is set, so
  ``be.set_backend("jax")`` sets it. The flag is global to the process: other
  JAX code in the same program also computes in 64 bits from then on.
  ``be.set_precision("float32")`` makes the backend create ``float32``
  arrays; as on the NumPy backend, Python scalars and ``be.arange`` still
  give ``float64``.
- **Immutable arrays.** ``x[mask] = value`` raises on a JAX array. Write
  ``x = be.where(mask, value, x)``, which works on all backends. For the
  same reason the backend has no ``be.copy_to``.
- **Gradients.** JAX has no ``requires_grad`` tensors and no ``backward()``.
  Write the computation as a function of the parameters and transform it:

  .. code:: python

     import jax
     from optiland.samples.objectives import CookeTriplet

     lens = CookeTriplet()

     def rms_spot(params):
         radius, thickness = params
         lens.updater.set_radius(radius, 1)
         lens.updater.set_thickness(thickness, 2)
         rays = lens.trace(Hx=0, Hy=0.7, wavelength=0.55, num_rays=16,
                           distribution="hexapolar")
         cx, cy = be.mean(rays.x), be.mean(rays.y)
         return be.sqrt(be.mean((rays.x - cx) ** 2 + (rays.y - cy) ** 2))

     params = be.array([22.01359, 6.07])
     gradient = jax.grad(rms_spot)(params)
     rms_spot(params)  # put concrete values back into the lens

  The optic is changed in place, so after ``jax.grad`` returns, every
  parameter that the function set still holds a JAX tracer, which is no
  longer valid. Using the optic then raises
  ``jax.errors.UnexpectedTracerError``. Set those parameters again with
  concrete values, as the last line does, before using the optic outside
  the transformed function.

  ``be.grad_mode`` exists with the same methods as on PyTorch, so that
  shared code can ask ``be.grad_mode.requires_grad``; JAX itself ignores it.
  The optimizers in ``optiland.optimization.optimizer.torch`` need PyTorch.
- **No ``jax.jit`` around an optic.** Building and tracing an ``Optic``
  branches in Python on parameter values (for example, whether a radius is
  infinite) and converts values to NumPy for caches. ``jax.grad`` allows
  both; ``jax.jit`` and ``jax.vmap`` do not, and raise.
- **Random numbers.** ``be.default_rng(seed)`` returns a small stateful
  generator that holds a JAX key and splits it on every draw, so the
  functions ``be.random_uniform`` and ``be.random_normal`` work as on the
  other backends.
- **Device.** JAX places arrays on its default device.
  ``be.get_device()`` returns its platform, and ``be.set_device("gpu")``
  selects the first GPU of a JAX installation that has one.

Adding New Functionality
------------------------

When extending Optiland, always import operations from `optiland.backend` to ensure compatibility across both backends:

.. code:: python

   from optiland.backend import array, dot, sqrt

   def my_custom_metric(x, y):
       return dot(x, y) / sqrt(dot(x, x) * dot(y, y))

If you define a function that relies on a backend-specific feature, add it to the `NumpyBackend` and `TorchBackend` classes, as well as the `AbstractBackend` interface, following the existing patterns.

Backend Implementation Details
------------------------------

- **Dynamic Dispatch**: The backend uses a `__getattr__` function to dynamically dispatch calls to the active backend class instance.
- **Abstract Backend**: The `backend.base.AbstractBackend` class defines the contract and common interfaces for all backends.
- **NumPy Backend**: The `backend.numpy_backend.NumpyBackend` class defines a `_lib` attribute that points to the `numpy` library, allowing for a fallback to the NumPy namespace for functions not explicitly defined.
- **PyTorch Backend**: The `backend.torch_backend.TorchBackend` class explicitly defines or maps all supported functions to PyTorch operations.
- **`to_numpy` Utility**: The `optiland.backend.utils` module provides a `to_numpy` function to convert backend-specific arrays to NumPy arrays.

Best Practices
--------------

- **Use `be.*` everywhere**. Never import `np` or `torch` directly in Optiland modules - you’ll break backend neutrality. There are exceptions, but they are rare.
- **Test on both backends**. Our CI includes pytest fixtures that run the full test suite under both NumPy and PyTorch modes. If you add a new feature, follow existing testing patterns to ensure it works on both backends.
- **Document backend-specific behavior**. If a function has different characteristics, note it in the docstring.
- **Use `to_numpy` for conversions**: When you need to convert a backend array to a NumPy array (e.g., for plotting), use the `to_numpy` function from `optiland.backend.utils`.

Troubleshooting
---------------

- **“Module ‘torch’ not found”**  
  Ensure you have installed a PyTorch build with appropriate CUDA support (if you wish to use GPU).
- **Precision mismatches**  
  Verify `be.get_precision()` matches your expectations before heavy computations. PyTorch is strict about tensor types and will raise errors if you mix types.
- **Device mismatches**  
  Check `be.get_device()` and confirm your tensors reside on the correct device.

Potential Future Backends
-------------------------

- Support for a **CuPy** backend is under consideration. If you have experience with this library and would like to contribute, please reach out!

.. note::
   For new contributions or questions about the backend layer, please open an issue or pull request on our GitHub repository. We welcome feedback and improvements!

How to Extend This
------------------

**Scenario:** Add a backend-agnostic utility function to Optiland.

**Step 1:** Write the function using only ``import optiland.backend as be`` — never import
``numpy`` or ``torch`` directly inside Optiland modules.
**Step 2:** If the function needs a backend-specific API (e.g., ``torch.autograd.grad``), add
abstract and concrete implementations to ``AbstractBackend``, ``NumpyBackend``, and
``TorchBackend`` in ``optiland/backend/``.
**Step 3:** Test on both backends using the ``set_test_backend`` fixture from
``tests/conftest.py`` and ``assert_allclose`` from ``tests/utils.py``.

For step-by-step guidance, see :ref:`extension_recipes`.
