# Package Rename Summary: MUFBVAR → MBFVAR

## Overview

This document summarizes the complete package rename from **MUFBVAR** (Multi-Frequency) to **MBFVAR** (Mixed-Frequency Bayesian VAR).

## What Was Changed

### 1. Core Package Structure

#### Directory and File Names
- **Directory**: `MUFBVAR/` → `MBFVAR/`
- **Module File**: `MUFBVAR/mufbvar_data.py` → `MBFVAR/mbfvar_data.py`
- **Module File**: `MUFBVAR/multifrequency_var.py` → `MBFVAR/mbfvar.py`

#### Class Names
- **Data Class**: `mufbvar_data` → `mbfvar_data`
- **Model Class**: `multifrequency_var` → `MixedFrequencyBVAR`

### 2. Configuration Files

#### pyproject.toml
```toml
# Before
name = "MUFBVAR"
include = ["MUFBVAR*"]
description = "Multifrequency Bayesian VAR Model"

# After
name = "MBFVAR"
include = ["MBFVAR*"]
description = "Mixed-Frequency Bayesian VAR Model"
```

#### setup.py
All C++ extension modules updated:
- `MUFBVAR.cholcov.cholcov_module` → `MBFVAR.cholcov.cholcov_module`
- `MUFBVAR.inverse.matrix_inversion` → `MBFVAR.inverse.matrix_inversion`
- `MUFBVAR.pseudo_inverse.pseudo_inverse` → `MBFVAR.pseudo_inverse.pseudo_inverse`
- `MUFBVAR.solve.solve` → `MBFVAR.solve.solve`

All file paths updated:
- `MUFBVAR/cholcov/...` → `MBFVAR/cholcov/...`
- `MUFBVAR/inverse/...` → `MBFVAR/inverse/...`
- etc.

### 3. Python Module Files

#### MBFVAR/__init__.py
```python
# Before
from .mufbvar_data import mufbvar_data
from .multifrequency_var import multifrequency_var

# After
from .mbfvar_data import mbfvar_data
from .mbfvar import MixedFrequencyBVAR
```

#### MBFVAR/mbfvar_data.py
- Class name: `class mufbvar_data:` → `class mbfvar_data:`
- Docstrings: "MUFBVAR estimation" → "MBFVAR estimation"
- Examples: `import MUFBVAR`, `MBFVAR.mbfvar_data()`

#### MBFVAR/mbfvar.py
- Class name: `class multifrequency_var:` → `class MixedFrequencyBVAR:`
- Docstrings: "MUFBVAR model" → "MBFVAR model"
- Examples: `MBFVAR.MixedFrequencyBVAR()`

#### MBFVAR/_estimation.py
- Function parameter: `def fit(self, mufbvar_data, ...)` → `def fit(self, mbfvar_data, ...)`
- All variable references: `mufbvar_data.attribute` → `mbfvar_data.attribute`
- Docstring references

#### MBFVAR/_hyp_opt.py
- Import: `from .mufbvar_data import mufbvar_data` → `from .mbfvar_data import mbfvar_data`

### 4. Examples

#### Python Examples (5 files updated)
All Python example files updated:
- `examples/basic_example.py`
- `examples/advanced_example.py`
- `examples/python_example.py`
- `examples/test_basic.py`
- `examples/generate_data.py`

Changes:
```python
# Before
import MUFBVAR
data_in = MUFBVAR.mufbvar_data(data, trans, frequencies)
model = MUFBVAR.multifrequency_var(nsim, nburn, nlags, thining)

# After
import MBFVAR
data_in = MBFVAR.mbfvar_data(data, trans, frequencies)
model = MBFVAR.MixedFrequencyBVAR(nsim, nburn, nlags, thining)
```

#### R Example
File: `examples/r_example.R`

Changes:
```r
# Before
mufbvar <- import("MUFBVAR")
data_in <- mufbvar$mufbvar_data(data, trans, frequencies)
model <- mufbvar$multifrequency_var(nsim, nburn, nlags, thining)

# After
mbfvar <- import("MBFVAR")
data_in <- mbfvar$mbfvar_data(data, trans, frequencies)
model <- mbfvar$MixedFrequencyBVAR(nsim, nburn, nlags, thining)
```

### 5. Documentation Files

#### docs/conf.py
```python
# Before
project = 'MUFBVAR'

# After
project = 'MBFVAR'
```

#### docs/source/ (4 RST files)
Files updated:
- `intro.rst`
- `examples.rst`
- `modules.rst`
- `MUFBVAR.rst` → `MBFVAR.rst` (renamed)

All code examples, references, and cross-references updated.

#### README.md
All code examples, package imports, and references updated throughout.

#### DOCUMENTATION_IMPROVEMENTS.md
All package references updated.

### 6. Build Artifacts

- Cleaned `dist/` directory (removed old .whl and .tar.gz files)
- Build artifacts in `docs/_build/` will be regenerated automatically

## API Changes Summary

### Import Changes
```python
# Old API
import MUFBVAR

# New API
import MBFVAR
```

### Class Usage Changes
```python
# Old API
data_in = MUFBVAR.mufbvar_data(data, trans, frequencies)
model = MUFBVAR.multifrequency_var(nsim, nburn, nlags, thining)

# New API
data_in = MBFVAR.mbfvar_data(data, trans, frequencies)
model = MBFVAR.MixedFrequencyBVAR(nsim, nburn, nlags, thining)
```

### R API Changes
```r
# Old API
mufbvar <- import("MUFBVAR")
data_in <- mufbvar$mufbvar_data(...)
model <- mufbvar$multifrequency_var(...)

# New API
mbfvar <- import("MBFVAR")
data_in <- mbfvar$mbfvar_data(...)
model <- mbfvar$MixedFrequencyBVAR(...)
```

## Verification Steps

1. ✅ **Package builds successfully**
   ```bash
   pip install -e .
   ```

2. ✅ **Package imports correctly**
   ```python
   import MBFVAR
   print(MBFVAR.mbfvar_data)
   print(MBFVAR.MixedFrequencyBVAR)
   ```

3. ✅ **Tests pass**
   ```bash
   python examples/test_basic.py
   ```

4. ✅ **Examples work**
   - All Python examples use correct imports
   - R example uses correct package name

## Files Changed (47 total)

### Created/Renamed:
- `MBFVAR/__init__.py`
- `MBFVAR/mbfvar_data.py` (from `MUFBVAR/mufbvar_data.py`)
- `MBFVAR/mbfvar.py` (from `MUFBVAR/multifrequency_var.py`)
- `docs/source/MBFVAR.rst` (from `MUFBVAR.rst`)
- All other files in `MBFVAR/` directory

### Modified:
- `pyproject.toml`
- `setup.py`
- `README.md`
- `DOCUMENTATION_IMPROVEMENTS.md`
- All files in `examples/` (6 files)
- All files in `docs/source/` (4 RST files)
- `docs/conf.py`

### Deleted:
- Old `MUFBVAR/` directory and all contents
- Old distribution files in `dist/`

## Breaking Changes

**This is a breaking change for existing users.** Users will need to update their code:

1. Change `import MUFBVAR` to `import MBFVAR`
2. Change `MUFBVAR.mufbvar_data` to `MBFVAR.mbfvar_data`
3. Change `MUFBVAR.multifrequency_var` to `MBFVAR.MixedFrequencyBVAR`

## Migration Guide for Users

### For Python Users

```python
# Step 1: Uninstall old package
pip uninstall mufbvar

# Step 2: Install new package
pip install mbfvar  # or pip install git+https://github.com/laurentflorin/MBFVAR.git

# Step 3: Update your code
# Old:
import MUFBVAR
data = MUFBVAR.mufbvar_data(...)
model = MUFBVAR.multifrequency_var(...)

# New:
import MBFVAR
data = MBFVAR.mbfvar_data(...)
model = MBFVAR.MixedFrequencyBVAR(...)
```

### For R Users

```r
# Step 1: Update Python package (see above)

# Step 2: Update R code
# Old:
mufbvar <- import("MUFBVAR")
data_in <- mufbvar$mufbvar_data(...)
model <- mufbvar$multifrequency_var(...)

# New:
mbfvar <- import("MBFVAR")
data_in <- mbfvar$mbfvar_data(...)
model <- mbfvar$MixedFrequencyBVAR(...)
```

## Notes

1. The rename was comprehensive and touched all parts of the codebase
2. All internal references were updated to maintain consistency
3. The package description was improved to "Mixed-Frequency Bayesian VAR Model"
4. Class naming now follows PascalCase convention for `MixedFrequencyBVAR`
5. All tests pass with the new package name
6. Documentation regeneration will happen automatically when building docs

## Rationale for Changes

- **Package Name**: MBFVAR is more concise and matches the "Mixed-Frequency" terminology
- **Class Name**: `MixedFrequencyBVAR` follows Python PascalCase convention and is more descriptive
- **Consistency**: All references updated for a clean, professional package
