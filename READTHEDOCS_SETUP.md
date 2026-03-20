# ReadTheDocs Setup Instructions

This document explains how to set up and deploy the MBFVAR documentation to ReadTheDocs (https://mbfvar.readthedocs.io).

## Prerequisites

- GitHub repository: https://github.com/laurentflorin/MBFVAR
- ReadTheDocs account (sign up at https://readthedocs.org)

## Setup Steps

### 1. Import the Project to ReadTheDocs

1. Go to https://readthedocs.org and log in with your GitHub account
2. Click on "Import a Project"
3. Find and select the `laurentflorin/MBFVAR` repository
4. Click "Import"

### 2. Configure the Project

The repository already includes a `.readthedocs.yml` configuration file that:
- Uses Python 3.11
- Installs documentation dependencies from `docs/requirements.txt`
- Builds HTML documentation using Sphinx
- Uses the Furo theme for a modern look

**No additional configuration is needed!** The `.readthedocs.yml` file handles everything.

### 3. Trigger the First Build

1. Once imported, ReadTheDocs will automatically trigger a build
2. You can monitor the build progress in the ReadTheDocs dashboard
3. The build should complete successfully in a few minutes

### 4. Access Your Documentation

After the build completes, your documentation will be available at:
- **https://mbfvar.readthedocs.io** (latest version)
- **https://mbfvar.readthedocs.io/en/stable/** (stable version)
- **https://mbfvar.readthedocs.io/en/latest/** (development version)

### 5. Configure Webhooks (Automatic)

ReadTheDocs automatically sets up a webhook with your GitHub repository. This means:
- Every push to the repository triggers a new documentation build
- Pull requests also get documentation previews
- No manual rebuilding is needed

## Custom Domain (Optional)

If you want to use a custom domain like `docs.yourdomain.com`:

1. Go to your ReadTheDocs project settings
2. Navigate to "Domains"
3. Add your custom domain
4. Follow the DNS configuration instructions

## Documentation Structure

The documentation is organized as follows:

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Documentation homepage
├── requirements.txt     # Documentation build dependencies
└── source/
    ├── intro.rst        # Introduction and getting started
    ├── examples.rst     # Usage examples
    ├── modules.rst      # API reference table of contents
    └── MBFVAR.rst       # Auto-generated API documentation
```

## Updating Documentation

To update the documentation:

1. Edit the `.rst` files in the `docs/` or `docs/source/` directories
2. Test locally by running:
   ```bash
   cd docs
   make html
   ```
3. View the generated HTML in `docs/_build/html/index.html`
4. Commit and push your changes
5. ReadTheDocs will automatically rebuild the documentation

## Troubleshooting

### Build Fails

1. Check the build logs in ReadTheDocs dashboard
2. Ensure all dependencies are listed in `docs/requirements.txt`
3. Test the build locally with `cd docs && make html`

### Documentation Not Updating

1. Check if the webhook is properly configured in GitHub repository settings
2. Manually trigger a build from the ReadTheDocs dashboard
3. Verify that the latest commit was pushed to the repository

### Import Errors in Documentation

The `.readthedocs.yml` file is configured to:
1. Install the package itself (`pip install -e .`)
2. Install all dependencies

If you still see import errors, ensure the module is properly importable.

## Additional Resources

- [ReadTheDocs Documentation](https://docs.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Furo Theme Documentation](https://pradyunsg.me/furo/)

## Files Added/Modified

This setup involved creating/modifying the following files:

- `.readthedocs.yml` - ReadTheDocs configuration
- `docs/requirements.txt` - Documentation build dependencies
- `docs/conf.py` - Enhanced Sphinx configuration
- `docs/index.rst` - Updated to use MBFVAR (not MUFBVAR)
- `docs/source/MBFVAR.rst` - Renamed from MUFBVAR.rst
- `README.md` - Updated documentation links to point to ReadTheDocs
