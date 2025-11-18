#!/bin/bash
# Quick build and test script for realjam package

set -e

echo "================================"
echo "Building RealJam Package"
echo "================================"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Build the package
echo "Building package..."
python -m build

echo ""
echo "✓ Build complete!"
echo ""
echo "Output files:"
ls -lh dist/

echo ""
echo "================================"
echo "Next Steps:"
echo "================================"
echo ""
echo "1. Test locally:"
echo "   pip install -e ."
echo "   realjam-download-weights --help"
echo "   realjam-start-server --help"
echo ""
echo "2. Test from wheel:"
echo "   pip install dist/realjam-0.1.0-py3-none-any.whl"
echo ""
echo "3. Upload to TestPyPI:"
echo "   python -m twine upload --repository testpypi dist/*"
echo ""
echo "4. Upload to PyPI:"
echo "   python -m twine upload dist/*"
echo ""
