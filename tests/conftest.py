import os
import tempfile

# app.main reads MAKER_OUTPUT at import time, so point it at a throwaway directory first.
os.environ.setdefault('MAKER_OUTPUT', tempfile.mkdtemp(prefix='3d-maker-test-'))
