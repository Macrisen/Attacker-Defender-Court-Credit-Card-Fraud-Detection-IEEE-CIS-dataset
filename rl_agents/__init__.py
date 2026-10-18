"""Mac runtime initialization. Training launcher also unifies DYLD_LIBRARY_PATH.

Import order alone permits inference but can deadlock mixed gradient/inference
work. scripts/run_remaining_phases.py re-execs with Homebrew libomp selected
before Python starts; actual mixed-runtime tests validate that configuration.
"""
import sys
if sys.platform=='darwin':
    import lightgbm
