
"""Pytest configuration for headless image rendering."""

import matplotlib

# Use a noninteractive backend during tests.
# Allows figure creation and saving without Tkinter.
matplotlib.use("Agg")
