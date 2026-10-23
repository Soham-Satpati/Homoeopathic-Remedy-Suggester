"""Launcher for the HoRUS desktop app (PyInstaller entry point)."""
import sys

from horus_app.app import main

if __name__ == "__main__":
    sys.exit(main())
