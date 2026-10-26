"""Launcher for the HoRUS desktop app (PyInstaller entry point)."""
import multiprocessing
import sys

if __name__ == "__main__":
    # Book imports run in a child process (horus/book_worker.py). In the packaged app the child is
    # this same HoRUS.exe; freeze_support() makes it run the import and exit instead of opening a window.
    multiprocessing.freeze_support()
    from horus_app.app import main
    sys.exit(main())
