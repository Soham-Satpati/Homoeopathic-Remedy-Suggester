"""Look and feel: the demo's soft slate-blue palette, tuned for a desktop app."""

from __future__ import annotations

BG = "#f6f8fb"
SURFACE = "#ffffff"
INK = "#1e293b"
MUTED = "#64748b"
BORDER = "#dbe3ee"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
ACCENT_SOFT = "#eff6ff"
DANGER = "#b91c1c"
DANGER_SOFT = "#fef2f2"
WARN = "#92400e"
WARN_SOFT = "#fffbeb"
OK = "#15803d"
OK_SOFT = "#f0fdf4"

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 10.5pt; color: {INK}; }}
QMainWindow, QDialog, QWidget#page {{ background: {BG}; }}
QLabel#h1 {{ font-size: 18pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 13pt; font-weight: 600; }}
QLabel#muted, QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel#banner {{ background: {DANGER_SOFT}; color: {DANGER}; border: 2px solid #dc2626;
                 border-radius: 8px; padding: 10px 12px; }}
QLabel#info {{ background: {ACCENT_SOFT}; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 12px; }}
QLabel#warnbox {{ background: {WARN_SOFT}; color: {WARN}; border: 1px solid #fcd34d; border-radius: 8px;
                  padding: 10px 12px; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#nav {{ background: #eef2f7; border-right: 1px solid {BORDER}; }}
QPushButton {{ background: {SURFACE}; border: 1px solid #cbd5e1; border-radius: 7px; padding: 7px 16px; }}
QPushButton:hover {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT_DARK}; }}
QPushButton:disabled {{ color: #94a3b8; background: #f1f5f9; border-color: #e2e8f0; }}
QPushButton[primary="true"] {{ background: {ACCENT}; color: white; border-color: {ACCENT}; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_DARK}; color: white; }}
QPushButton[primary="true"]:disabled {{ background: #93c5fd; border-color: #93c5fd; color: white; }}
QPushButton[danger="true"] {{ color: {DANGER}; border-color: #fca5a5; }}
QPushButton#navbtn {{ text-align: left; border: none; border-radius: 8px; padding: 9px 14px; background: transparent; }}
QPushButton#navbtn:checked {{ background: {SURFACE}; color: {ACCENT_DARK}; font-weight: 600;
                              border: 1px solid {BORDER}; }}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {{ background: {SURFACE}; border: 1px solid #cbd5e1;
    border-radius: 6px; padding: 5px 7px; selection-background-color: #bfdbfe; selection-color: {INK}; }}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QTableWidget, QTableView, QListWidget, QTreeWidget {{ background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 8px; gridline-color: #eef2f7; selection-background-color: {ACCENT_SOFT};
    selection-color: {INK}; }}
QHeaderView::section {{ background: #f1f5f9; border: none; border-bottom: 1px solid {BORDER};
    padding: 6px; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {SURFACE}; top: -1px; }}
QTabBar::tab {{ padding: 7px 16px; border: 1px solid transparent; border-bottom: none; }}
QTabBar::tab:selected {{ background: {SURFACE}; border-color: {BORDER}; border-top-left-radius: 6px;
    border-top-right-radius: 6px; font-weight: 600; }}
QProgressBar {{ border: 1px solid {BORDER}; border-radius: 5px; background: {SURFACE}; height: 8px;
    text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QStatusBar {{ background: #eef2f7; color: {MUTED}; }}
QToolTip {{ background: {INK}; color: white; border: none; padding: 5px; }}
"""

REPORT_CSS = f"""
body {{ font-family: 'Segoe UI', sans-serif; font-size: 10.5pt; color: {INK}; }}
h1 {{ font-size: 16pt; margin: 0 0 4px 0; }}
h2 {{ font-size: 11.5pt; color: #1e3a8a; margin: 16px 0 6px 0; }}
.muted {{ color: {MUTED}; }}
.flag {{ background: {DANGER_SOFT}; border: 2px solid #dc2626; padding: 8px; color: #7f1d1d; }}
.warn {{ color: {WARN}; }}
.rem {{ font-weight: 600; font-size: 11pt; }}
.ev {{ color: #334155; margin-left: 18px; }}
td {{ padding: 3px 8px 3px 0; vertical-align: top; }}
th {{ text-align: left; padding: 4px 8px 4px 0; border-bottom: 1px solid {BORDER}; }}
"""
