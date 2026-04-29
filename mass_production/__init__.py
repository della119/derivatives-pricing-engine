"""
Mass Production module
=======================
Generates user-input Excel templates and processes uploaded files
to perform batch valuations across many products at once.
"""
from .template import build_template_workbook, get_template_bytes
from .processor import process_uploaded_workbook

__all__ = [
    "build_template_workbook",
    "get_template_bytes",
    "process_uploaded_workbook",
]
