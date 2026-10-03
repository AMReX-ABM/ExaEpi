"""Reference ("oracle") implementation of ExaEpi's in-process population generation.

Every stage is order-independent and keyed on global identifiers through KR64 (kr64.py), so the
C++ port in src/ can be checked against it stage by stage. See popgen/stages.py for the stage table
shared with C++.
"""
