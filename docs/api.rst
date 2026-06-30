.. _api:

=============
API Reference
=============

This page documents the stable public API of :mod:`py_amica`.

Most users will interact with one of two interfaces:

* :class:`py_amica.Amica` for fitting AMICA directly on NumPy arrays.
* :func:`py_amica.fit_ica` for MNE-Python workflows.

Core API
========

.. currentmodule:: py_amica

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Amica
   AmicaConfig
   AmicaResult

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fit_ica

Low-level Solver
----------------

.. autofunction:: py_amica.amica
