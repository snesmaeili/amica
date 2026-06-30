.. _api:

=============
API Reference
=============

This page documents the stable public API of :mod:`amica`.

Most users will interact with one of two interfaces:

* :class:`amica.Amica` for fitting AMICA directly on NumPy arrays.
* :func:`amica.fit_ica` for MNE-Python workflows.

Core API
========

.. currentmodule:: amica

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

.. autofunction:: amica.amica
