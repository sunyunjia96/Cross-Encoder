#!/usr/bin/env python3

from .data_loader_Columbia import ImagerLoader as Columbia
from .data_loader_UTMultiview import ImagerLoader as UTMultiview
from .data_loader_MPIIGaze import ImagerLoader as MPIIGaze
from .data_loader_XGaze import ImagerLoader as XGaze
from .data_loader_unite import ImagerLoader as Unite

__all__ = ('Columbia', 'UTMultiview', 'MPIIGaze', 'XGaze', 'Unite')
