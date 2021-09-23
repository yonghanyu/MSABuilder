from typing import Any, Callable, Mapping, Optional, Sequence

class HHSearchParam:
    def __init__(self,
               n_cpu: int = 4,
               n_iter: int = 3,
               e_value: float = 0.001,
               maxseq: int = 1_000_000,
               realign_max: int = 100_000,
               p: int = 20,
               z: int = 500,
               flags: Optional[Sequence[str]] = None):

        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.maxseq = maxseq
        self.realign_max = realign_max
        self.p = p
        self.z = z
        self.flags = flags
    

class HmmSearchParam:
    def __init__(self,
               n_cpu: int = 8,
               e_value: float = 0.0001,
               z_value: Optional[int] = None,
               filter_f1: float = 0.0005,
               filter_f2: float = 0.00005,
               filter_f3: float = 0.0000005,
               incdom_e: Optional[float] = None,
               dom_e: Optional[float] = None,
               flags: Optional[Sequence[str]] = None):
        self.flags = flags
        self.n_cpu = n_cpu
        self.e_value = e_value
        self.z_value = z_value
        self.filter_f1 = filter_f1
        self.filter_f2 = filter_f2
        self.filter_f3 = filter_f3
        self.incdom_e = incdom_e
        self.dom_e = dom_e


class HHBlitsParam:
    def __init__(self,
               n_cpu: int = 4,
               n_iter: int = 3,
               e_value: float = 0.001,
               maxseq: int = 1_000_000,
               realign_max: int = 100_000,
               maxfilt: int = 100_000,
               min_prefilter_hits: int = 1000,
               all_seqs: bool = False,
               alt: Optional[int] = None,
               p: int = 20,
               z: int = 500,
               flags: Optional[Sequence[str]] = None):
        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.maxseq = maxseq
        self.realign_max = realign_max
        self.maxfilt = maxfilt
        self.min_prefilter_hits = min_prefilter_hits
        self.all_seqs = all_seqs
        self.alt = alt
        self.p = p
        self.z = z
        self.flags = flags


class JackHmmerParam:
    def __init__(self,
               n_cpu: int = 8,
               n_iter: int = 1,
               e_value: float = 0.0001,
               z_value: Optional[int] = None,
               get_tblout: bool = False,
               filter_f1: float = 0.0005,
               filter_f2: float = 0.00005,
               filter_f3: float = 0.0000005,
               incdom_e: Optional[float] = None,
               dom_e: Optional[float] = None,
               flags: Optional[Sequence[str]] = None):
        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.z_value = z_value
        self.filter_f1 = filter_f1
        self.filter_f2 = filter_f2
        self.filter_f3 = filter_f3
        self.incdom_e = incdom_e
        self.dom_e = dom_e
        self.get_tblout = get_tblout
        self.flags = flags