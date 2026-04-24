"""pyMetaheuristic src — Engine registry."""
from .protocol import (
    BaseEngine, ProblemSpec, EngineConfig, CapabilityProfile,
    CandidateRecord, EngineState, OptimizationResult,
)

from .abco         import ABCOEngine
from .acgwo        import ACGWOEngine
from .afsa         import AFSAEngine
from .alo          import ALOEngine
from .aoa          import AOAEngine
from .ars          import ARSEngine
from .bat_a        import BATAEngine
from .bbo          import BBOEngine
from .ca           import CAEngine
from .cat_so       import CAT_SOEngine
from .cem          import CEMEngine
from .chicken_so   import CHICKEN_SOEngine
from .clonalg      import CLONALGEngine
from .coati_oa     import COATI_OAEngine
from .cockroach_so import COCKROACH_SOEngine
from .csa          import CSAEngine
from .cuckoo_s     import CUCKOO_SEngine
from .da           import DAEngine
from .de           import DEEngine
from .dfo          import DFOEngine
from .dvba         import DVBAEngine
from .eho          import EHOEngine
from .fda          import FDAEngine
from .firefly_a    import FIREFLY_AEngine
from .fpa          import FPAEngine
from .ga           import GAEngine
from .gmo          import GMOEngine
from .goa          import GOAEngine
from .gsa          import GSAEngine
from .gwo          import GWOEngine
from .hho          import HHOEngine
from .hsa          import HSAEngine
from .hus          import HUSEngine
from .i_gwo        import I_GWOEngine
from .i_woa        import I_WOAEngine
from .jso          import JSOEngine
from .jy           import JYEngine
from .kha          import KHAEngine
from .mbo          import MBOEngine
from .memetic_a    import MEMETIC_AEngine
from .mfa          import MFAEngine
from .mvo          import MVOEngine
from .pbil         import PBILEngine
from .pfa          import PFAEngine
from .pcx          import PCXEngine
from .pso          import PSOEngine
from .random_s     import RANDOM_SEngine
from .sa           import SAEngine
from .sine_cosine_a import SINE_COSINE_AEngine
from .sos          import SOSEngine
from .spbo         import SPBOEngine
from .ssa          import SSAEngine
from .tlbo         import TLBOEngine
from .woa          import WOAEngine

REGISTRY: dict[str, type[BaseEngine]] = {
    cls.algorithm_id: cls
    for cls in [
        ABCOEngine, ACGWOEngine, AFSAEngine, ALOEngine, AOAEngine,
        ARSEngine, BATAEngine, BBOEngine, CAEngine, CAT_SOEngine,
        CEMEngine, CHICKEN_SOEngine, CLONALGEngine, COATI_OAEngine,
        COCKROACH_SOEngine, CSAEngine, CUCKOO_SEngine, DAEngine,
        DEEngine, DFOEngine, DVBAEngine, EHOEngine, FDAEngine,
        FIREFLY_AEngine, FPAEngine, GAEngine, GMOEngine, GOAEngine,
        GSAEngine, GWOEngine, HHOEngine, HSAEngine, HUSEngine,
        I_GWOEngine, I_WOAEngine, JSOEngine, JYEngine, KHAEngine,
        MBOEngine, MEMETIC_AEngine, MFAEngine, MVOEngine, PBILEngine,
        PFAEngine, PCXEngine, PSOEngine, RANDOM_SEngine, SAEngine, SINE_COSINE_AEngine,
        SOSEngine, SPBOEngine, SSAEngine, TLBOEngine, WOAEngine,
    ]
}

__all__ = ["REGISTRY", "BaseEngine", "ProblemSpec", "EngineConfig",
           "CapabilityProfile", "CandidateRecord", "EngineState", "OptimizationResult"]


# Enable native candidate injection for the algorithms whose state can absorb
# migrants safely through either the generic population-replacement policy or
# an engine-specific repair implementation.
_INJECTION_ENABLED = {
    "abco", "acgwo", "afsa", "aoa", "bbo", "ca", "cat_so", "clonalg",
    "coati_oa", "cockroach_so", "csa", "cuckoo_s", "de", "dfo", "dvba",
    "eho", "fda", "firefly_a", "fpa", "ga", "gmo", "goa", "gsa", "gwo",
    "hus", "i_gwo", "jso", "jy", "mbo", "memetic_a", "mfa", "mvo",
    "pfa", "pcx", "pso", "random_s", "sa", "sine_cosine_a", "sos", "spbo",
    "ssa", "tlbo", "woa", "alo", "ars", "bat_a", "cem", "da", "hho",
    "i_woa",
}
for _aid in _INJECTION_ENABLED:
    REGISTRY[_aid].capabilities.supports_candidate_injection = True


# Curated DOI metadata derived from the user-supplied algorithm table.
# Only canonical DOI strings are retained here. Non-DOI links from the table
# are intentionally omitted so BaseEngine.info() reports only DOI values.
_ALGORITHM_DOIS: dict[str, str] = {
    "acgwo": "10.1007/s42835-023-01621-w",
    "alo": "10.1016/j.advengsoft.2015.01.010",
    "aoa": "10.1016/j.cma.2020.113609",
    "bbo": "10.1109/TEVC.2008.919004",
    "cat_so": "10.1007/978-3-540-36668-3_94",
    "chicken_so": "10.1007/978-3-319-11857-4_10",
    "coati_oa": "10.1016/j.knosys.2022.110011",
    "cockroach_so": "10.1109/ICCET.2010.5485993",
    "cem": "10.1016/S0377-2217(96)00385-2",
    "csa": "10.1016/j.compstruc.2016.03.001",
    "ca": "10.1142/9789814534116",
    "de": "10.1023/A:1008202821328",
    "dfo": "10.15439/2014F142",
    "da": "10.1007/s00521-015-1920-1",
    "dvba": "10.1109/INCoS.2014.40",
    "eho": "10.1109/ISCBI.2015.8",
    "fda": "10.1016/j.cie.2021.107224",
    "gmo": "10.1007/s00500-023-08202-z",
    "goa": "10.1016/j.advengsoft.2017.01.004",
    "gsa": "10.1016/j.ins.2009.03.004",
    "gwo": "10.1016/j.advengsoft.2013.12.007",
    "hsa": "10.1177/003754970107600201",
    "hho": "10.1016/j.future.2019.02.028",
    "hus": "10.1109/ICSCCW.2009.5379451",
    "i_gwo": "10.1016/j.eswa.2020.113917",
    "i_woa": "10.1016/j.jcde.2019.02.002",
    "jso": "10.1016/j.amc.2020.125535",
    "kha": "10.1016/j.asoc.2016.08.041",
    "mbo": "10.1007/s00521-015-1923-y",
    "mfa": "10.1016/j.knosys.2015.07.006",
    "mvo": "10.1007/s00521-015-1870-7",
    "pso": "10.1109/ICNN.1995.488968",
    "pfa": "10.1016/j.asoc.2019.03.012",
    "pcx": "10.1162/106365602760972767",
    "random_s": "10.1080/01621459.1953.10501200",
    "ssa": "10.1016/j.advengsoft.2017.07.002",
    "sine_cosine_a": "10.1016/j.knosys.2015.12.022",
    "spbo": "10.1016/j.advengsoft.2020.102804",
    "sos": "10.1016/j.compstruc.2014.03.007",
    "tlbo": "10.1016/j.compstruc.2014.03.007",
    "woa": "10.1016/j.advengsoft.2016.01.008",
}

for _aid, _doi in _ALGORITHM_DOIS.items():
    if _aid in REGISTRY:
        _reference = dict(getattr(REGISTRY[_aid], "_REFERENCE", {}) or {})
        _reference["doi"] = _doi
        REGISTRY[_aid]._REFERENCE = _reference
