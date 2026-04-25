"""pyMetaheuristic src — Engine registry."""
from .protocol import (
    BaseEngine, ProblemSpec, EngineConfig, CapabilityProfile,
    CandidateRecord, EngineState, OptimizationResult,
)

from .abco          import ABCOEngine
from .acgwo         import ACGWOEngine
from .acor          import ACOREngine
from .aefa          import AEFAEngine
from .aeo           import AEOEngine
from .afsa          import AFSAEngine
from .aft           import AFTEngine
from .agto          import AGTOEngine
from .aha           import AHAEngine
from .ala           import ALAEngine
from .alo           import ALOEngine
from .ao            import AOEngine
from .aoa           import AOAEngine
from .apo           import APOEngine
from .arch_oa       import ARCHOAEngine
from .aro           import AROEngine
from .ars           import ARSEngine
from .artemisinin_o import ArtemisininOEngine
from .aso           import ASOEngine
from .avoa          import AVOAEngine
from .bat_a         import BATAEngine
from .bbo           import BBOEngine
from .bboa          import BBOAEngine
from .bco           import BCOEngine
from .bea           import BEAEngine
from .bes           import BESEngine
from .bfo           import BFOEngine
from .bka           import BKAEngine
from .bmo           import BMOEngine
from .bo_bonobo     import BonobOEngine
from .boa           import BOAEngine
from .bro           import BROEngine
from .bsa           import BSAEngine
from .bso           import BSOEngine
from .bwo           import BWOEngine
from .ca            import CAEngine
from .camel         import CamelEngine
from .capsa         import CapSAEngine
from .cat_so        import CAT_SOEngine
from .cddo          import CDDOEngine
from .cdo           import CDOEngine
from .cem           import CEMEngine
from .cgo           import CGOEngine
from .chameleon_sa  import ChameleonSAEngine
from .chicken_so    import CHICKEN_SOEngine
from .chio          import CHIOEngine
from .choa          import ChOAEngine
from .circle_sa     import CIRCLESAEngine
from .clonalg       import CLONALGEngine
from .coa           import COAEngine
from .coati_oa      import COATI_OAEngine
from .cockroach_so  import COCKROACH_SOEngine
from .coot          import COOTEngine
from .cro           import CROEngine
from .csa           import CSAEngine
from .csbo          import CSBOEngine
from .cuckoo_s      import CUCKOO_SEngine
from .da            import DAEngine
from .dbo           import DBOEngine
from .ddao          import DDAOEngine
from .de            import DEEngine
from .dfo           import DFOEngine
from .dmoa          import DMOAEngine
from .do_dandelion  import DandelionOEngine
from .doa           import DOAEngine
from .dso           import DSOEngine
from .dvba          import DVBAEngine
from .eao           import EAOEngine
from .eco           import ECOEngine
from .edo           import EDOEngine
from .efo           import EFOEngine
from .eho           import EHOEngine
from .elk_ho        import ElkHOEngine
from .eo            import EOEngine
from .eoa           import EOAEngine
from .ep            import EPEngine
from .epc           import EPCEngine
from .es            import ESEngine
from .esc           import ESCEngine
from .eso           import ESOEngine
from .esoa          import ESOAEngine
from .eto           import ETOEngine
from .evo           import EVOEngine
from .fata          import FATAEngine
from .fbio          import FBIOEngine
from .fda           import FDAEngine
from .fdo           import FDOEngine
from .ffa           import FFAEngine
from .ffo           import FFOEngine
from .firefly_a     import FIREFLY_AEngine
from .fla           import FLAEngine
from .flood_a       import FloodAEngine
from .foa           import FOAEngine
from .fox           import FOXEngine
from .fpa           import FPAEngine
from .fss           import FSSEngine
from .fwa           import FWAEngine
from .ga            import GAEngine
from .gazelle_oa    import GazelleOAEngine
from .gbo           import GBOEngine
from .gco           import GCOEngine
from .gea           import GEAEngine
from .ggo           import GGOEngine
from .gjo           import GJOEngine
from .gkso          import GKSOEngine
from .gmo           import GMOEngine
from .gndo          import GNDOEngine
from .go_growth     import GOGrowthEngine
from .goa           import GOAEngine
from .gsa           import GSAEngine
from .gska          import GSKAEngine
from .gso           import GSOEngine
from .gto           import GTOEngine
from .gwo           import GWOEngine
from .hba           import HBAEngine
from .hbo           import HBOEngine
from .hc            import HCEngine
from .hco           import HCOEngine
from .hde           import HDEEngine
from .heoa          import HEOAEngine
from .hgs           import HGSEngine
from .hgso          import HGSOEngine
from .hho           import HHOEngine
from .hiking_oa     import HikingOAEngine
from .ho_hippo      import HippoEngine
from .horse_oa      import HorseOAEngine
from .hsa           import HSAEngine
from .hsaba         import HSABAEngine
from .hus           import HUSEngine
from .i_gwo         import I_GWOEngine
from .i_woa         import I_WOAEngine
from .ica           import ICAEngine
from .ilshade       import ILSHADEEngine
from .info          import INFOEngine
from .ivya          import IVYAEngine
from .iwo           import IWOEngine
from .jde           import JDEEngine
from .jso           import JSOEngine
from .jy            import JYEngine
from .kha           import KHAEngine
from .lca           import LCAEngine
from .lco           import LCOEngine
from .lfd           import LFDEngine
from .loa           import LOAEngine
from .lpo           import LPOEngine
from .mbo           import MBOEngine
from .memetic_a     import MEMETIC_AEngine
from .mfa           import MFAEngine
from .mfea          import MFEAEngine
from .mfea2         import MFEA2Engine
from .mke           import MKEEngine
from .moss_go       import MossGOEngine
from .mpa           import MPAEngine
from .mrfo          import MRFOEngine
from .msa_e         import MSAEngine
from .mshoa         import MSHOAEngine
from .mso           import MSOEngine
from .mts           import MTSEngine
from .mvo           import MVOEngine
from .nndrea_so     import NNDREASOEngine
from .ngo           import NGOEngine
from .nmm           import NMMEngine
from .nmra          import NMRAEngine
from .nro           import NROEngine
from .ooa           import OOAEngine
from .parrot_o      import ParrotOEngine
from .pbil          import PBILEngine
from .pcx           import PCXEngine
from .pdo           import PDOEngine
from .pfa           import PFAEngine
from .pko           import PKOEngine
from .plba          import PLBAEngine
from .plo           import PLOEngine
from .poa           import POAEngine
from .political_o   import PoliticalOEngine
from .pro           import PROEngine
from .pso           import PSOEngine
from .pss           import PSSEngine
from .puma_o        import PumaOEngine
from .qio           import QIOEngine
from .qsa           import QSAEngine
from .random_s      import RANDOM_SEngine
from .rbmo          import RBMOEngine
from .rfo           import RFOEngine
from .rime          import RIMEEngine
from .roa           import ROAEngine
from .rsa           import RSAEngine
from .rso           import RSOEngine
from .run           import RUNEngine
from .sa            import SAEngine
from .saba          import SABAEngine
from .sacc_eam2     import SACCEAMIIEngine
from .samso         import SAMSOEngine
from .sapo          import SAPOEngine
from .saro          import SAROEngine
from .sbo           import SBOEngine
from .sboa          import SBOAEngine
from .scho          import SCHOEngine
from .scso          import SCSoEngine
from .seaho         import SeaHOEngine
from .serval_oa     import SERVALOAEngine
from .sfo           import SFOEngine
from .sfoa          import SFOAEngine
from .shade         import SHADEEngine
from .shio          import SHIOEngine
from .sho           import SHOEngine
from .sine_cosine_a import SINE_COSINE_AEngine
from .slo           import SLOEngine
from .sma           import SMAEngine
from .smo           import SMOEngine
from .snow_oa       import SnowOAEngine
from .so_snake      import SnakeOptimizerEngine
from .soa           import SOAEngine
from .soo           import SOOEngine
from .sos           import SOSEngine
from .sparrow_sa    import SparrowSAEngine
from .spbo          import SPBOEngine
from .squirrel_sa   import SquirrelSAEngine
from .srsr          import SRSREngine
from .ssa           import SSAEngine
from .ssdo          import SSDOEngine
from .ssio_rl       import SSIORLEngine
from .sso           import SSOEngine
from .sspider_a     import SSPIDERAEngine
from .sto           import STOEngine
from .superb_foa    import SuperbFOAEngine
from .supply_do     import SupplyDOEngine
from .tdo           import TDOEngine
from .thro          import THROEngine
from .tlbo          import TLBOEngine
from .tlco          import TLCOEngine
from .toa           import TOAEngine
from .toc           import TOCEngine
from .tpo           import TPOEngine
from .ts            import TSEngine
from .tsa           import TSAEngine
from .tso           import TSOEngine
from .ttao          import TTAOEngine
from .two           import TWOEngine
from .vcs           import VCSEngine
from .waoa          import WAOAEngine
from .warso         import WARSOEngine
from .wca           import WCAEngine
from .wdo           import WDOEngine
from .who           import WHOEngine
from .wo_wave       import WaveOptEngine
from .woa           import WOAEngine
from .wso           import WSOEngine
from .wutp          import WUTPEngine
from .ydse          import YDSEEngine
from .zoa           import ZOAEngine

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
        ASOEngine,
        BEAEngine,
        BFOEngine,
        CamelEngine,
        CROEngine,
        ESEngine,
        FOAEngine,
        FSSEngine,
        FWAEngine,
        GSOEngine,
        HBAEngine,
        HCEngine,
        HDEEngine,
        HSABAEngine,
        ILSHADEEngine,
        JDEEngine,
        LOAEngine,
        MKEEngine,
        MSHOAEngine,
        MTSEngine,
        NMMEngine,
        PLBAEngine,
        SABAEngine,
        SHADEEngine,
        SOSEngine, SPBOEngine, SSAEngine, TLBOEngine, WOAEngine,
        AROEngine, EOEngine, MPAEngine, MRFOEngine, SMAEngine,
        AOEngine, BESEngine, GJOEngine, GTOEngine, SCSoEngine,
        AGTOEngine, AVOAEngine, FOXEngine, HGSEngine, NGOEngine,
        ARCHOAEngine, EFOEngine, RIMEEngine, TWOEngine, WDOEngine,
        BWOEngine, IWOEngine, SBOEngine, VCSEngine, WHOEngine,
        BSOEngine, CHIOEngine, ICAEngine, QSAEngine, SSDOEngine,
        CGOEngine, GBOEngine, INFOEngine, RUNEngine, TSEngine,
        COAEngine, DMOAEngine, SFOEngine, SHOEngine, SLOEngine,
        SeaHOEngine, SMOEngine, SRSREngine, TSOEngine, ZOAEngine,
        HGSOEngine, MSAEngine, NMRAEngine, POAEngine, SquirrelSAEngine,
        EPEngine, ESOEngine, FLAEngine, NROEngine, SOOEngine,
        EOAEngine, FBIOEngine, GSKAEngine, LCOEngine, SAROEngine,
        AFTEngine, BROEngine, CDDOEngine, DOAEngine, HBOEngine,
        BCOEngine, HCOEngine, SOAEngine, TOAEngine, WARSOEngine,
        ACOREngine, AEOEngine, BSAEngine, CDOEngine, EPCEngine, ESOAEngine, EVOEngine, FDOEngine, FFAEngine, FFOEngine, GCOEngine, MSOEngine, OOAEngine, PSSEngine, SERVALOAEngine, SFOAEngine, SHIOEngine, SSOEngine, SSPIDERAEngine, STOEngine, TDOEngine, THROEngine, TPOEngine, WAOAEngine, WCAEngine,
        BBOAEngine, BMOEngine, CIRCLESAEngine, EAOEngine, TSAEngine,
        AEFAEngine, AHAEngine, ALAEngine, APOEngine, ArtemisininOEngine,
        BKAEngine, BonobOEngine, BOAEngine, CapSAEngine, ChameleonSAEngine,
        ChOAEngine, COOTEngine, CSBOEngine, DBOEngine, DDAOEngine,
        DandelionOEngine, DSOEngine, ECOEngine, EDOEngine, ElkHOEngine,
        ESCEngine, ETOEngine, FATAEngine, FloodAEngine, GazelleOAEngine,
        GEAEngine, GGOEngine, GKSOEngine, GNDOEngine, GOGrowthEngine,
        HEOAEngine, HikingOAEngine, HippoEngine, HorseOAEngine, IVYAEngine,
        LCAEngine, LFDEngine, LPOEngine, MossGOEngine, ParrotOEngine,
        PDOEngine, PKOEngine, PLOEngine, PoliticalOEngine, PROEngine,
        PumaOEngine, QIOEngine, RBMOEngine, RFOEngine, ROAEngine,
        RSAEngine, RSOEngine, SBOAEngine, SCHOEngine, SnowOAEngine,
        SnakeOptimizerEngine, SparrowSAEngine, SuperbFOAEngine, SupplyDOEngine,
        TLCOEngine, TOCEngine, TTAOEngine, WaveOptEngine, WSOEngine,
        WUTPEngine, YDSEEngine,
    ]
}

__all__ = ["REGISTRY", "BaseEngine", "ProblemSpec", "EngineConfig",
           "CapabilityProfile", "CandidateRecord", "EngineState", "OptimizationResult"]


# Enable native candidate injection for the algorithms whose state can absorb
# migrants safely through either the generic population-replacement policy or
# an engine-specific repair implementation.
_INJECTION_ENABLED = {
    "abco", "acgwo", "afsa", "aoa", "aso", "bbo", "bea", "bfo",
    "ca", "camel", "cat_so", "clonalg", "coati_oa", "cockroach_so", "cro", "csa",
    "cuckoo_s", "de", "dfo", "dvba", "eho", "es", "fda", "firefly_a",
    "foa", "fpa", "fss", "fwa", "ga", "gmo", "goa", "gsa",
    "gso", "gwo", "hba", "hde", "hho", "hsaba", "hus", "i_gwo",
    "i_woa", "ilshade", "jde", "jso", "jy", "kha", "loa", "mbo",
    "memetic_a", "mfa", "mke", "mshoa", "mts", "mvo", "nmm", "pbil",
    "pcx", "pfa", "plba", "pso", "random_s", "sa", "saba", "shade",
    "sine_cosine_a", "sos", "spbo", "ssa", "tlbo", "woa", "alo", "ars",
    "bat_a", "cem", "chicken_so", "da",
    "aro", "eo", "mpa", "mrfo", "sma",
    "ao", "bes", "gjo", "gto", "scso",
    "agto", "avoa", "fox", "hgs", "ngo",
    "arch_oa", "efo", "rime", "two", "wdo",
    "bwo", "iwo", "sbo", "vcs", "who",
    "bso", "chio", "ica", "qsa", "ssdo",
    "cgo", "gbo", "info", "run",
    "coa", "dmoa", "sfo", "sho", "slo",
    "seaho", "smo", "srsr", "tso", "zoa",
    "hgso", "msa_e", "nmra", "poa", "squirrel_sa",
    "ep", "eso", "fla", "nro", "soo",
    "eoa", "fbio", "gska", "lco", "saro",
    "aft", "bro", "cddo", "doa", "hbo",
    "bco", "hco", "soa", "toa", "warso",
    "bboa", "bmo", "circle_sa", "eao", "tsa",
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
    "mshoa": "10.3390/math13091500",
    "mke": "10.1016/j.knosys.2016.01.009",
    "loa": "10.1016/j.jcde.2015.06.003",
    "fwa": "10.1016/j.asoc.2017.10.046",
    "foa": "10.1016/j.eswa.2014.05.009",
    "cro": "10.1155/2014/739768",
    "bfo": "10.1109/MCS.2002.1004010",
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
    "aro":  "10.1016/j.engappai.2022.105082",
    "eo":   "10.1016/j.knosys.2019.105190",
    "mrfo": "10.1016/j.engappai.2019.103300",
    "sma":  "10.1016/j.future.2020.03.055",
    "ao":   "10.1016/j.cie.2021.107250",
    "bes":  "10.1007/s10462-019-09732-5",
    "scso": "10.1007/s00366-022-01604-x",
    "agto": "10.1002/int.22535",
    "arch_oa": "10.1007/s10489-020-01893-z",
    "rime":    "10.1016/j.neucom.2023.02.010",
    "bwo":  "10.1016/j.engappai.2019.103249",
    "sbo":  "10.1016/j.engappai.2017.01.006",
    "vcs":  "10.1016/j.advengsoft.2015.11.004",
    "who":  "10.3233/JIFS-190495",
    "bso":  "10.1007/978-3-642-21515-5_36",
    "chio": "10.1007/s00521-020-05296-6",
    "qsa":  "10.1007/s12652-020-02849-4",
    "ssdo": "10.1007/s00521-019-04159-z",
    "cgo":  "10.1007/s10462-020-09867-w",
    "info": "10.1016/j.eswa.2022.116516",
    "run":  "10.1016/j.eswa.2021.115079",
    "dmoa": "10.1016/j.cma.2022.114570",
    "sfo":  "10.1016/j.engappai.2019.01.001",
    "sho":  "10.1016/j.advengsoft.2017.05.014",
    "slo":  "10.14569/IJACSA.2019.0100548",
    "seaho":"10.1007/s10489-022-03994-3",
    "smo":  "10.1007/s12293-013-0128-0",
    "srsr": "10.1016/j.asoc.2017.02.028",
    "msa_e":"10.1007/s12293-016-0212-3",
    "eso":  "10.3390/make7010024",
    "eoa":  "10.1504/IJBIC.2015.10004283",
    "fbio": "10.1016/j.asoc.2020.106339",
    "gska": "10.1007/s13042-019-01053-x",
    "lco":  "10.1007/s00500-019-04443-z",
    "saro": "10.1155/2019/2482543",
    "aft":  "10.1007/s00521-021-06392-x",
    "bro":  "10.1007/s00521-020-05004-4",
    "cddo": "10.1007/s13369-021-05928-6",
    "bco":  "10.1109/MCS.2002.1004010",
    "soa":  "10.1016/j.knosys.2018.11.024",
    "acor": "10.1007/s10732-008-9062-4",
    "bsa":  "10.1080/0952813X.2015.1042530",
    "epc":  "10.1016/j.knosys.2018.06.001",
    "gco":  "10.1002/int.21892",
    "sfoa_ref": "10.1016/j.swevo.2023.101262",
    "sspider_a":"10.1016/j.asoc.2015.02.014",
    "wca":  "10.1016/j.compstruc.2012.07.010",
    "bmo":  "10.1109/ICOICA.2019.8895393",
    "squirrel_sa":"10.1016/j.swevo.2018.02.013",
    "two":     "10.1016/j.procs.2020.03.063",
    "fox":  "10.1007/s10489-022-03533-0",
}


for _aid, _doi in _ALGORITHM_DOIS.items():
    if _aid in REGISTRY:
        _reference = dict(getattr(REGISTRY[_aid], "_REFERENCE", {}) or {})
        _reference["doi"] = _doi
        REGISTRY[_aid]._REFERENCE = _reference

from .aco          import ACOEngine
from .adam         import AdamEngine
from .aesspso      import AESSPSOEngine
from .autov        import AutoVEngine
from .bfgs         import BFGSEngine
from .bspga        import BSPGAEngine
from .cmaes        import CMAESEngine
from .cso          import CSOEngine
from .ecpo         import ECPOEngine
from .ego          import EGOEngine
from .fep          import FEPEngine
from .frcg         import FRCGEngine
from .frofi        import FROFIEngine
from .gpso         import GPSOEngine
from .imode        import IMODEEngine
from .kma          import KMAEngine
from .l2smea       import L2SMEAEngine
from .mfea         import MFEAEngine, MFEA2Engine
from .mgo          import MGOEngine
from .misaco       import MiSACOEngine
from .mvpa         import MVPAEngine
from .nndrea_so    import NNDREASOEngine, SACCEAMIIEngine, SSIORLEngine
from .ofa          import OFAEngine
from .rmsprop      import RMSPropEngine
from .sacoso       import SACOSOEngine
from .sade_amss    import SADEAMSSEngine
from .sade_atdsc   import SADEATDSCEngine
from .sade_sammon  import SADESammonEngine
from .samso        import SAMSOEngine, SAPOEngine
from .sd           import SDEngine
from .sqp          import SQPEngine

_NEW_ENGINES = [
    ACOEngine, AdamEngine, AESSPSOEngine, AutoVEngine, BFGSEngine, BSPGAEngine,
    CMAESEngine, CSOEngine, ECPOEngine, EGOEngine, FEPEngine, FRCGEngine, FROFIEngine,
    GPSOEngine, IMODEEngine, KMAEngine, L2SMEAEngine, MFEAEngine, MFEA2Engine,
    MGOEngine, MiSACOEngine, MVPAEngine, NNDREASOEngine, OFAEngine, RMSPropEngine,
    SACCEAMIIEngine, SACOSOEngine, SADEAMSSEngine, SADEATDSCEngine, SADESammonEngine,
    SAMSOEngine, SAPOEngine, SDEngine, SQPEngine, SSIORLEngine,
]

for _cls in _NEW_ENGINES:
    REGISTRY[_cls.algorithm_id] = _cls

# Injection-enabled new algorithms
_NEW_INJECTION = {
    "aesspso", "autov", "bspga", "cmaes", "cso", "ecpo", "ego", "fep", "frofi",
    "gpso", "imode", "kma", "l2smea", "mfea", "mfea2", "mgo", "misaco", "mvpa",
    "nndrea_so", "ofa", "sacoso", "sacc_eam2", "sade_amss", "sade_atdsc",
    "sade_sammon", "samso", "sapo", "ssio_rl",
}
_INJECTION_ENABLED.update(_NEW_INJECTION)
for _aid in _NEW_INJECTION:
    if _aid in REGISTRY:
        REGISTRY[_aid].capabilities.supports_candidate_injection = True

# DOIs for new algorithms
_NEW_DOIS = {
    "aesspso":     "10.1016/j.swevo.2025.101868",
    "cmaes":       "10.1162/106365602760972767",
    "cso":         "10.1109/TCYB.2014.2314537",
    "fep":         "10.1109/4235.771163",
    "frofi":       "10.1109/TCYB.2015.2493239",
    "kma":         "10.1016/j.asoc.2022.108043",
    "mgo":         "10.1016/j.advengsoft.2022.103282",
    "ofa":         "10.1016/j.asoc.2017.01.006",
    "ego":         "10.1023/A:1008306431147",
    "sacoso":      "10.1109/TEVC.2017.2674885",
    "samso":       "10.1109/TCYB.2019.2950169",
    "mfea":        "10.1109/TEVC.2015.2458037",
    "mfea2":       "10.1109/TEVC.2019.2904771",
    "gpso":        "10.1016/j.asoc.2011.10.007",
    "frcg":        "10.1093/comjnl/7.2.149",
    "bspga":       "10.1016/j.ins.2019.11.055",
    "l2smea":      "10.1109/TEVC.2024.3354543",
    "misaco":      "10.1109/TCYB.2020.3035521",
    "nndrea_so":   "10.1109/TEVC.2024.3378530",
    "sacc_eam2":   "10.1109/CEC.2019.8790061",
    "ssio_rl":     "10.1109/JAS.2025.125018",
    "autov":       "10.23919/CJE.2022.00.038",
    "sapo":        "10.1007/978-3-031-70085-9_22",
}
_ALGORITHM_DOIS.update(_NEW_DOIS)
for _aid, _doi in _NEW_DOIS.items():
    if _aid in REGISTRY:
        _ref = dict(getattr(REGISTRY[_aid], "_REFERENCE", {}) or {})
        _ref["doi"] = _doi
        REGISTRY[_aid]._REFERENCE = _ref
