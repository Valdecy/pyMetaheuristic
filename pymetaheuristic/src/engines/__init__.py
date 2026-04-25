"""pyMetaheuristic src â€” Engine registry."""

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)

from .abco               import ABCOEngine
from .acgwo              import ACGWOEngine
from .aco                import ACOEngine
from .acor               import ACOREngine
from .adam               import AdamEngine
from .aefa               import AEFAEngine
from .aeo                import AEOEngine
from .aesspso            import AESSPSOEngine
from .afsa               import AFSAEngine
from .aft                import AFTEngine
from .agto               import AGTOEngine
from .aha                import AHAEngine
from .ala                import ALAEngine
from .alo                import ALOEngine
from .ao                 import AOEngine
from .aoa                import AOAEngine
from .apo                import APOEngine
from .arch_oa            import ARCHOAEngine
from .aro                import AROEngine
from .ars                import ARSEngine
from .artemisinin_o      import ArtemisininOEngine
from .aso                import ASOEngine
from .autov              import AutoVEngine
from .avoa               import AVOAEngine
from .bat_a              import BATAEngine
from .bbo                import BBOEngine
from .bboa               import BBOAEngine
from .bco                import BCOEngine
from .bea                import BEAEngine
from .bes                import BESEngine
from .bfgs               import BFGSEngine
from .bfo                import BFOEngine
from .bka                import BKAEngine
from .bmo                import BMOEngine
from .bo_bonobo          import BonobOEngine
from .boa                import BOAEngine
from .bro                import BROEngine
from .bsa                import BSAEngine
from .bso                import BSOEngine
from .bspga              import BSPGAEngine
from .bwo                import BWOEngine
from .ca                 import CAEngine
from .camel              import CamelEngine
from .capsa              import CapSAEngine
from .cat_so             import CAT_SOEngine
from .cddo               import CDDOEngine
from .cdo                import CDOEngine
from .cem                import CEMEngine
from .cgo                import CGOEngine
from .chameleon_sa       import ChameleonSAEngine
from .chicken_so         import CHICKEN_SOEngine
from .chio               import CHIOEngine
from .choa               import ChOAEngine
from .circle_sa          import CIRCLESAEngine
from .clonalg            import CLONALGEngine
from .cmaes              import CMAESEngine
from .coa                import COAEngine
from .coati_oa           import COATI_OAEngine
from .cockroach_so       import COCKROACH_SOEngine
from .coot               import COOTEngine
from .crayfish_oa        import CrayfishOAEngine
from .cro                import CROEngine
from .csa                import CSAEngine
from .csbo               import CSBOEngine
from .cso                import CSOEngine
from .cuckoo_s           import CUCKOO_SEngine
from .da                 import DAEngine
from .dbo                import DBOEngine
from .ddao               import DDAOEngine
from .de                 import DEEngine
from .dfo                import DFOEngine
from .dmoa               import DMOAEngine
from .do_dandelion       import DandelionOEngine
from .doa                import DOAEngine
from .dso                import DSOEngine
from .dvba               import DVBAEngine
from .eao                import EAOEngine
from .eco                import ECOEngine
from .ecological_cycle_o import EcologicalCycleOEngine
from .ecpo               import ECPOEngine
from .edo                import EDOEngine
from .efo                import EFOEngine
from .ego                import EGOEngine
from .eho                import EHOEngine
from .elk_ho             import ElkHOEngine
from .eo                 import EOEngine
from .eoa                import EOAEngine
from .ep                 import EPEngine
from .epc                import EPCEngine
from .es                 import ESEngine
from .esc                import ESCEngine
from .eso                import ESOEngine
from .esoa               import ESOAEngine
from .eto                import ETOEngine
from .evo                import EVOEngine
from .fata               import FATAEngine
from .fbio               import FBIOEngine
from .fda                import FDAEngine
from .fdo                import FDOEngine
from .fep                import FEPEngine
from .ffa                import FFAEngine
from .ffo                import FFOEngine
from .firefly_a          import FIREFLY_AEngine
from .fla                import FLAEngine
from .flood_a            import FloodAEngine
from .foa                import FOAEngine
from .fox                import FOXEngine
from .fpa                import FPAEngine
from .frcg               import FRCGEngine
from .frofi              import FROFIEngine
from .fss                import FSSEngine
from .fwa                import FWAEngine
from .ga                 import GAEngine
from .gazelle_oa         import GazelleOAEngine
from .gbo                import GBOEngine
from .gco                import GCOEngine
from .gea                import GEAEngine
from .ggo                import GGOEngine
from .gjo                import GJOEngine
from .gkso               import GKSOEngine
from .gmo                import GMOEngine
from .gndo               import GNDOEngine
from .go_growth          import GOGrowthEngine
from .goa                import GOAEngine
from .gpso               import GPSOEngine
from .gsa                import GSAEngine
from .gska               import GSKAEngine
from .gso                import GSOEngine
from .gto                import GTOEngine
from .gwo                import GWOEngine
from .hba                import HBAEngine
from .hbo                import HBOEngine
from .hc                 import HCEngine
from .hco                import HCOEngine
from .hde                import HDEEngine
from .heoa               import HEOAEngine
from .hgs                import HGSEngine
from .hgso               import HGSOEngine
from .hho                import HHOEngine
from .hiking_oa          import HikingOAEngine
from .ho_hippo           import HippoEngine
from .horse_oa           import HorseOAEngine
from .hsa                import HSAEngine
from .hsaba              import HSABAEngine
from .hus                import HUSEngine
from .i_gwo              import I_GWOEngine
from .i_woa              import I_WOAEngine
from .ica                import ICAEngine
from .ilshade            import ILSHADEEngine
from .imode              import IMODEEngine
from .info               import INFOEngine
from .ivya               import IVYAEngine
from .iwo                import IWOEngine
from .jde                import JDEEngine
from .jso                import JSOEngine
from .jy                 import JYEngine
from .kha                import KHAEngine
from .kma                import KMAEngine
from .l2smea             import L2SMEAEngine
from .lca                import LCAEngine
from .lco                import LCOEngine
from .lfd                import LFDEngine
from .liwo               import LiWOEngine
from .loa                import LOAEngine
from .lpo                import LPOEngine
from .mbo                import MBOEngine
from .memetic_a          import MEMETIC_AEngine
from .mfa                import MFAEngine
from .mfea               import MFEAEngine
from .mfea2              import MFEA2Engine
from .mgo                import MGOEngine
from .misaco             import MiSACOEngine
from .mke                import MKEEngine
from .moss_go            import MossGOEngine
from .mpa                import MPAEngine
from .mrfo               import MRFOEngine
from .msa_e              import MSAEngine
from .mshoa              import MSHOAEngine
from .mso                import MSOEngine
from .mts                import MTSEngine
from .mvo                import MVOEngine
from .mvpa               import MVPAEngine
from .nca                import NCAEngine
from .ngo                import NGOEngine
from .nmm                import NMMEngine
from .nmra               import NMRAEngine
from .nndrea_so          import NNDREASOEngine
from .noa                import NOAEngine
from .nro                import NROEngine
from .ofa                import OFAEngine
from .ooa                import OOAEngine
from .parrot_o           import ParrotOEngine
from .pbil               import PBILEngine
from .pcx                import PCXEngine
from .pdo                import PDOEngine
from .pfa                import PFAEngine
from .pko                import PKOEngine
from .plba               import PLBAEngine
from .plo                import PLOEngine
from .poa                import POAEngine
from .political_o        import PoliticalOEngine
from .pro                import PROEngine
from .pso                import PSOEngine
from .pss                import PSSEngine
from .puma_o             import PumaOEngine
from .qio                import QIOEngine
from .qsa                import QSAEngine
from .random_s           import RANDOM_SEngine
from .rbmo               import RBMOEngine
from .rfo                import RFOEngine
from .rime               import RIMEEngine
from .rmsprop            import RMSPropEngine
from .roa                import ROAEngine
from .rsa                import RSAEngine
from .rso                import RSOEngine
from .run                import RUNEngine
from .sa                 import SAEngine
from .saba               import SABAEngine
from .sacc_eam2          import SACCEAMIIEngine
from .sacoso             import SACOSOEngine
from .sade_amss          import SADEAMSSEngine
from .sade_atdsc         import SADEATDSCEngine
from .sade_sammon        import SADESammonEngine
from .samso              import SAMSOEngine
from .sapo               import SAPOEngine
from .saro               import SAROEngine
from .sbo                import SBOEngine
from .sboa               import SBOAEngine
from .scho               import SCHOEngine
from .scso               import SCSoEngine
from .sd                 import SDEngine
from .seaho              import SeaHOEngine
from .serval_oa          import SERVALOAEngine
from .sfo                import SFOEngine
from .sfoa               import SFOAEngine
from .shade              import SHADEEngine
from .shio               import SHIOEngine
from .sho                import SHOEngine
from .sine_cosine_a      import SINE_COSINE_AEngine
from .slo                import SLOEngine
from .sma                import SMAEngine
from .smo                import SMOEngine
from .snow_oa            import SnowOAEngine
from .so_snake           import SnakeOptimizerEngine
from .soa                import SOAEngine
from .soo                import SOOEngine
from .sos                import SOSEngine
from .sparrow_sa         import SparrowSAEngine
from .spbo               import SPBOEngine
from .sqp                import SQPEngine
from .squirrel_sa        import SquirrelSAEngine
from .srsr               import SRSREngine
from .ssa                import SSAEngine
from .ssdo               import SSDOEngine
from .ssio_rl            import SSIORLEngine
from .sso                import SSOEngine
from .sspider_a          import SSPIDERAEngine
from .sto                import STOEngine
from .superb_foa         import SuperbFOAEngine
from .supply_do          import SupplyDOEngine
from .tdo                import TDOEngine
from .thro               import THROEngine
from .tlbo               import TLBOEngine
from .tlco               import TLCOEngine
from .toa                import TOAEngine
from .toc                import TOCEngine
from .tpo                import TPOEngine
from .ts                 import TSEngine
from .tsa                import TSAEngine
from .tso                import TSOEngine
from .ttao               import TTAOEngine
from .two                import TWOEngine
from .vcs                import VCSEngine
from .waoa               import WAOAEngine
from .warso              import WARSOEngine
from .wca                import WCAEngine
from .wdo                import WDOEngine
from .who                import WHOEngine
from .wo_wave            import WaveOptEngine
from .woa                import WOAEngine
from .wso                import WSOEngine
from .wutp               import WUTPEngine
from .ydse               import YDSEEngine
from .zoa                import ZOAEngine


_ENGINE_CLASSES: tuple[type[BaseEngine], ...] = (
    ABCOEngine, ACGWOEngine, ACOEngine, ACOREngine, AdamEngine, AEFAEngine, AEOEngine,
    AESSPSOEngine, AFSAEngine, AFTEngine, AGTOEngine, AHAEngine, ALAEngine, ALOEngine,
    AOEngine, AOAEngine, APOEngine, ARCHOAEngine, AROEngine, ARSEngine,
    ArtemisininOEngine, ASOEngine, AutoVEngine, AVOAEngine, BATAEngine, BBOEngine,
    BBOAEngine, BCOEngine, BEAEngine, BESEngine, BFGSEngine, BFOEngine, BKAEngine,
    BMOEngine, BonobOEngine, BOAEngine, BROEngine, BSAEngine, BSOEngine, BSPGAEngine,
    BWOEngine, CAEngine, CamelEngine, CapSAEngine, CAT_SOEngine, CDDOEngine, CDOEngine,
    CEMEngine, CGOEngine, ChameleonSAEngine, CHICKEN_SOEngine, CHIOEngine, ChOAEngine,
    CIRCLESAEngine, CLONALGEngine, CMAESEngine, COAEngine, COATI_OAEngine,
    COCKROACH_SOEngine, COOTEngine, CrayfishOAEngine, CROEngine, CSAEngine, CSBOEngine,
    CSOEngine, CUCKOO_SEngine, DAEngine, DBOEngine, DDAOEngine, DEEngine, DFOEngine,
    DMOAEngine, DandelionOEngine, DOAEngine, DSOEngine, DVBAEngine, EAOEngine,
    ECOEngine, EcologicalCycleOEngine, ECPOEngine, EDOEngine, EFOEngine, EGOEngine,
    EHOEngine, ElkHOEngine, EOEngine, EOAEngine, EPEngine, EPCEngine, ESEngine,
    ESCEngine, ESOEngine, ESOAEngine, ETOEngine, EVOEngine, FATAEngine, FBIOEngine,
    FDAEngine, FDOEngine, FEPEngine, FFAEngine, FFOEngine, FIREFLY_AEngine, FLAEngine,
    FloodAEngine, FOAEngine, FOXEngine, FPAEngine, FRCGEngine, FROFIEngine, FSSEngine,
    FWAEngine, GAEngine, GazelleOAEngine, GBOEngine, GCOEngine, GEAEngine, GGOEngine,
    GJOEngine, GKSOEngine, GMOEngine, GNDOEngine, GOGrowthEngine, GOAEngine,
    GPSOEngine, GSAEngine, GSKAEngine, GSOEngine, GTOEngine, GWOEngine, HBAEngine,
    HBOEngine, HCEngine, HCOEngine, HDEEngine, HEOAEngine, HGSEngine, HGSOEngine,
    HHOEngine, HikingOAEngine, HippoEngine, HorseOAEngine, HSAEngine, HSABAEngine,
    HUSEngine, I_GWOEngine, I_WOAEngine, ICAEngine, ILSHADEEngine, IMODEEngine,
    INFOEngine, IVYAEngine, IWOEngine, JDEEngine, JSOEngine, JYEngine, KHAEngine,
    KMAEngine, L2SMEAEngine, LCAEngine, LCOEngine, LFDEngine, LiWOEngine, LOAEngine,
    LPOEngine, MBOEngine, MEMETIC_AEngine, MFAEngine, MFEAEngine, MFEA2Engine,
    MGOEngine, MiSACOEngine, MKEEngine, MossGOEngine, MPAEngine, MRFOEngine, MSAEngine,
    MSHOAEngine, MSOEngine, MTSEngine, MVOEngine, MVPAEngine, NCAEngine, NGOEngine,
    NMMEngine, NMRAEngine, NNDREASOEngine, NOAEngine, NROEngine, OFAEngine, OOAEngine,
    ParrotOEngine, PBILEngine, PCXEngine, PDOEngine, PFAEngine, PKOEngine, PLBAEngine,
    PLOEngine, POAEngine, PoliticalOEngine, PROEngine, PSOEngine, PSSEngine,
    PumaOEngine, QIOEngine, QSAEngine, RANDOM_SEngine, RBMOEngine, RFOEngine,
    RIMEEngine, RMSPropEngine, ROAEngine, RSAEngine, RSOEngine, RUNEngine, SAEngine,
    SABAEngine, SACCEAMIIEngine, SACOSOEngine, SADEAMSSEngine, SADEATDSCEngine,
    SADESammonEngine, SAMSOEngine, SAPOEngine, SAROEngine, SBOEngine, SBOAEngine,
    SCHOEngine, SCSoEngine, SDEngine, SeaHOEngine, SERVALOAEngine, SFOEngine,
    SFOAEngine, SHADEEngine, SHIOEngine, SHOEngine, SINE_COSINE_AEngine, SLOEngine,
    SMAEngine, SMOEngine, SnowOAEngine, SnakeOptimizerEngine, SOAEngine, SOOEngine,
    SOSEngine, SparrowSAEngine, SPBOEngine, SQPEngine, SquirrelSAEngine, SRSREngine,
    SSAEngine, SSDOEngine, SSIORLEngine, SSOEngine, SSPIDERAEngine, STOEngine,
    SuperbFOAEngine, SupplyDOEngine, TDOEngine, THROEngine, TLBOEngine, TLCOEngine,
    TOAEngine, TOCEngine, TPOEngine, TSEngine, TSAEngine, TSOEngine, TTAOEngine,
    TWOEngine, VCSEngine, WAOAEngine, WARSOEngine, WCAEngine, WDOEngine, WHOEngine,
    WaveOptEngine, WOAEngine, WSOEngine, WUTPEngine, YDSEEngine, ZOAEngine,
)

REGISTRY: dict[str, type[BaseEngine]] = {
    cls.algorithm_id: cls
    for cls in _ENGINE_CLASSES
}

__all__ = [
    "REGISTRY",
    "BaseEngine",
    "ProblemSpec",
    "EngineConfig",
    "CapabilityProfile",
    "CandidateRecord",
    "EngineState",
    "OptimizationResult",
]


# Enable native candidate injection for algorithms whose state can safely absorb
# migrants through either generic population replacement or engine-specific repair.
_INJECTION_ENABLED: set[str] = {
    'abco', 'acgwo', 'aesspso', 'afsa', 'aft', 'agto', 'alo', 'ao', 'aoa', 'arch_oa',
    'aro', 'ars', 'aso', 'autov', 'avoa', 'bat_a', 'bbo', 'bboa', 'bco', 'bea', 'bes',
    'bfo', 'bmo', 'bro', 'bso', 'bspga', 'bwo', 'ca', 'camel', 'cat_so', 'cddo', 'cem',
    'cgo', 'chicken_so', 'chio', 'circle_sa', 'clonalg', 'cmaes', 'coa', 'coati_oa',
    'cockroach_so', 'crayfish_oa', 'cro', 'csa', 'cso', 'cuckoo_s', 'da', 'de', 'dfo',
    'dmoa', 'doa', 'dvba', 'eao', 'ecological_cycle_o', 'ecpo', 'efo', 'ego', 'eho',
    'eo', 'eoa', 'ep', 'es', 'eso', 'fbio', 'fda', 'fep', 'firefly_a', 'fla', 'foa',
    'fox', 'fpa', 'frofi', 'fss', 'fwa', 'ga', 'gbo', 'gjo', 'gmo', 'goa', 'gpso',
    'gsa', 'gska', 'gso', 'gto', 'gwo', 'hba', 'hbo', 'hco', 'hde', 'hgs', 'hgso',
    'hho', 'hsaba', 'hus', 'i_gwo', 'i_woa', 'ica', 'ilshade', 'imode', 'info', 'iwo',
    'jde', 'jso', 'jy', 'kha', 'kma', 'l2smea', 'lco', 'liwo', 'loa', 'mbo',
    'memetic_a', 'mfa', 'mfea', 'mfea2', 'mgo', 'misaco', 'mke', 'mpa', 'mrfo',
    'msa_e', 'mshoa', 'mts', 'mvo', 'mvpa', 'nca', 'ngo', 'nmm', 'nmra', 'nndrea_so',
    'noa', 'nro', 'ofa', 'pbil', 'pcx', 'pfa', 'plba', 'poa', 'pso', 'qsa', 'random_s',
    'rime', 'run', 'sa', 'saba', 'sacc_eam2', 'sacoso', 'sade_amss', 'sade_atdsc',
    'sade_sammon', 'samso', 'sapo', 'saro', 'sbo', 'scso', 'seaho', 'sfo', 'shade',
    'sho', 'sine_cosine_a', 'slo', 'sma', 'smo', 'soa', 'soo', 'sos', 'spbo',
    'squirrel_sa', 'srsr', 'ssa', 'ssdo', 'ssio_rl', 'tlbo', 'toa', 'tsa', 'tso',
    'two', 'vcs', 'warso', 'wdo', 'who', 'woa', 'zoa',
}

for _aid in _INJECTION_ENABLED:
    if _aid in REGISTRY:
        REGISTRY[_aid].capabilities.supports_candidate_injection = True


# Curated DOI metadata derived from the user-supplied algorithm table.
# Only canonical DOI strings are retained here. Non-DOI links are intentionally
# omitted so BaseEngine.info() reports only DOI values.
_ALGORITHM_DOIS: dict[str, str] = {
    'acgwo'           : '10.1007/s42835-023-01621-w',
    'acor'            : '10.1007/s10732-008-9062-4',
    'aesspso'         : '10.1016/j.swevo.2025.101868',
    'aft'             : '10.1007/s00521-021-06392-x',
    'agto'            : '10.1002/int.22535',
    'alo'             : '10.1016/j.advengsoft.2015.01.010',
    'ao'              : '10.1016/j.cie.2021.107250',
    'aoa'             : '10.1016/j.cma.2020.113609',
    'arch_oa'         : '10.1007/s10489-020-01893-z',
    'aro'             : '10.1016/j.engappai.2022.105082',
    'autov'           : '10.23919/CJE.2022.00.038',
    'bbo'             : '10.1109/TEVC.2008.919004',
    'bco'             : '10.1109/MCS.2002.1004010',
    'bes'             : '10.1007/s10462-019-09732-5',
    'bfo'             : '10.1109/MCS.2002.1004010',
    'bmo'             : '10.1109/ICOICA.2019.8895393',
    'bro'             : '10.1007/s00521-020-05004-4',
    'bsa'             : '10.1080/0952813X.2015.1042530',
    'bso'             : '10.1007/978-3-642-21515-5_36',
    'bspga'           : '10.1016/j.ins.2019.11.055',
    'bwo'             : '10.1016/j.engappai.2019.103249',
    'ca'              : '10.1142/9789814534116',
    'cat_so'          : '10.1007/978-3-540-36668-3_94',
    'cddo'            : '10.1007/s13369-021-05928-6',
    'cem'             : '10.1016/S0377-2217(96)00385-2',
    'cgo'             : '10.1007/s10462-020-09867-w',
    'chicken_so'      : '10.1007/978-3-319-11857-4_10',
    'chio'            : '10.1007/s00521-020-05296-6',
    'cmaes'           : '10.1162/106365602760972767',
    'coati_oa'        : '10.1016/j.knosys.2022.110011',
    'cockroach_so'    : '10.1109/ICCET.2010.5485993',
    'cro'             : '10.1155/2014/739768',
    'csa'             : '10.1016/j.compstruc.2016.03.001',
    'cso'             : '10.1109/TCYB.2014.2314537',
    'da'              : '10.1007/s00521-015-1920-1',
    'de'              : '10.1023/A:1008202821328',
    'dfo'             : '10.15439/2014F142',
    'dmoa'            : '10.1016/j.cma.2022.114570',
    'dvba'            : '10.1109/INCoS.2014.40',
    'ego'             : '10.1023/A:1008306431147',
    'eho'             : '10.1109/ISCBI.2015.8',
    'eo'              : '10.1016/j.knosys.2019.105190',
    'eoa'             : '10.1504/IJBIC.2015.10004283',
    'epc'             : '10.1016/j.knosys.2018.06.001',
    'eso'             : '10.3390/make7010024',
    'fbio'            : '10.1016/j.asoc.2020.106339',
    'fda'             : '10.1016/j.cie.2021.107224',
    'fep'             : '10.1109/4235.771163',
    'foa'             : '10.1016/j.eswa.2014.05.009',
    'fox'             : '10.1007/s10489-022-03533-0',
    'frcg'            : '10.1093/comjnl/7.2.149',
    'frofi'           : '10.1109/TCYB.2015.2493239',
    'fwa'             : '10.1016/j.asoc.2017.10.046',
    'gco'             : '10.1002/int.21892',
    'gmo'             : '10.1007/s00500-023-08202-z',
    'goa'             : '10.1016/j.advengsoft.2017.01.004',
    'gpso'            : '10.1016/j.asoc.2011.10.007',
    'gsa'             : '10.1016/j.ins.2009.03.004',
    'gska'            : '10.1007/s13042-019-01053-x',
    'gwo'             : '10.1016/j.advengsoft.2013.12.007',
    'hho'             : '10.1016/j.future.2019.02.028',
    'hsa'             : '10.1177/003754970107600201',
    'hus'             : '10.1109/ICSCCW.2009.5379451',
    'i_gwo'           : '10.1016/j.eswa.2020.113917',
    'i_woa'           : '10.1016/j.jcde.2019.02.002',
    'info'            : '10.1016/j.eswa.2022.116516',
    'jso'             : '10.1016/j.amc.2020.125535',
    'kha'             : '10.1016/j.asoc.2016.08.041',
    'kma'             : '10.1016/j.asoc.2022.108043',
    'l2smea'          : '10.1109/TEVC.2024.3354543',
    'lco'             : '10.1007/s00500-019-04443-z',
    'loa'             : '10.1016/j.jcde.2015.06.003',
    'mbo'             : '10.1007/s00521-015-1923-y',
    'mfa'             : '10.1016/j.knosys.2015.07.006',
    'mfea'            : '10.1109/TEVC.2015.2458037',
    'mfea2'           : '10.1109/TEVC.2019.2904771',
    'mgo'             : '10.1016/j.advengsoft.2022.103282',
    'misaco'          : '10.1109/TCYB.2020.3035521',
    'mke'             : '10.1016/j.knosys.2016.01.009',
    'mrfo'            : '10.1016/j.engappai.2019.103300',
    'msa_e'           : '10.1007/s12293-016-0212-3',
    'mshoa'           : '10.3390/math13091500',
    'mvo'             : '10.1007/s00521-015-1870-7',
    'nndrea_so'       : '10.1109/TEVC.2024.3378530',
    'ofa'             : '10.1016/j.asoc.2017.01.006',
    'pcx'             : '10.1162/106365602760972767',
    'pfa'             : '10.1016/j.asoc.2019.03.012',
    'pso'             : '10.1109/ICNN.1995.488968',
    'qsa'             : '10.1007/s12652-020-02849-4',
    'random_s'        : '10.1080/01621459.1953.10501200',
    'rime'            : '10.1016/j.neucom.2023.02.010',
    'run'             : '10.1016/j.eswa.2021.115079',
    'sacc_eam2'       : '10.1109/CEC.2019.8790061',
    'sacoso'          : '10.1109/TEVC.2017.2674885',
    'samso'           : '10.1109/TCYB.2019.2950169',
    'sapo'            : '10.1007/978-3-031-70085-9_22',
    'saro'            : '10.1155/2019/2482543',
    'sbo'             : '10.1016/j.engappai.2017.01.006',
    'scso'            : '10.1007/s00366-022-01604-x',
    'seaho'           : '10.1007/s10489-022-03994-3',
    'sfo'             : '10.1016/j.engappai.2019.01.001',
    'sfoa_ref'        : '10.1016/j.swevo.2023.101262',
    'sho'             : '10.1016/j.advengsoft.2017.05.014',
    'sine_cosine_a'   : '10.1016/j.knosys.2015.12.022',
    'slo'             : '10.14569/IJACSA.2019.0100548',
    'sma'             : '10.1016/j.future.2020.03.055',
    'smo'             : '10.1007/s12293-013-0128-0',
    'soa'             : '10.1016/j.knosys.2018.11.024',
    'sos'             : '10.1016/j.compstruc.2014.03.007',
    'spbo'            : '10.1016/j.advengsoft.2020.102804',
    'squirrel_sa'     : '10.1016/j.swevo.2018.02.013',
    'srsr'            : '10.1016/j.asoc.2017.02.028',
    'ssa'             : '10.1016/j.advengsoft.2017.07.002',
    'ssdo'            : '10.1007/s00521-019-04159-z',
    'ssio_rl'         : '10.1109/JAS.2025.125018',
    'sspider_a'       : '10.1016/j.asoc.2015.02.014',
    'tlbo'            : '10.1016/j.compstruc.2014.03.007',
    'two'             : '10.1016/j.procs.2020.03.063',
    'vcs'             : '10.1016/j.advengsoft.2015.11.004',
    'wca'             : '10.1016/j.compstruc.2012.07.010',
    'who'             : '10.3233/JIFS-190495',
    'woa'             : '10.1016/j.advengsoft.2016.01.008',
}

for _aid, _doi in _ALGORITHM_DOIS.items():
    if _aid in REGISTRY:
        _reference = dict(getattr(REGISTRY[_aid], "_REFERENCE", {}) or {})
        _reference["doi"] = _doi
        REGISTRY[_aid]._REFERENCE = _reference
