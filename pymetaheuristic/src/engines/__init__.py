"""pyMetaheuristic src Engine registry.

This module is generated from the curated algorithm table. It centralizes
engine imports, registry construction, table-derived capability flags, and
canonical DOI metadata.
"""

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)

from .adam               import AdamEngine
from .acgwo              import ACGWOEngine
from .aesspso            import AESSPSOEngine
from .ars                import ARSEngine
from .aft                import AFTEngine
from .avoa               import AVOAEngine
from .aso                import ASOEngine
from .aco                import ACOEngine
from .acor               import ACOREngine
from .alo                import ALOEngine
from .ao                 import AOEngine
from .arch_oa            import ARCHOAEngine
from .aoa                import AOAEngine
from .artemisinin_o      import ArtemisininOEngine
from .abco               import ABCOEngine
from .aeo                import AEOEngine
from .aefa               import AEFAEngine
from .afsa               import AFSAEngine
from .agto               import AGTOEngine
from .aha                import AHAEngine
from .ala                import ALAEngine
from .apo                import APOEngine
from .aro                import AROEngine
from .autov              import AutoVEngine
from .bco                import BCOEngine
from .bfo                import BFOEngine
from .bes                import BESEngine
from .bmo                import BMOEngine
from .bat_a              import BATAEngine
from .bro                import BROEngine
from .bea                import BEAEngine
from .bfgs               import BFGSEngine
from .bspga              import BSPGAEngine
from .bbo                import BBOEngine
from .bsa                import BSAEngine
from .bwo                import BWOEngine
from .bka                import BKAEngine
from .bo_bonobo          import BonobOEngine
from .bso                import BSOEngine
from .bboa               import BBOAEngine
from .boa                import BOAEngine
from .camel              import CamelEngine
from .capsa              import CapSAEngine
from .cat_so             import CAT_SOEngine
from .chameleon_sa       import ChameleonSAEngine
from .cgo                import CGOEngine
from .cddo               import CDDOEngine
from .cdo                import CDOEngine
from .chicken_so         import CHICKEN_SOEngine
from .choa               import ChOAEngine
from .circle_sa          import CIRCLESAEngine
from .csbo               import CSBOEngine
from .clonalg            import CLONALGEngine
from .coati_oa           import COATI_OAEngine
from .cockroach_so       import COCKROACH_SOEngine
from .cso                import CSOEngine
from .coot               import COOTEngine
from .cro                import CROEngine
from .chio               import CHIOEngine
from .cmaes              import CMAESEngine
from .coa                import COAEngine
from .crayfish_oa        import CrayfishOAEngine
from .cem                import CEMEngine
from .csa                import CSAEngine
from .cuckoo_s           import CUCKOO_SEngine
from .ca                 import CAEngine
from .do_dandelion       import DandelionOEngine
from .dso                import DSOEngine
from .doa                import DOAEngine
from .de                 import DEEngine
from .hde                import HDEEngine
from .dfo                import DFOEngine
from .da                 import DAEngine
from .dbo                import DBOEngine
from .dmoa               import DMOAEngine
from .ddao               import DDAOEngine
from .dvba               import DVBAEngine
from .eoa                import EOAEngine
from .ecological_cycle_o import EcologicalCycleOEngine
from .eco                import ECOEngine
from .ego                import EGOEngine
from .esoa               import ESOAEngine
from .ecpo               import ECPOEngine
from .eso                import ESOEngine
from .efo                import EFOEngine
from .eho                import EHOEngine
from .elk_ho             import ElkHOEngine
from .epc                import EPCEngine
from .evo                import EVOEngine
from .eao                import EAOEngine
from .eo                 import EOEngine
from .esc                import ESCEngine
from .es                 import ESEngine
from .ep                 import EPEngine
from .edo                import EDOEngine
from .eto                import ETOEngine
from .fep                import FEPEngine
from .fata               import FATAEngine
from .frofi              import FROFIEngine
from .ffo                import FFOEngine
from .fla                import FLAEngine
from .firefly_a          import FIREFLY_AEngine
from .fwa                import FWAEngine
from .fss                import FSSEngine
from .frcg               import FRCGEngine
from .flood_a            import FloodAEngine
from .fda                import FDAEngine
from .fpa                import FPAEngine
from .fdo                import FDOEngine
from .fbio               import FBIOEngine
from .foa                import FOAEngine
from .fox                import FOXEngine
from .ffa                import FFAEngine
from .gska               import GSKAEngine
from .gazelle_oa         import GazelleOAEngine
from .gndo               import GNDOEngine
from .ga                 import GAEngine
from .gkso               import GKSOEngine
from .gmo                import GMOEngine
from .gco                import GCOEngine
from .gea                import GEAEngine
from .gto                import GTOEngine
from .gso                import GSOEngine
from .gjo                import GJOEngine
from .gbo                import GBOEngine
from .gpso               import GPSOEngine
from .goa                import GOAEngine
from .gsa                import GSAEngine
from .gwo                import GWOEngine
from .ggo                import GGOEngine
from .go_growth          import GOGrowthEngine
from .hsa                import HSAEngine
from .hho                import HHOEngine
from .hbo                import HBOEngine
from .hgso               import HGSOEngine
from .hiking_oa          import HikingOAEngine
from .hc                 import HCEngine
from .ho_hippo           import HippoEngine
from .horse_oa           import HorseOAEngine
from .hco                import HCOEngine
from .heoa               import HEOAEngine
from .hgs                import HGSEngine
from .hus                import HUSEngine
from .hba                import HBAEngine
from .hsaba              import HSABAEngine
from .ica                import ICAEngine
from .i_gwo              import I_GWOEngine
from .ilshade            import ILSHADEEngine
from .imode              import IMODEEngine
from .i_woa              import I_WOAEngine
from .iwo                import IWOEngine
from .ivya               import IVYAEngine
from .jy                 import JYEngine
from .jso                import JSOEngine
from .kma                import KMAEngine
from .kha                import KHAEngine
from .liwo               import LiWOEngine
from .lco                import LCOEngine
from .l2smea             import L2SMEAEngine
from .loa                import LOAEngine
from .lca                import LCAEngine
from .lpo                import LPOEngine
from .lfd                import LFDEngine
from .mrfo               import MRFOEngine
from .mshoa              import MSHOAEngine
from .mpa                import MPAEngine
from .memetic_a          import MEMETIC_AEngine
from .mso                import MSOEngine
from .mbo                import MBOEngine
from .mke                import MKEEngine
from .moss_go            import MossGOEngine
from .mvpa               import MVPAEngine
from .mfa                import MFAEngine
from .msa_e              import MSAEngine
from .mgo                import MGOEngine
from .misaco             import MiSACOEngine
from .mvo                import MVOEngine
from .mfea               import MFEAEngine
from .mfea2              import MFEA2Engine
from .mts                import MTSEngine
from .samso              import SAMSOEngine
from .nmra               import NMRAEngine
from .nmm                import NMMEngine
from .nndrea_so          import NNDREASOEngine
from .noa                import NOAEngine
from .ngo                import NGOEngine
from .nro                import NROEngine
from .nca                import NCAEngine
from .ofa                import OFAEngine
from .ooa                import OOAEngine
from .plba               import PLBAEngine
from .pcx                import PCXEngine
from .parrot_o           import ParrotOEngine
from .pso                import PSOEngine
from .pfa                import PFAEngine
from .poa                import POAEngine
from .pko                import PKOEngine
from .plo                import PLOEngine
from .political_o        import PoliticalOEngine
from .pro                import PROEngine
from .pbil               import PBILEngine
from .pdo                import PDOEngine
from .pss                import PSSEngine
from .puma_o             import PumaOEngine
from .qio                import QIOEngine
from .qsa                import QSAEngine
from .random_s           import RANDOM_SEngine
from .rso                import RSOEngine
from .rbmo               import RBMOEngine
from .roa                import ROAEngine
from .rsa                import RSAEngine
from .rime               import RIMEEngine
from .rmsprop            import RMSPropEngine
from .run                import RUNEngine
from .rfo                import RFOEngine
from .sfo                import SFOEngine
from .ssa                import SSAEngine
from .sade_sammon        import SADESammonEngine
from .scso               import SCSoEngine
from .sbo                import SBOEngine
from .slo                import SLOEngine
from .soa                import SOAEngine
from .seaho              import SeaHOEngine
from .saro               import SAROEngine
from .ssio_rl            import SSIORLEngine
from .sboa               import SBOAEngine
from .saba               import SABAEngine
from .jde                import JDEEngine
from .sqp                import SQPEngine
from .serval_oa          import SERVALOAEngine
from .srsr               import SRSREngine
from .sto                import STOEngine
from .sa                 import SAEngine
from .sine_cosine_a      import SINE_COSINE_AEngine
from .scho               import SCHOEngine
from .sma                import SMAEngine
from .so_snake           import SnakeOptimizerEngine
from .snow_oa            import SnowOAEngine
from .ssdo               import SSDOEngine
from .sspider_a          import SSPIDERAEngine
from .sso                import SSOEngine
from .sparrow_sa         import SparrowSAEngine
from .smo                import SMOEngine
from .shio               import SHIOEngine
from .sho                import SHOEngine
from .squirrel_sa        import SquirrelSAEngine
from .soo                import SOOEngine
from .sfoa               import SFOAEngine
from .sd                 import SDEngine
from .spbo               import SPBOEngine
from .shade              import SHADEEngine
from .superb_foa         import SuperbFOAEngine
from .supply_do          import SupplyDOEngine
from .sacc_eam2          import SACCEAMIIEngine
from .sacoso             import SACOSOEngine
from .sade_amss          import SADEAMSSEngine
from .sade_atdsc         import SADEATDSCEngine
from .sapo               import SAPOEngine
from .sos                import SOSEngine
from .ts                 import TSEngine
from .tdo                import TDOEngine
from .tlbo               import TLBOEngine
from .toa                import TOAEngine
from .tlco               import TLCOEngine
from .thro               import THROEngine
from .toc                import TOCEngine
from .tpo                import TPOEngine
from .ttao               import TTAOEngine
from .two                import TWOEngine
from .tso                import TSOEngine
from .tsa                import TSAEngine
from .vcs                import VCSEngine
from .waoa               import WAOAEngine
from .warso              import WARSOEngine
from .wca                import WCAEngine
from .wutp               import WUTPEngine
from .wo_wave            import WaveOptEngine
from .info               import INFOEngine
from .woa                import WOAEngine
from .wso                import WSOEngine
from .who                import WHOEngine
from .wdo                import WDOEngine
from .ydse               import YDSEEngine
from .zoa                import ZOAEngine


_ENGINE_CLASSES: tuple[type[BaseEngine], ...] = (
    AdamEngine, ACGWOEngine, AESSPSOEngine, ARSEngine, AFTEngine, AVOAEngine, ASOEngine,
    ACOEngine, ACOREngine, ALOEngine, AOEngine, ARCHOAEngine, AOAEngine, ArtemisininOEngine,
    ABCOEngine, AEOEngine, AEFAEngine, AFSAEngine, AGTOEngine, AHAEngine, ALAEngine, APOEngine,
    AROEngine, AutoVEngine, BCOEngine, BFOEngine, BESEngine, BMOEngine, BATAEngine, BROEngine,
    BEAEngine, BFGSEngine, BSPGAEngine, BBOEngine, BSAEngine, BWOEngine, BKAEngine,
    BonobOEngine, BSOEngine, BBOAEngine, BOAEngine, CamelEngine, CapSAEngine, CAT_SOEngine,
    ChameleonSAEngine, CGOEngine, CDDOEngine, CDOEngine, CHICKEN_SOEngine, ChOAEngine,
    CIRCLESAEngine, CSBOEngine, CLONALGEngine, COATI_OAEngine, COCKROACH_SOEngine, CSOEngine,
    COOTEngine, CROEngine, CHIOEngine, CMAESEngine, COAEngine, CrayfishOAEngine, CEMEngine,
    CSAEngine, CUCKOO_SEngine, CAEngine, DandelionOEngine, DSOEngine, DOAEngine, DEEngine,
    HDEEngine, DFOEngine, DAEngine, DBOEngine, DMOAEngine, DDAOEngine, DVBAEngine, EOAEngine,
    EcologicalCycleOEngine, ECOEngine, EGOEngine, ESOAEngine, ECPOEngine, ESOEngine, EFOEngine,
    EHOEngine, ElkHOEngine, EPCEngine, EVOEngine, EAOEngine, EOEngine, ESCEngine, ESEngine,
    EPEngine, EDOEngine, ETOEngine, FEPEngine, FATAEngine, FROFIEngine, FFOEngine, FLAEngine,
    FIREFLY_AEngine, FWAEngine, FSSEngine, FRCGEngine, FloodAEngine, FDAEngine, FPAEngine,
    FDOEngine, FBIOEngine, FOAEngine, FOXEngine, FFAEngine, GSKAEngine, GazelleOAEngine,
    GNDOEngine, GAEngine, GKSOEngine, GMOEngine, GCOEngine, GEAEngine, GTOEngine, GSOEngine,
    GJOEngine, GBOEngine, GPSOEngine, GOAEngine, GSAEngine, GWOEngine, GGOEngine,
    GOGrowthEngine, HSAEngine, HHOEngine, HBOEngine, HGSOEngine, HikingOAEngine, HCEngine,
    HippoEngine, HorseOAEngine, HCOEngine, HEOAEngine, HGSEngine, HUSEngine, HBAEngine,
    HSABAEngine, ICAEngine, I_GWOEngine, ILSHADEEngine, IMODEEngine, I_WOAEngine, IWOEngine,
    IVYAEngine, JYEngine, JSOEngine, KMAEngine, KHAEngine, LiWOEngine, LCOEngine, L2SMEAEngine,
    LOAEngine, LCAEngine, LPOEngine, LFDEngine, MRFOEngine, MSHOAEngine, MPAEngine,
    MEMETIC_AEngine, MSOEngine, MBOEngine, MKEEngine, MossGOEngine, MVPAEngine, MFAEngine,
    MSAEngine, MGOEngine, MiSACOEngine, MVOEngine, MFEAEngine, MFEA2Engine, MTSEngine,
    SAMSOEngine, NMRAEngine, NMMEngine, NNDREASOEngine, NOAEngine, NGOEngine, NROEngine,
    NCAEngine, OFAEngine, OOAEngine, PLBAEngine, PCXEngine, ParrotOEngine, PSOEngine,
    PFAEngine, POAEngine, PKOEngine, PLOEngine, PoliticalOEngine, PROEngine, PBILEngine,
    PDOEngine, PSSEngine, PumaOEngine, QIOEngine, QSAEngine, RANDOM_SEngine, RSOEngine,
    RBMOEngine, ROAEngine, RSAEngine, RIMEEngine, RMSPropEngine, RUNEngine, RFOEngine,
    SFOEngine, SSAEngine, SADESammonEngine, SCSoEngine, SBOEngine, SLOEngine, SOAEngine,
    SeaHOEngine, SAROEngine, SSIORLEngine, SBOAEngine, SABAEngine, JDEEngine, SQPEngine,
    SERVALOAEngine, SRSREngine, STOEngine, SAEngine, SINE_COSINE_AEngine, SCHOEngine,
    SMAEngine, SnakeOptimizerEngine, SnowOAEngine, SSDOEngine, SSPIDERAEngine, SSOEngine,
    SparrowSAEngine, SMOEngine, SHIOEngine, SHOEngine, SquirrelSAEngine, SOOEngine, SFOAEngine,
    SDEngine, SPBOEngine, SHADEEngine, SuperbFOAEngine, SupplyDOEngine, SACCEAMIIEngine,
    SACOSOEngine, SADEAMSSEngine, SADEATDSCEngine, SAPOEngine, SOSEngine, TSEngine, TDOEngine,
    TLBOEngine, TOAEngine, TLCOEngine, THROEngine, TOCEngine, TPOEngine, TTAOEngine, TWOEngine,
    TSOEngine, TSAEngine, VCSEngine, WAOAEngine, WARSOEngine, WCAEngine, WUTPEngine,
    WaveOptEngine, INFOEngine, WOAEngine, WSOEngine, WHOEngine, WDOEngine, YDSEEngine,
    ZOAEngine,
)

REGISTRY: dict[str, type[BaseEngine]] = {
    cls.algorithm_id: cls
    for cls in _ENGINE_CLASSES
}

# Compatibility alias for the historical Bonobo Optimizer module name.
# The public algorithm ID in the table is "bono".
_REGISTRY_ALIASES: dict[str, str] = {
    "bo_bonobo": "bono",
}

for _old_id, _new_id in _REGISTRY_ALIASES.items():
    if _new_id not in REGISTRY and _old_id in REGISTRY:
        REGISTRY[_new_id] = REGISTRY[_old_id]


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


# Table-derived algorithm IDs.
_TABLE_ALGORITHM_IDS: set[str] = {
    'adam', 'acgwo', 'aesspso', 'ars', 'aft', 'avoa', 'aso', 'aco', 'acor', 'alo', 'ao',
    'arch_oa', 'aoa', 'artemisinin_o', 'abco', 'aeo', 'aefa', 'afsa', 'agto', 'aha', 'ala',
    'apo', 'aro', 'autov', 'bco', 'bfo', 'bes', 'bmo', 'bat_a', 'bro', 'bea', 'bfgs', 'bspga',
    'bbo', 'bsa', 'bwo', 'bka', 'bono', 'bso', 'bboa', 'boa', 'camel', 'capsa', 'cat_so',
    'chameleon_sa', 'cgo', 'cddo', 'cdo', 'chicken_so', 'choa', 'circle_sa', 'csbo', 'clonalg',
    'coati_oa', 'cockroach_so', 'cso', 'coot', 'cro', 'chio', 'cmaes', 'coa', 'crayfish_oa',
    'cem', 'csa', 'cuckoo_s', 'ca', 'do_dandelion', 'dso', 'doa', 'de', 'hde', 'dfo', 'da',
    'dbo', 'dmoa', 'ddao', 'dvba', 'eoa', 'ecological_cycle_o', 'eco', 'ego', 'esoa', 'ecpo',
    'eso', 'efo', 'eho', 'elk_ho', 'epc', 'evo', 'eao', 'eo', 'esc', 'es', 'ep', 'edo', 'eto',
    'fep', 'fata', 'frofi', 'ffo', 'fla', 'firefly_a', 'fwa', 'fss', 'frcg', 'flood_a', 'fda',
    'fpa', 'fdo', 'fbio', 'foa', 'fox', 'ffa', 'gska', 'gazelle_oa', 'gndo', 'ga', 'gkso',
    'gmo', 'gco', 'gea', 'gto', 'gso', 'gjo', 'gbo', 'gpso', 'goa', 'gsa', 'gwo', 'ggo',
    'go_growth', 'hsa', 'hho', 'hbo', 'hgso', 'hiking_oa', 'hc', 'ho_hippo', 'horse_oa', 'hco',
    'heoa', 'hgs', 'hus', 'hba', 'hsaba', 'ica', 'i_gwo', 'ilshade', 'imode', 'i_woa', 'iwo',
    'ivya', 'jy', 'jso', 'kma', 'kha', 'liwo', 'lco', 'l2smea', 'loa', 'lca', 'lpo', 'lfd',
    'mrfo', 'mshoa', 'mpa', 'memetic_a', 'mso', 'mbo', 'mke', 'moss_go', 'mvpa', 'mfa',
    'msa_e', 'mgo', 'misaco', 'mvo', 'mfea', 'mfea2', 'mts', 'samso', 'nmra', 'nmm',
    'nndrea_so', 'noa', 'ngo', 'nro', 'nca', 'ofa', 'ooa', 'plba', 'pcx', 'parrot_o', 'pso',
    'pfa', 'poa', 'pko', 'plo', 'political_o', 'pro', 'pbil', 'pdo', 'pss', 'puma_o', 'qio',
    'qsa', 'random_s', 'rso', 'rbmo', 'roa', 'rsa', 'rime', 'rmsprop', 'run', 'rfo', 'sfo',
    'ssa', 'sade_sammon', 'scso', 'sbo', 'slo', 'soa', 'seaho', 'saro', 'ssio_rl', 'sboa',
    'saba', 'jde', 'sqp', 'serval_oa', 'srsr', 'sto', 'sa', 'sine_cosine_a', 'scho', 'sma',
    'so_snake', 'snow_oa', 'ssdo', 'sspider_a', 'sso', 'sparrow_sa', 'smo', 'shio', 'sho',
    'squirrel_sa', 'soo', 'sfoa', 'sd', 'spbo', 'shade', 'superb_foa', 'supply_do',
    'sacc_eam2', 'sacoso', 'sade_amss', 'sade_atdsc', 'sapo', 'sos', 'ts', 'tdo', 'tlbo',
    'toa', 'tlco', 'thro', 'toc', 'tpo', 'ttao', 'two', 'tso', 'tsa', 'vcs', 'waoa', 'warso',
    'wca', 'wutp', 'wo_wave', 'info', 'woa', 'wso', 'who', 'wdo', 'ydse', 'zoa',
}

# Algorithms marked as population-based in the table.
_POPULATION_BASED: set[str] = {
    'acgwo', 'aesspso', 'ars', 'aft', 'avoa', 'aso', 'aco', 'acor', 'alo', 'ao', 'arch_oa',
    'aoa', 'artemisinin_o', 'abco', 'aeo', 'aefa', 'afsa', 'agto', 'aha', 'ala', 'apo', 'aro',
    'autov', 'bco', 'bfo', 'bes', 'bmo', 'bat_a', 'bro', 'bea', 'bspga', 'bbo', 'bsa', 'bwo',
    'bka', 'bono', 'bso', 'bboa', 'boa', 'camel', 'capsa', 'cat_so', 'chameleon_sa', 'cgo',
    'cddo', 'cdo', 'chicken_so', 'choa', 'circle_sa', 'csbo', 'clonalg', 'coati_oa',
    'cockroach_so', 'cso', 'coot', 'cro', 'chio', 'cmaes', 'coa', 'crayfish_oa', 'cem', 'csa',
    'cuckoo_s', 'ca', 'do_dandelion', 'dso', 'doa', 'de', 'hde', 'dfo', 'da', 'dbo', 'dmoa',
    'ddao', 'dvba', 'eoa', 'ecological_cycle_o', 'eco', 'ego', 'esoa', 'ecpo', 'eso', 'efo',
    'eho', 'elk_ho', 'epc', 'evo', 'eao', 'eo', 'esc', 'es', 'ep', 'edo', 'eto', 'fep', 'fata',
    'frofi', 'ffo', 'fla', 'firefly_a', 'fwa', 'fss', 'flood_a', 'fda', 'fpa', 'fdo', 'fbio',
    'foa', 'fox', 'ffa', 'gska', 'gazelle_oa', 'gndo', 'ga', 'gkso', 'gmo', 'gco', 'gea',
    'gto', 'gso', 'gjo', 'gbo', 'gpso', 'goa', 'gsa', 'gwo', 'ggo', 'go_growth', 'hsa', 'hho',
    'hbo', 'hgso', 'hiking_oa', 'ho_hippo', 'horse_oa', 'hco', 'heoa', 'hgs', 'hus', 'hba',
    'hsaba', 'ica', 'i_gwo', 'ilshade', 'imode', 'i_woa', 'iwo', 'ivya', 'jy', 'jso', 'kma',
    'kha', 'liwo', 'lco', 'l2smea', 'loa', 'lca', 'lpo', 'lfd', 'mrfo', 'mshoa', 'mpa',
    'memetic_a', 'mso', 'mbo', 'mke', 'moss_go', 'mvpa', 'mfa', 'msa_e', 'mgo', 'misaco',
    'mvo', 'mfea', 'mfea2', 'mts', 'samso', 'nmra', 'nmm', 'nndrea_so', 'noa', 'ngo', 'nro',
    'nca', 'ofa', 'ooa', 'plba', 'pcx', 'parrot_o', 'pso', 'pfa', 'poa', 'pko', 'plo',
    'political_o', 'pro', 'pdo', 'pss', 'puma_o', 'qio', 'qsa', 'random_s', 'rso', 'rbmo',
    'roa', 'rsa', 'rime', 'run', 'rfo', 'sfo', 'ssa', 'sade_sammon', 'scso', 'sbo', 'slo',
    'soa', 'seaho', 'saro', 'ssio_rl', 'sboa', 'saba', 'jde', 'serval_oa', 'srsr', 'sto',
    'sine_cosine_a', 'scho', 'sma', 'so_snake', 'snow_oa', 'ssdo', 'sspider_a', 'sso',
    'sparrow_sa', 'smo', 'shio', 'sho', 'squirrel_sa', 'soo', 'sfoa', 'spbo', 'shade',
    'superb_foa', 'supply_do', 'sacc_eam2', 'sacoso', 'sade_amss', 'sade_atdsc', 'sapo', 'sos',
    'tdo', 'tlbo', 'toa', 'tlco', 'thro', 'toc', 'tpo', 'ttao', 'two', 'tso', 'tsa', 'vcs',
    'waoa', 'warso', 'wca', 'wutp', 'wo_wave', 'info', 'woa', 'wso', 'who', 'wdo', 'ydse',
    'zoa',
}

# Algorithms marked as supporting native candidate injection in the table.
_INJECTION_ENABLED: set[str] = {
    'acgwo', 'aesspso', 'ars', 'aft', 'avoa', 'aso', 'acor', 'alo', 'ao', 'arch_oa', 'aoa',
    'artemisinin_o', 'abco', 'aeo', 'aefa', 'afsa', 'agto', 'aha', 'ala', 'apo', 'aro',
    'autov', 'bco', 'bfo', 'bes', 'bmo', 'bat_a', 'bro', 'bea', 'bspga', 'bbo', 'bsa', 'bwo',
    'bka', 'bono', 'bso', 'bboa', 'boa', 'camel', 'capsa', 'cat_so', 'chameleon_sa', 'cgo',
    'cddo', 'cdo', 'choa', 'circle_sa', 'csbo', 'clonalg', 'coati_oa', 'cockroach_so', 'cso',
    'coot', 'cro', 'chio', 'cmaes', 'coa', 'crayfish_oa', 'cem', 'csa', 'cuckoo_s', 'ca',
    'do_dandelion', 'dso', 'doa', 'de', 'hde', 'dfo', 'da', 'dbo', 'dmoa', 'ddao', 'dvba',
    'eoa', 'ecological_cycle_o', 'eco', 'ego', 'esoa', 'ecpo', 'eso', 'efo', 'eho', 'elk_ho',
    'epc', 'evo', 'eao', 'eo', 'esc', 'es', 'ep', 'edo', 'eto', 'fep', 'fata', 'frofi', 'ffo',
    'fla', 'firefly_a', 'fwa', 'fss', 'flood_a', 'fda', 'fpa', 'fdo', 'fbio', 'foa', 'fox',
    'ffa', 'gska', 'gazelle_oa', 'gndo', 'ga', 'gkso', 'gmo', 'gco', 'gea', 'gto', 'gso',
    'gjo', 'gbo', 'gpso', 'goa', 'gsa', 'gwo', 'ggo', 'go_growth', 'hho', 'hbo', 'hgso',
    'hiking_oa', 'ho_hippo', 'horse_oa', 'hco', 'heoa', 'hgs', 'hus', 'hba', 'hsaba', 'ica',
    'i_gwo', 'ilshade', 'imode', 'i_woa', 'iwo', 'ivya', 'jy', 'jso', 'kma', 'liwo', 'lco',
    'l2smea', 'loa', 'lca', 'lpo', 'lfd', 'mrfo', 'mshoa', 'mpa', 'memetic_a', 'mso', 'mbo',
    'mke', 'moss_go', 'mvpa', 'mfa', 'msa_e', 'mgo', 'misaco', 'mvo', 'mfea', 'mfea2', 'mts',
    'samso', 'nmra', 'nmm', 'nndrea_so', 'noa', 'ngo', 'nro', 'nca', 'ofa', 'ooa', 'plba',
    'pcx', 'parrot_o', 'pso', 'pfa', 'poa', 'pko', 'plo', 'political_o', 'pro', 'pdo', 'pss',
    'puma_o', 'qio', 'qsa', 'random_s', 'rso', 'rbmo', 'roa', 'rsa', 'rime', 'run', 'rfo',
    'sfo', 'ssa', 'sade_sammon', 'scso', 'sbo', 'slo', 'soa', 'seaho', 'saro', 'ssio_rl',
    'sboa', 'saba', 'jde', 'serval_oa', 'srsr', 'sto', 'sa', 'sine_cosine_a', 'scho', 'sma',
    'so_snake', 'snow_oa', 'ssdo', 'sspider_a', 'sso', 'sparrow_sa', 'smo', 'shio', 'sho',
    'squirrel_sa', 'soo', 'sfoa', 'spbo', 'shade', 'superb_foa', 'supply_do', 'sacc_eam2',
    'sacoso', 'sade_amss', 'sade_atdsc', 'sapo', 'sos', 'tdo', 'tlbo', 'toa', 'tlco', 'thro',
    'toc', 'tpo', 'ttao', 'two', 'tso', 'tsa', 'vcs', 'waoa', 'warso', 'wca', 'wutp',
    'wo_wave', 'info', 'woa', 'wso', 'who', 'wdo', 'ydse', 'zoa',
}

# Algorithms marked as supporting restart in the table.
_RESTART_ENABLED: set[str] = {
    'sa',
}

# Algorithms marked as snapshot-fit compatible in the table.
_SNAPSHOT_FIT_ENABLED: set[str] = {
    'acgwo', 'aesspso', 'ars', 'aft', 'avoa', 'aso', 'aco', 'acor', 'alo', 'ao', 'arch_oa',
    'aoa', 'artemisinin_o', 'abco', 'aeo', 'aefa', 'afsa', 'agto', 'aha', 'ala', 'apo', 'aro',
    'autov', 'bco', 'bfo', 'bes', 'bmo', 'bat_a', 'bro', 'bea', 'bspga', 'bbo', 'bsa', 'bwo',
    'bka', 'bono', 'bso', 'bboa', 'boa', 'camel', 'capsa', 'cat_so', 'chameleon_sa', 'cgo',
    'cddo', 'cdo', 'chicken_so', 'choa', 'circle_sa', 'csbo', 'clonalg', 'coati_oa',
    'cockroach_so', 'cso', 'coot', 'cro', 'chio', 'cmaes', 'coa', 'crayfish_oa', 'cem', 'csa',
    'cuckoo_s', 'ca', 'do_dandelion', 'dso', 'doa', 'de', 'hde', 'dfo', 'da', 'dbo', 'dmoa',
    'ddao', 'dvba', 'eoa', 'ecological_cycle_o', 'eco', 'ego', 'esoa', 'ecpo', 'eso', 'efo',
    'eho', 'elk_ho', 'epc', 'evo', 'eao', 'eo', 'esc', 'es', 'ep', 'edo', 'eto', 'fep', 'fata',
    'frofi', 'ffo', 'fla', 'firefly_a', 'fwa', 'fss', 'flood_a', 'fda', 'fpa', 'fdo', 'fbio',
    'foa', 'fox', 'ffa', 'gska', 'gazelle_oa', 'gndo', 'ga', 'gkso', 'gmo', 'gco', 'gea',
    'gto', 'gso', 'gjo', 'gbo', 'gpso', 'goa', 'gsa', 'gwo', 'ggo', 'go_growth', 'hsa', 'hho',
    'hbo', 'hgso', 'hiking_oa', 'ho_hippo', 'horse_oa', 'hco', 'heoa', 'hgs', 'hus', 'hba',
    'hsaba', 'ica', 'i_gwo', 'ilshade', 'imode', 'i_woa', 'iwo', 'ivya', 'jy', 'jso', 'kma',
    'kha', 'liwo', 'lco', 'l2smea', 'loa', 'lca', 'lpo', 'lfd', 'mrfo', 'mshoa', 'mpa',
    'memetic_a', 'mso', 'mbo', 'mke', 'moss_go', 'mvpa', 'mfa', 'msa_e', 'mgo', 'misaco',
    'mvo', 'mfea', 'mfea2', 'mts', 'samso', 'nmra', 'nmm', 'nndrea_so', 'noa', 'ngo', 'nro',
    'nca', 'ofa', 'ooa', 'plba', 'pcx', 'parrot_o', 'pso', 'pfa', 'poa', 'pko', 'plo',
    'political_o', 'pro', 'pdo', 'pss', 'puma_o', 'qio', 'qsa', 'random_s', 'rso', 'rbmo',
    'roa', 'rsa', 'rime', 'run', 'rfo', 'sfo', 'ssa', 'sade_sammon', 'scso', 'sbo', 'slo',
    'soa', 'seaho', 'saro', 'ssio_rl', 'sboa', 'saba', 'jde', 'serval_oa', 'srsr', 'sto',
    'sine_cosine_a', 'scho', 'sma', 'so_snake', 'snow_oa', 'ssdo', 'sspider_a', 'sso',
    'sparrow_sa', 'smo', 'shio', 'sho', 'squirrel_sa', 'soo', 'sfoa', 'spbo', 'shade',
    'superb_foa', 'supply_do', 'sacc_eam2', 'sacoso', 'sade_amss', 'sade_atdsc', 'sapo', 'sos',
    'tdo', 'tlbo', 'toa', 'tlco', 'thro', 'toc', 'tpo', 'ttao', 'two', 'tso', 'tsa', 'vcs',
    'waoa', 'warso', 'wca', 'wutp', 'wo_wave', 'info', 'woa', 'wso', 'who', 'wdo', 'ydse',
    'zoa',
}

# Optional descriptive metadata from the table.
_ALGORITHM_NAMES: dict[str, str] = {
    'adam'              : 'Adam (Adaptive Moment Estimation)',
    'acgwo'             : 'Adaptive Chaotic Grey Wolf Optimizer',
    'aesspso'           : 'Adaptive Exploration State-Space Particle Swarm Optimization',
    'ars'               : 'Adaptive Random Search',
    'aft'               : 'Affix Optimization',
    'avoa'              : 'African Vultures Optimization Algorithm',
    'aso'               : 'Anarchic Society Optimization',
    'aco'               : 'Ant Colony Optimization',
    'acor'              : 'Ant Colony Optimization (Continuous)',
    'alo'               : 'Ant Lion Optimizer',
    'ao'                : 'Aquila Optimizer',
    'arch_oa'           : 'Archimedes Optimization Algorithm',
    'aoa'               : 'Arithmetic Optimization Algorithm',
    'artemisinin_o'     : 'Artemisinin Optimization',
    'abco'              : 'Artificial Bee Colony Optimization',
    'aeo'               : 'Artificial Ecosystem Optimization',
    'aefa'              : 'Artificial Electric Field Algorithm',
    'afsa'              : 'Artificial Fish Swarm Algorithm',
    'agto'              : 'Artificial Gorilla Troops Optimizer',
    'aha'               : 'Artificial Hummingbird Algorithm',
    'ala'               : 'Artificial Lemming Algorithm',
    'apo'               : 'Artificial Protozoa Optimizer',
    'aro'               : 'Artificial Rabbits Optimization',
    'autov'             : 'Automated Design of Variation Operators',
    'bco'               : 'Bacterial Chemotaxis Optimizer',
    'bfo'               : 'Bacterial Foraging Optimization',
    'bes'               : 'Bald Eagle Search',
    'bmo'               : 'Barnacles Mating Optimizer',
    'bat_a'             : 'Bat Algorithm',
    'bro'               : 'Battle Royale Optimization',
    'bea'               : 'Bees Algorithm',
    'bfgs'              : 'BFGS Quasi-Newton Method',
    'bspga'             : 'Binary Space Partition Tree Genetic Algorithm',
    'bbo'               : 'Biogeography-Based Optimization',
    'bsa'               : 'Bird Swarm Algorithm',
    'bwo'               : 'Black Widow Optimization',
    'bka'               : 'Black-winged Kite Algorithm',
    'bono'              : 'Bonobo Optimizer',
    'bso'               : 'Brain Storm Optimization',
    'bboa'              : 'Brown-Bear Optimization Algorithm',
    'boa'               : 'Butterfly Optimization Algorithm',
    'camel'             : 'Camel Algorithm',
    'capsa'             : 'Capuchin Search Algorithm',
    'cat_so'            : 'Cat Swarm Optimization',
    'chameleon_sa'      : 'Chameleon Swarm Algorithm',
    'cgo'               : 'Chaos Game Optimization',
    'cddo'              : 'Cheetah Based Optimization',
    'cdo'               : 'Cheetah Optimizer',
    'chicken_so'        : 'Chicken Swarm Optimization',
    'choa'              : 'Chimp Optimization Algorithm',
    'circle_sa'         : 'Circle-Based Search Algorithm',
    'csbo'              : 'Circulatory System Based Optimization',
    'clonalg'           : 'Clonal Selection Algorithm',
    'coati_oa'          : 'Coati Optimization Algorithm',
    'cockroach_so'      : 'Cockroach Swarm Optimization',
    'cso'               : 'Competitive Swarm Optimizer',
    'coot'              : 'COOT Bird Optimization',
    'cro'               : 'Coral Reefs Optimization',
    'chio'              : 'Coronavirus Herd Immunity Optimization',
    'cmaes'             : 'Covariance Matrix Adaptation Evolution Strategy',
    'coa'               : 'Coyote Optimization Algorithm',
    'crayfish_oa'       : 'Crayfish Optimization Algorithm',
    'cem'               : 'Cross Entropy Method',
    'csa'               : 'Crow Search Algorithm',
    'cuckoo_s'          : 'Cuckoo Search',
    'ca'                : 'Cultural Algorithm',
    'do_dandelion'      : 'Dandelion Optimizer',
    'dso'               : 'Deep Sleep Optimiser',
    'doa'               : 'Deer Hunting Optimization Algorithm',
    'de'                : 'Differential Evolution',
    'hde'               : 'Differential Evolution MTS',
    'dfo'               : 'Dispersive Fly Optimization',
    'da'                : 'Dragonfly Algorithm',
    'dbo'               : 'Dung Beetle Optimizer',
    'dmoa'              : 'Dwarf Mongoose Optimization Algorithm',
    'ddao'              : 'Dynamic Differential Annealed Optimization',
    'dvba'              : 'Dynamic Virtual Bats Algorithm',
    'eoa'               : 'Earthworm Optimization Algorithm',
    'ecological_cycle_o': 'Ecological Cycle Optimizer',
    'eco'               : 'Educational Competition Optimizer',
    'ego'               : 'Efficient Global Optimization',
    'esoa'              : 'Egret Swarm Optimization Algorithm',
    'ecpo'              : 'Electric Charged Particles Optimization',
    'eso'               : 'Electric Squirrel Optimizer',
    'efo'               : 'Electromagnetic Field Optimization',
    'eho'               : 'Elephant Herding Optimization',
    'elk_ho'            : 'Elk Herd Optimizer',
    'epc'               : 'Emperor Penguin Colony',
    'evo'               : 'Energy Valley Optimizer',
    'eao'               : 'Enzyme Activity Optimizer',
    'eo'                : 'Equilibrium Optimizer',
    'esc'               : 'Escape Algorithm',
    'es'                : 'Evolution Strategy (mu + lambda)',
    'ep'                : 'Evolutionary Programming',
    'edo'               : 'Exponential Distribution Optimizer',
    'eto'               : 'Exponential-Trigonometric Optimization',
    'fep'               : 'Fast Evolutionary Programming',
    'fata'              : 'FATA Geophysics Optimizer',
    'frofi'             : 'Feasibility Rule with Objective Function Information',
    'ffo'               : 'Fennec Fox Optimizer',
    'fla'               : "Fick's Law Algorithm",
    'firefly_a'         : 'Firefly Algorithm',
    'fwa'               : 'Fireworks Algorithm',
    'fss'               : 'Fish School Search',
    'frcg'              : 'Fletcher-Reeves Conjugate Gradient',
    'flood_a'           : 'Flood Algorithm',
    'fda'               : 'Flow Direction Algorithm',
    'fpa'               : 'Flower Pollination Algorithm',
    'fdo'               : 'Flying Dobsonflies Optimizer',
    'fbio'              : 'Forensic-Based Investigation Optimization',
    'foa'               : 'Forest Optimization Algorithm',
    'fox'               : 'Fox Optimizer',
    'ffa'               : 'Fruit-Fly Algorithm',
    'gska'              : 'Gaining-Sharing Knowledge Algorithm',
    'gazelle_oa'        : 'Gazelle Optimization Algorithm',
    'gndo'              : 'Generalized Normal Distribution Optimizer',
    'ga'                : 'Genetic Algorithm',
    'gkso'              : 'Genghis Khan Shark Optimizer',
    'gmo'               : 'Geometric Mean Optimizer',
    'gco'               : 'Germinal Center Optimization',
    'gea'               : 'Geyser Inspired Algorithm',
    'gto'               : 'Giant Trevally Optimizer',
    'gso'               : 'Glowworm Swarm Optimization',
    'gjo'               : 'Golden Jackal Optimizer',
    'gbo'               : 'Gradient-Based Optimizer',
    'gpso'              : 'Gradient-Based Particle Swarm Optimization',
    'goa'               : 'Grasshopper Optimization Algorithm',
    'gsa'               : 'Gravitational Search Algorithm',
    'gwo'               : 'Grey Wolf Optimizer',
    'ggo'               : 'Greylag Goose Optimization',
    'go_growth'         : 'Growth Optimizer',
    'hsa'               : 'Harmony Search Algorithm',
    'hho'               : 'Harris Hawks Optimization',
    'hbo'               : 'Heap-Based Optimizer',
    'hgso'              : 'Henry Gas Solubility Optimization',
    'hiking_oa'         : 'Hiking Optimization Algorithm',
    'hc'                : 'Hill Climb Algorithm',
    'ho_hippo'          : 'Hippopotamus Optimization Algorithm',
    'horse_oa'          : 'Horse Herd Optimization Algorithm',
    'hco'               : 'Human Conception Optimizer',
    'heoa'              : 'Human Evolutionary Optimization Algorithm',
    'hgs'               : 'Hunger Games Search',
    'hus'               : 'Hunting Search Algorithm',
    'hba'               : 'Hybrid Bat Algorithm',
    'hsaba'             : 'Hybrid Self-Adaptive Bat Algorithm',
    'ica'               : 'Imperialist Competitive Algorithm',
    'i_gwo'             : 'Improved Grey Wolf Optimizer',
    'ilshade'           : 'Improved L-SHADE',
    'imode'             : 'Improved Multi-Operator Differential Evolution',
    'i_woa'             : 'Improved Whale Optimization Algorithm',
    'iwo'               : 'Invasive Weed Optimization',
    'ivya'              : 'Ivy Algorithm',
    'jy'                : 'Jaya Algorithm',
    'jso'               : 'Jellyfish Search Optimizer',
    'kma'               : 'Komodo Mlipir Algorithm',
    'kha'               : 'Krill Herd Algorithm',
    'liwo'              : 'Leaf in Wind Optimization',
    'lco'               : 'Life Choice-Based Optimizer',
    'l2smea'            : 'Linear Subspace Surrogate Modeling Evolutionary Algorithm',
    'loa'               : 'Lion Optimization Algorithm',
    'lca'               : 'Liver Cancer Algorithm',
    'lpo'               : 'Lungs Performance-Based Optimization',
    'lfd'               : 'LÃ©vy Flight Distribution',
    'mrfo'              : 'Manta Ray Foraging Optimization',
    'mshoa'             : 'Mantis Shrimp Optimization Algorithm',
    'mpa'               : 'Marine Predators Algorithm',
    'memetic_a'         : 'Memetic Algorithm',
    'mso'               : 'Mirage-Search Optimizer',
    'mbo'               : 'Monarch Butterfly Optimization',
    'mke'               : 'Monkey King Evolution V1',
    'moss_go'           : 'Moss Growth Optimization',
    'mvpa'              : 'Most Valuable Player Algorithm',
    'mfa'               : 'Moth Flame Algorithm',
    'msa_e'             : 'Moth Search Algorithm',
    'mgo'               : 'Mountain Gazelle Optimizer',
    'misaco'            : 'Multi-Surrogate-Assisted Ant Colony Optimization',
    'mvo'               : 'Multi-Verse Optimizer',
    'mfea'              : 'Multifactorial Evolutionary Algorithm',
    'mfea2'             : 'Multifactorial Evolutionary Algorithm II',
    'mts'               : 'Multiple Trajectory Search',
    'samso'             : 'Multiswarm-Assisted Expensive Optimization',
    'nmra'              : 'Naked Mole-Rat Algorithm',
    'nmm'               : 'Nelder-Mead Method',
    'nndrea_so'         : 'Neural Network-Based Dimensionality Reduction Evolutionary Algorithm (SO)',
    'noa'               : 'Nizar Optimization Algorithm',
    'ngo'               : 'Northern Goshawk Optimization',
    'nro'               : 'Nuclear Reaction Optimization',
    'nca'               : 'Numeric Crunch Algorithm',
    'ofa'               : 'Optimal Foraging Algorithm',
    'ooa'               : 'Osprey Optimization Algorithm',
    'plba'              : 'Parameter-Free Bat Algorithm',
    'pcx'               : 'Parent-Centric Crossover (G3-PCX style)',
    'parrot_o'          : 'Parrot Optimizer',
    'pso'               : 'Particle Swarm Optimization',
    'pfa'               : 'Pathfinder Algorithm',
    'poa'               : 'Pelican Optimization Algorithm',
    'pko'               : 'Pied Kingfisher Optimizer',
    'plo'               : 'Polar Lights Optimizer',
    'political_o'       : 'Political Optimizer',
    'pro'               : 'Poor and Rich Optimization Algorithm',
    'pbil'              : 'Population-Based Incremental Learning',
    'pdo'               : 'Prairie Dog Optimization Algorithm',
    'pss'               : 'Prominent Space Search',
    'puma_o'            : 'Puma Optimizer',
    'qio'               : 'Quadratic Interpolation Optimization',
    'qsa'               : 'Queuing Search Algorithm',
    'random_s'          : 'Random Search',
    'rso'               : 'Rat Swarm Optimizer',
    'rbmo'              : 'Red-billed Blue Magpie Optimizer',
    'roa'               : 'Remora Optimization Algorithm',
    'rsa'               : 'Reptile Search Algorithm',
    'rime'              : 'RIME-ice Algorithm',
    'rmsprop'           : 'RMSProp',
    'run'               : 'RUNge Kutta Optimizer',
    'rfo'               : "RÃ¼ppell's Fox Optimizer",
    'sfo'               : 'Sailfish Optimizer',
    'ssa'               : 'Salp Swarm Algorithm',
    'sade_sammon'       : 'Sammon Mapping Assisted Differential Evolution',
    'scso'              : 'Sand Cat Swarm Optimization',
    'sbo'               : 'Satin Bowerbird Optimizer',
    'slo'               : 'Sea Lion Optimization',
    'soa'               : 'Seagull Optimization Algorithm',
    'seaho'             : 'Seahorse Optimizer',
    'saro'              : 'Search And Rescue Optimization',
    'ssio_rl'           : 'Search Space Independent Operator Based Deep Reinforcement Learning',
    'sboa'              : 'Secretary Bird Optimization Algorithm',
    'saba'              : 'Self-Adaptive Bat Algorithm',
    'jde'               : 'Self-Adaptive Differential Evolution',
    'sqp'               : 'Sequential Quadratic Programming',
    'serval_oa'         : 'Serval Optimization Algorithm',
    'srsr'              : 'Shuffle-based Runner-Root Algorithm',
    'sto'               : 'Siberian Tiger Optimization',
    'sa'                : 'Simulated Annealing',
    'sine_cosine_a'     : 'Sine Cosine Algorithm',
    'scho'              : 'Sinh Cosh Optimizer',
    'sma'               : 'Slime Mould Algorithm',
    'so_snake'          : 'Snake Optimizer',
    'snow_oa'           : 'Snow Ablation Optimizer',
    'ssdo'              : 'Social Ski-Driver Optimization',
    'sspider_a'         : 'Social Spider Algorithm',
    'sso'               : 'Social Spider Swarm Optimizer',
    'sparrow_sa'        : 'Sparrow Search Algorithm',
    'smo'               : 'Spider Monkey Optimization',
    'shio'              : 'Spotted Hyena Inspired Optimizer',
    'sho'               : 'Spotted Hyena Optimizer',
    'squirrel_sa'       : 'Squirrel Search Algorithm',
    'soo'               : 'Star Oscillator Optimization',
    'sfoa'              : 'Starfish Optimization Algorithm',
    'sd'                : 'Steepest Descent',
    'spbo'              : 'Student Psychology Based Optimization',
    'shade'             : 'Success-History Adaptive Differential Evolution',
    'superb_foa'        : 'Superb Fairy-wren Optimization Algorithm',
    'supply_do'         : 'Supply-Demand-Based Optimization',
    'sacc_eam2'         : 'Surrogate-Assisted Cooperative Co-Evolutionary Algorithm of Minamo II',
    'sacoso'            : 'Surrogate-Assisted Cooperative Swarm Optimization',
    'sade_amss'         : 'Surrogate-Assisted DE with Adaptive Multi-Subspace Search',
    'sade_atdsc'        : 'Surrogate-Assisted DE with Adaptive Training Data Selection Criterion',
    'sapo'              : 'Surrogate-Assisted Partial Optimization',
    'sos'               : 'Symbiotic Organisms Search',
    'ts'                : 'Tabu Search',
    'tdo'               : 'Tasmanian Devil Optimization',
    'tlbo'              : 'Teaching Learning Based Optimization',
    'toa'               : 'Teamwork Optimization Algorithm',
    'tlco'              : 'Termite Life Cycle Optimizer',
    'thro'              : 'Tianji Horse Racing Optimizer',
    'toc'               : 'Tornado Optimizer with Coriolis Force',
    'tpo'               : 'Tree Physiology Optimization',
    'ttao'              : 'Triangulation Topology Aggregation Optimizer',
    'two'               : 'Tug of War Optimization',
    'tso'               : 'Tuna Swarm Optimization',
    'tsa'               : 'Tunicate Swarm Algorithm',
    'vcs'               : 'Virus Colony Search',
    'waoa'              : 'Walrus Optimization Algorithm',
    'warso'             : 'War Strategy Optimization',
    'wca'               : 'Water Cycle Algorithm',
    'wutp'              : 'Water Uptake and Transport in Plants',
    'wo_wave'           : 'Wave Optimization Algorithm',
    'info'              : 'Weighting and Inertia Random Walk Optimizer',
    'woa'               : 'Whale Optimization Algorithm',
    'wso'               : 'White Shark Optimizer',
    'who'               : 'Wildebeest Herd Optimization',
    'wdo'               : 'Wind Driven Optimization',
    'ydse'              : "Young's Double-Slit Experiment Optimizer",
    'zoa'               : 'Zebra Optimization Algorithm',
}

_ALGORITHM_FAMILIES: dict[str, str] = {
    'adam'                  : 'math',
    'acgwo'                 : 'swarm',
    'aesspso'               : 'swarm',
    'ars'                   : 'trajectory',
    'aft'                   : 'human',
    'avoa'                  : 'swarm',
    'aso'                   : 'swarm',
    'aco'                   : 'swarm',
    'acor'                  : 'swarm',
    'alo'                   : 'swarm',
    'ao'                    : 'swarm',
    'arch_oa'               : 'physics',
    'aoa'                   : 'swarm',
    'artemisinin_o'         : 'nature',
    'abco'                  : 'swarm',
    'aeo'                   : 'human',
    'aefa'                  : 'physics',
    'afsa'                  : 'swarm',
    'agto'                  : 'swarm',
    'aha'                   : 'swarm',
    'ala'                   : 'swarm',
    'apo'                   : 'swarm',
    'aro'                   : 'swarm',
    'autov'                 : 'evolutionary',
    'bco'                   : 'nature',
    'bfo'                   : 'swarm',
    'bes'                   : 'swarm',
    'bmo'                   : 'swarm',
    'bat_a'                 : 'swarm',
    'bro'                   : 'human',
    'bea'                   : 'swarm',
    'bfgs'                  : 'math',
    'bspga'                 : 'evolutionary',
    'bbo'                   : 'evolutionary',
    'bsa'                   : 'swarm',
    'bwo'                   : 'evolutionary',
    'bka'                   : 'swarm',
    'bono'                  : 'swarm',
    'bso'                   : 'human',
    'bboa'                  : 'swarm',
    'boa'                   : 'swarm',
    'camel'                 : 'swarm',
    'capsa'                 : 'swarm',
    'cat_so'                : 'swarm',
    'chameleon_sa'          : 'swarm',
    'cgo'                   : 'math',
    'cddo'                  : 'human',
    'cdo'                   : 'swarm',
    'chicken_so'            : 'swarm',
    'choa'                  : 'swarm',
    'circle_sa'             : 'math',
    'csbo'                  : 'nature',
    'clonalg'               : 'evolutionary',
    'coati_oa'              : 'swarm',
    'cockroach_so'          : 'swarm',
    'cso'                   : 'swarm',
    'coot'                  : 'swarm',
    'cro'                   : 'evolutionary',
    'chio'                  : 'human',
    'cmaes'                 : 'evolutionary',
    'coa'                   : 'swarm',
    'crayfish_oa'           : 'swarm',
    'cem'                   : 'distribution',
    'csa'                   : 'swarm',
    'cuckoo_s'              : 'swarm',
    'ca'                    : 'evolutionary',
    'do_dandelion'          : 'physics',
    'dso'                   : 'human',
    'doa'                   : 'human',
    'de'                    : 'evolutionary',
    'hde'                   : 'evolutionary',
    'dfo'                   : 'swarm',
    'da'                    : 'swarm',
    'dbo'                   : 'swarm',
    'dmoa'                  : 'swarm',
    'ddao'                  : 'physics',
    'dvba'                  : 'swarm',
    'eoa'                   : 'swarm',
    'ecological_cycle_o'    : 'swarm',
    'eco'                   : 'human',
    'ego'                   : 'distribution',
    'esoa'                  : 'swarm',
    'ecpo'                  : 'physics',
    'eso'                   : 'physics',
    'efo'                   : 'physics',
    'eho'                   : 'swarm',
    'elk_ho'                : 'swarm',
    'epc'                   : 'swarm',
    'evo'                   : 'physics',
    'eao'                   : 'nature',
    'eo'                    : 'physics',
    'esc'                   : 'human',
    'es'                    : 'evolutionary',
    'ep'                    : 'evolutionary',
    'edo'                   : 'math',
    'eto'                   : 'math',
    'fep'                   : 'evolutionary',
    'fata'                  : 'physics',
    'frofi'                 : 'evolutionary',
    'ffo'                   : 'swarm',
    'fla'                   : 'physics',
    'firefly_a'             : 'swarm',
    'fwa'                   : 'swarm',
    'fss'                   : 'swarm',
    'frcg'                  : 'math',
    'flood_a'               : 'physics',
    'fda'                   : 'swarm',
    'fpa'                   : 'swarm',
    'fdo'                   : 'swarm',
    'fbio'                  : 'human',
    'foa'                   : 'swarm',
    'fox'                   : 'swarm',
    'ffa'                   : 'swarm',
    'gska'                  : 'human',
    'gazelle_oa'            : 'swarm',
    'gndo'                  : 'math',
    'ga'                    : 'evolutionary',
    'gkso'                  : 'swarm',
    'gmo'                   : 'swarm',
    'gco'                   : 'human',
    'gea'                   : 'physics',
    'gto'                   : 'swarm',
    'gso'                   : 'swarm',
    'gjo'                   : 'swarm',
    'gbo'                   : 'math',
    'gpso'                  : 'swarm',
    'goa'                   : 'swarm',
    'gsa'                   : 'physics',
    'gwo'                   : 'swarm',
    'ggo'                   : 'swarm',
    'go_growth'             : 'swarm',
    'hsa'                   : 'trajectory',
    'hho'                   : 'swarm',
    'hbo'                   : 'human',
    'hgso'                  : 'physics',
    'hiking_oa'             : 'human',
    'hc'                    : 'trajectory',
    'ho_hippo'              : 'swarm',
    'horse_oa'              : 'swarm',
    'hco'                   : 'human',
    'heoa'                  : 'human',
    'hgs'                   : 'swarm',
    'hus'                   : 'swarm',
    'hba'                   : 'swarm',
    'hsaba'                 : 'swarm',
    'ica'                   : 'human',
    'i_gwo'                 : 'swarm',
    'ilshade'               : 'evolutionary',
    'imode'                 : 'evolutionary',
    'i_woa'                 : 'swarm',
    'iwo'                   : 'nature',
    'ivya'                  : 'nature',
    'jy'                    : 'swarm',
    'jso'                   : 'swarm',
    'kma'                   : 'swarm',
    'kha'                   : 'swarm',
    'liwo'                  : 'physics',
    'lco'                   : 'human',
    'l2smea'                : 'evolutionary',
    'loa'                   : 'swarm',
    'lca'                   : 'nature',
    'lpo'                   : 'nature',
    'lfd'                   : 'swarm',
    'mrfo'                  : 'swarm',
    'mshoa'                 : 'swarm',
    'mpa'                   : 'swarm',
    'memetic_a'             : 'evolutionary',
    'mso'                   : 'physics',
    'mbo'                   : 'swarm',
    'mke'                   : 'evolutionary',
    'moss_go'               : 'nature',
    'mvpa'                  : 'human',
    'mfa'                   : 'swarm',
    'msa_e'                 : 'swarm',
    'mgo'                   : 'swarm',
    'misaco'                : 'swarm',
    'mvo'                   : 'swarm',
    'mfea'                  : 'evolutionary',
    'mfea2'                 : 'evolutionary',
    'mts'                   : 'trajectory',
    'samso'                 : 'swarm',
    'nmra'                  : 'swarm',
    'nmm'                   : 'trajectory',
    'nndrea_so'             : 'evolutionary',
    'noa'                   : 'math',
    'ngo'                   : 'swarm',
    'nro'                   : 'physics',
    'nca'                   : 'math',
    'ofa'                   : 'swarm',
    'ooa'                   : 'swarm',
    'plba'                  : 'swarm',
    'pcx'                   : 'evolutionary',
    'parrot_o'              : 'swarm',
    'pso'                   : 'swarm',
    'pfa'                   : 'swarm',
    'poa'                   : 'swarm',
    'pko'                   : 'swarm',
    'plo'                   : 'physics',
    'political_o'           : 'human',
    'pro'                   : 'human',
    'pbil'                  : 'distribution',
    'pdo'                   : 'swarm',
    'pss'                   : 'math',
    'puma_o'                : 'swarm',
    'qio'                   : 'math',
    'qsa'                   : 'human',
    'random_s'              : 'trajectory',
    'rso'                   : 'swarm',
    'rbmo'                  : 'swarm',
    'roa'                   : 'swarm',
    'rsa'                   : 'swarm',
    'rime'                  : 'physics',
    'rmsprop'               : 'math',
    'run'                   : 'math',
    'rfo'                   : 'swarm',
    'sfo'                   : 'swarm',
    'ssa'                   : 'swarm',
    'sade_sammon'           : 'evolutionary',
    'scso'                  : 'swarm',
    'sbo'                   : 'swarm',
    'slo'                   : 'swarm',
    'soa'                   : 'swarm',
    'seaho'                 : 'swarm',
    'saro'                  : 'human',
    'ssio_rl'               : 'evolutionary',
    'sboa'                  : 'swarm',
    'saba'                  : 'swarm',
    'jde'                   : 'evolutionary',
    'sqp'                   : 'math',
    'serval_oa'             : 'swarm',
    'srsr'                  : 'swarm',
    'sto'                   : 'swarm',
    'sa'                    : 'trajectory',
    'sine_cosine_a'         : 'swarm',
    'scho'                  : 'math',
    'sma'                   : 'nature',
    'so_snake'              : 'swarm',
    'snow_oa'               : 'physics',
    'ssdo'                  : 'human',
    'sspider_a'             : 'swarm',
    'sso'                   : 'swarm',
    'sparrow_sa'            : 'swarm',
    'smo'                   : 'swarm',
    'shio'                  : 'math',
    'sho'                   : 'swarm',
    'squirrel_sa'           : 'swarm',
    'soo'                   : 'physics',
    'sfoa'                  : 'swarm',
    'sd'                    : 'math',
    'spbo'                  : 'swarm',
    'shade'                 : 'evolutionary',
    'superb_foa'            : 'swarm',
    'supply_do'             : 'human',
    'sacc_eam2'             : 'evolutionary',
    'sacoso'                : 'swarm',
    'sade_amss'             : 'evolutionary',
    'sade_atdsc'            : 'evolutionary',
    'sapo'                  : 'evolutionary',
    'sos'                   : 'swarm',
    'ts'                    : 'trajectory',
    'tdo'                   : 'swarm',
    'tlbo'                  : 'swarm',
    'toa'                   : 'human',
    'tlco'                  : 'swarm',
    'thro'                  : 'human',
    'toc'                   : 'physics',
    'tpo'                   : 'nature',
    'ttao'                  : 'math',
    'two'                   : 'physics',
    'tso'                   : 'swarm',
    'tsa'                   : 'swarm',
    'vcs'                   : 'swarm',
    'waoa'                  : 'swarm',
    'warso'                 : 'human',
    'wca'                   : 'human',
    'wutp'                  : 'nature',
    'wo_wave'               : 'physics',
    'info'                  : 'math',
    'woa'                   : 'swarm',
    'wso'                   : 'swarm',
    'who'                   : 'swarm',
    'wdo'                   : 'physics',
    'ydse'                  : 'physics',
    'zoa'                   : 'swarm',
}

_ALGORITHM_DOIS: dict[str, str] = {
    'adam'              : '10.48550/arXiv.1412.6980',
    'acgwo'             : '10.1007/s42835-023-01621-w',
    'aesspso'           : '10.1016/j.swevo.2025.101868',
    'ars'               : '10.1287/ijoc.1110.0494',
    'aft'               : '10.1007/s00521-021-06392-x',
    'avoa'              : '10.1016/j.cie.2021.107408',
    'aso'               : '10.1109/CEC.2011.5949940',
    'aco'               : '10.1109/CEC.1999.782657',
    'acor'              : '10.1007/s10732-008-9062-4',
    'alo'               : '10.1016/j.advengsoft.2015.01.010',
    'ao'                : '10.1016/j.cie.2021.107250',
    'arch_oa'           : '10.1007/s10489-020-01893-z',
    'aoa'               : '10.1016/j.cma.2020.113609',
    'artemisinin_o'     : '10.1016/j.displa.2024.102740',
    'abco'              : '10.1007/978-3-540-72950-1_77',
    'aeo'               : '10.1007/s00521-019-04452-x',
    'aefa'              : '10.1016/j.swevo.2019.03.013',
    'afsa'              : '10.12011/1000-6788(2002)11-32',
    'agto'              : '10.1002/int.22535',
    'aha'               : '10.1016/j.cma.2022.114194',
    'ala'               : '10.1007/s10462-025-11108-5',
    'apo'               : '10.1016/j.knosys.2024.111737',
    'aro'               : '10.1016/j.engappai.2022.105082',
    'autov'             : '10.23919/CJE.2022.00.038',
    'bco'               : '10.1109/MCS.2002.1004010',
    'bfo'               : '10.1109/MCS.2002.1004010',
    'bes'               : '10.1007/s10462-019-09732-5',
    'bmo'               : '10.1109/ICOICA.2019.8895393',
    'bat_a'             : '10.1007/978-3-642-12538-6_6',
    'bro'               : '10.1007/s00521-020-05004-4',
    'bea'               : '10.1016/B978-008045157-2/50081-X',
    'bfgs'              : '10.1090/S0025-5718-1970-0274029-X',
    'bspga'             : '10.1016/j.ins.2019.11.055',
    'bbo'               : '10.1109/TEVC.2008.919004',
    'bsa'               : '10.1080/0952813X.2015.1042530',
    'bwo'               : '10.1016/j.engappai.2019.103249',
    'bka'               : '10.1007/s10462-024-10723-4',
    'bono'              : '10.1007/s10489-021-02830-0',
    'bso'               : '10.1007/978-3-642-21515-5_36',
    'bboa'              : '10.1201/9781003337003-6',
    'boa'               : '10.1007/s00500-018-3102-4',
    'camel'             : '10.33762/eeej.2016.118375',
    'capsa'             : '10.1007/s00521-020-05066-5',
    'cat_so'            : '10.1007/978-3-540-36668-3_94',
    'chameleon_sa'      : '10.1016/j.eswa.2021.114685',
    'cgo'               : '10.1007/s10462-020-09867-w',
    'cddo'              : '10.1038/s41598-022-14338-z',
    'cdo'               : '10.1038/s41598-022-14338-z',
    'chicken_so'        : '10.1007/978-3-319-11857-4_10',
    'choa'              : '10.1016/j.eswa.2020.113338',
    'circle_sa'         : '10.3390/math10101626',
    'csbo'              : '10.1007/s10462-021-10044-y',
    'clonalg'           : '10.1109/TEVC.2002.1011539',
    'coati_oa'          : '10.1016/j.knosys.2022.110011',
    'cockroach_so'      : '10.1109/ICCET.2010.5485993',
    'cso'               : '10.1109/TCYB.2014.2314537',
    'coot'              : '10.1016/j.eswa.2021.115352',
    'cro'               : '10.1155/2014/739768',
    'chio'              : '10.1007/s00521-020-05296-6',
    'cmaes'             : '10.1162/106365602760972767',
    'coa'               : '10.1109/CEC.2018.8477769',
    'crayfish_oa'       : '10.1007/s10462-023-10567-4',
    'cem'               : '10.1016/S0377-2217(96)00385-2',
    'csa'               : '10.1016/j.compstruc.2016.03.001',
    'cuckoo_s'          : '10.1109/NABIC.2009.5393690',
    'ca'                : '10.1142/9789814534116',
    'do_dandelion'      : '10.1016/j.engappai.2022.105075',
    'dso'               : '10.1109/ACCESS.2023.3299804',
    'doa'               : '10.1093/comjnl/bxy133',
    'de'                : '10.1023/A:1008202821328',
    'hde'               : '10.1109/CEC.2009.4983179',
    'dfo'               : '10.15439/2014F142',
    'da'                : '10.1007/s00521-015-1920-1',
    'dbo'               : '10.1007/s11227-022-04959-6',
    'dmoa'              : '10.1016/j.cma.2022.114570',
    'ddao'              : '10.1016/j.asoc.2020.106392',
    'dvba'              : '10.1109/INCoS.2014.40',
    'eoa'               : '10.1504/IJBIC.2015.10004283',
    'ecological_cycle_o': '10.48550/arXiv.2508.20458',
    'eco'               : '10.1080/00207721.2024.2308282',
    'ego'               : '10.1023/A:1008306431147',
    'esoa'              : '10.3390/biomimetics7040144',
    'ecpo'              : '10.1007/s10462-020-09920-8',
    'eso'               : '10.3390/make7010024',
    'efo'               : '10.1016/j.asoc.2015.10.048',
    'eho'               : '10.1109/ISCBI.2015.8',
    'elk_ho'            : '10.1007/s10462-023-10680-4',
    'epc'               : '10.1016/j.knosys.2018.06.001',
    'evo'               : '10.1038/s41598-022-27344-y',
    'eao'               : '10.3390/math12213326',
    'eo'                : '10.1016/j.knosys.2019.105190',
    'esc'               : '10.1007/s13748-024-00351-6',
    'es'                : '10.1023/A:1015059928466',
    'ep'                : '10.1007/BF00175356',
    'edo'               : '10.1007/s10462-022-10317-4',
    'eto'               : '10.1016/j.asoc.2023.110148',
    'fep'               : '10.1109/4235.771163',
    'fata'              : '10.1016/j.neucom.2024.128289',
    'frofi'             : '10.1109/TCYB.2015.2493239',
    'ffo'               : '10.1109/ACCESS.2022.3197745',
    'fla'               : '10.1016/j.knosys.2022.110146',
    'firefly_a'         : '10.1504/IJBIC.2010.032124',
    'fwa'               : '10.1016/j.asoc.2017.10.046',
    'fss'               : '10.1109/ICSMC.2008.4811695',
    'frcg'              : '10.1093/comjnl/7.2.149',
    'flood_a'           : '10.1007/s11227-024-06054-6',
    'fda'               : '10.1016/j.cie.2021.107224',
    'fpa'               : '10.1007/978-3-642-32894-7_27',
    'fdo'               : '10.1016/j.knosys.2020.105574',
    'fbio'              : '10.1016/j.asoc.2020.106339',
    'foa'               : '10.1016/j.eswa.2014.05.009',
    'fox'               : '10.1007/s10489-022-03533-0',
    'ffa'               : '10.1016/j.knosys.2011.07.001',
    'gska'              : '10.1007/s13042-019-01053-x',
    'gazelle_oa'        : '10.1007/s00521-022-07224-4',
    'gndo'              : '10.1016/j.enconman.2020.113301',
    'ga'                : '10.7551/mitpress/1090.001.0001',
    'gkso'              : '10.1007/s10462-023-10618-w',
    'gmo'               : '10.1007/s00500-023-08202-z',
    'gco'               : '10.1002/int.21892',
    'gea'               : '10.1007/s42235-023-00426-5',
    'gto'               : '10.1109/ACCESS.2022.3223388',
    'gso'               : '10.1007/978-3-319-51595-3',
    'gjo'               : '10.1016/j.eswa.2022.116924',
    'gbo'               : '10.1007/s00500-020-05180-6',
    'gpso'              : '10.1016/j.asoc.2011.10.007',
    'goa'               : '10.1016/j.advengsoft.2017.01.004',
    'gsa'               : '10.1016/j.ins.2009.03.004',
    'gwo'               : '10.1016/j.advengsoft.2013.12.007',
    'ggo'               : '10.1016/j.eswa.2023.122147',
    'go_growth'         : '10.1016/j.knosys.2022.110206',
    'hsa'               : '10.1177/003754970107600201',
    'hho'               : '10.1016/j.future.2019.02.028',
    'hbo'               : '10.1016/j.eswa.2020.113702',
    'hgso'              : '10.1016/j.future.2019.07.015',
    'hiking_oa'         : '10.1016/j.knosys.2024.111880',
    'hc'                : '10.1007/978-3-540-75256-1_52',
    'ho_hippo'          : '10.1038/s41598-024-55040-6',
    'horse_oa'          : '10.1016/j.knosys.2020.106711',
    'hco'               : '10.1038/s41598-022-25031-6',
    'heoa'              : '10.1016/j.eswa.2023.122638',
    'hgs'               : '10.1016/j.eswa.2021.114864',
    'hus'               : '10.1109/ICSCCW.2009.5379451',
    'hba'               : '10.48550/arXiv.1303.6310',
    'hsaba'             : '10.1155/2014/709738',
    'ica'               : '10.1109/CEC.2007.4425083',
    'i_gwo'             : '10.1016/j.eswa.2020.113917',
    'ilshade'           : '10.1109/CEC.2016.7744312',
    'imode'             : '10.1109/CEC48606.2020.9185577',
    'i_woa'             : '10.1016/j.jcde.2019.02.002',
    'iwo'               : '10.1016/j.ecoinf.2006.07.003',
    'ivya'              : '10.1016/j.knosys.2024.111850',
    'jy'                : '10.5267/j.ijiec.2015.8.004',
    'jso'               : '10.1016/j.amc.2020.125535',
    'kma'               : '10.1016/j.asoc.2022.108043',
    'kha'               : '10.1016/j.asoc.2016.08.041',
    'liwo'              : '10.1109/ACCESS.2024.3390670',
    'lco'               : '10.1007/s00500-019-04443-z',
    'l2smea'            : '10.1109/TEVC.2024.3354543',
    'loa'               : '10.1016/j.jcde.2015.06.003',
    'lca'               : '10.1016/j.asoc.2023.111039',
    'lpo'               : '10.1016/j.cma.2023.116582',
    'lfd'               : '10.1016/j.engappai.2020.103731',
    'mrfo'              : '10.1016/j.engappai.2019.103300',
    'mshoa'             : '10.3390/math13091500',
    'mpa'               : '10.1016/j.eswa.2020.113377',
    'memetic_a'         : '10.1162/evco.1991.1.1.67',
    'mso'               : '10.1016/j.advengsoft.2025.103883',
    'mbo'               : '10.1007/s00521-015-1923-y',
    'mke'               : '10.1016/j.knosys.2016.01.009',
    'moss_go'           : '10.1007/s10489-024-05673-7',
    'mvpa'              : '10.1007/s12351-017-0307-5',
    'mfa'               : '10.1016/j.knosys.2015.07.006',
    'msa_e'             : '10.1007/s12293-016-0212-3',
    'mgo'               : '10.1016/j.advengsoft.2022.103282',
    'misaco'            : '10.1109/TCYB.2020.3035521',
    'mvo'               : '10.1007/s00521-015-1870-7',
    'mfea'              : '10.1109/TEVC.2015.2458037',
    'mfea2'             : '10.1109/TEVC.2019.2904771',
    'mts'               : '10.5555/1689599.1689856',
    'samso'             : '10.1109/TCYB.2019.2950169',
    'nmra'              : '10.1007/s00521-017-3287-8',
    'nmm'               : '10.1093/comjnl/7.4.308',
    'nndrea_so'         : '10.1109/TEVC.2024.3378530',
    'noa'               : '10.1007/s11227-023-05579-4',
    'ngo'               : '10.1109/ACCESS.2021.3133286',
    'nro'               : '10.1109/ACCESS.2019.2918406',
    'nca'               : '10.1007/s00500-023-08925-z',
    'ofa'               : '10.1016/j.asoc.2017.01.006',
    'ooa'               : '10.3389/fmech.2022.1126450',
    'plba'              : '10.3390/s21134389',
    'pcx'               : '10.1109/CEC.2004.1331141',
    'parrot_o'          : '10.1016/j.heliyon.2024.e27743',
    'pso'               : '10.1109/ICNN.1995.488968',
    'pfa'               : '10.1016/j.asoc.2019.03.012',
    'poa'               : '10.3390/s22030855',
    'pko'               : '10.1007/s00521-024-09679-3',
    'plo'               : '10.1016/j.neucom.2024.128427',
    'political_o'       : '10.1016/j.knosys.2020.106376',
    'pro'               : '10.1016/j.engappai.2019.06.016',
    'pbil'              : '10.5555/865146',
    'pdo'               : '10.1007/s00521-022-07530-5',
    'pss'               : '10.1007/s00500-020-05274-3',
    'puma_o'            : '10.1016/j.knosys.2024.111257',
    'qio'               : '10.1016/j.cma.2023.116446',
    'qsa'               : '10.1007/s12652-020-02849-4',
    'random_s'          : '10.1080/01621459.1953.10501200',
    'rso'               : '10.1007/s12652-020-02073-6',
    'rbmo'              : '10.1007/s10462-024-10894-0',
    'roa'               : '10.1016/j.eswa.2021.115665',
    'rsa'               : '10.1016/j.eswa.2021.116158',
    'rime'              : '10.1016/j.neucom.2023.02.010',
    'run'               : '10.1016/j.eswa.2021.115079',
    'rfo'               : '10.1007/s10586-024-04823-3',
    'sfo'               : '10.1016/j.engappai.2019.01.001',
    'ssa'               : '10.1016/j.advengsoft.2017.07.002',
    'sade_sammon'       : '10.1109/TEVC.2016.2590750',
    'scso'              : '10.1007/s00366-022-01604-x',
    'sbo'               : '10.1016/j.engappai.2017.01.006',
    'slo'               : '10.14569/IJACSA.2019.0100548',
    'soa'               : '10.1016/j.knosys.2018.11.024',
    'seaho'             : '10.1007/s10489-022-03994-3',
    'saro'              : '10.1155/2019/2482543',
    'ssio_rl'           : '10.1109/JAS.2025.125018',
    'sboa'              : '10.1007/s10462-024-10902-3',
    'saba'              : '10.1155/2014/709738',
    'jde'               : '10.1109/TEVC.2006.872133',
    'sqp'               : '10.1017/S0962492900002518',
    'serval_oa'         : '10.3390/biomimetics7040204',
    'srsr'              : '10.1016/j.asoc.2017.02.028',
    'sto'               : '10.1109/ACCESS.2022.3229964',
    'sa'                : '10.1126/science.220.4598.671',
    'sine_cosine_a'     : '10.1016/j.knosys.2015.12.022',
    'scho'              : '10.1016/j.knosys.2023.111081',
    'sma'               : '10.1016/j.future.2020.03.055',
    'so_snake'          : '10.1016/j.knosys.2022.108320',
    'snow_oa'           : '10.1016/j.eswa.2023.120069',
    'ssdo'              : '10.1007/s00521-019-04159-z',
    'sspider_a'         : '10.1016/j.asoc.2015.02.014',
    'sso'               : '10.1016/j.eswa.2013.05.041',
    'sparrow_sa'        : '10.1080/21642583.2019.1708830',
    'smo'               : '10.1007/s12293-013-0128-0',
    'shio'              : '10.1016/j.advengsoft.2017.05.014',
    'sho'               : '10.1016/j.advengsoft.2017.05.014',
    'squirrel_sa'       : '10.1016/j.swevo.2018.02.013',
    'soo'               : '10.3390/math11112536',
    'sfoa'              : '10.1016/j.swevo.2023.101262',
    'sd'                : '10.1006/hmat.1996.2146',
    'spbo'              : '10.1016/j.advengsoft.2020.102804',
    'shade'             : '10.1109/CEC.2014.6900380',
    'superb_foa'        : '10.1007/s10586-024-04638-2',
    'supply_do'         : '10.1109/ACCESS.2019.2919408',
    'sacc_eam2'         : '10.1109/CEC.2019.8790061',
    'sacoso'            : '10.1109/TEVC.2017.2674885',
    'sade_amss'         : '10.1109/TEVC.2022.3168745',
    'sade_atdsc'        : '10.1109/SSCI51031.2022.10022105',
    'sapo'              : '10.1007/978-3-031-70085-9_22',
    'sos'               : '10.1016/j.compstruc.2014.03.007',
    'ts'                : '10.1287/ijoc.1.3.190',
    'tdo'               : '10.1109/ACCESS.2022.3151642',
    'tlbo'              : '10.1016/j.cad.2010.12.015',
    'toa'               : '10.1007/s13042-021-01432-3',
    'tlco'              : '10.1016/j.eswa.2022.119211',
    'thro'              : '10.1007/s10462-025-11269-9',
    'toc'               : '10.1016/j.eswa.2023.120701',
    'tpo'               : '10.1080/0305215X.2017.1305421',
    'ttao'              : '10.1016/j.eswa.2023.121744',
    'two'               : '10.1016/j.procs.2020.03.063',
    'tso'               : '10.1155/2021/9210050',
    'tsa'               : '10.1016/j.engappai.2020.103541',
    'vcs'               : '10.1016/j.advengsoft.2015.11.004',
    'waoa'              : '10.1038/s41598-023-35863-5',
    'warso'             : '10.1007/s11831-022-09822-0',
    'wca'               : '10.1016/j.compstruc.2012.07.010',
    'wutp'              : '10.1007/s00521-025-11059-6',
    'wo_wave'           : '10.1016/j.knosys.2021.107760',
    'info'              : '10.1016/j.eswa.2022.116516',
    'woa'               : '10.1016/j.advengsoft.2016.01.008',
    'wso'               : '10.1016/j.knosys.2022.108457',
    'who'               : '10.3233/JIFS-190495',
    'wdo'               : '10.1109/APS.2010.5562213',
    'ydse'              : '10.1016/j.cma.2022.115652',
    'zoa'               : '10.1109/ACCESS.2022.3172789',
}


def _table_id_for_registry_id(registry_id: str) -> str:
    """Return the curated table ID corresponding to a registry key."""
    return _REGISTRY_ALIASES.get(registry_id, registry_id)


def _set_capability(engine_cls: type[BaseEngine], attr_name: str, value: bool) -> None:
    """Set an optional capability flag when the engine profile exposes it."""
    capabilities = getattr(engine_cls, "capabilities", None)
    if capabilities is None or not hasattr(capabilities, attr_name):
        return
    try:
        setattr(capabilities, attr_name, value)
    except (AttributeError, TypeError):
        # Some projects may expose read-only/frozen capability profiles.
        # In that case, the engine's native declaration remains authoritative.
        return


def _set_reference_doi(engine_cls: type[BaseEngine], doi: str) -> None:
    """Attach DOI metadata without discarding existing reference fields."""
    reference = dict(getattr(engine_cls, "_REFERENCE", {}) or {})
    reference["doi"] = doi
    engine_cls._REFERENCE = reference


for _registry_id, _engine_cls in REGISTRY.items():
    _table_id = _table_id_for_registry_id(_registry_id)
    if _table_id not in _TABLE_ALGORITHM_IDS:
        continue

    # Always synchronize known boolean flags with the table, including False.
    _set_capability(
        _engine_cls,
        "supports_candidate_injection",
        _table_id in _INJECTION_ENABLED,
    )
    _set_capability(_engine_cls, "supports_restart", _table_id in _RESTART_ENABLED)
    _set_capability(_engine_cls, "supports_snapshot_fit", _table_id in _SNAPSHOT_FIT_ENABLED)
    _set_capability(_engine_cls, "is_population_based", _table_id in _POPULATION_BASED)

    if _table_id in _ALGORITHM_DOIS:
        _set_reference_doi(_engine_cls, _ALGORITHM_DOIS[_table_id])
