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

from .aaa                import AAAEngine
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
from .aho                import AHOEngine
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
from .aso_atom           import ASOAtomEngine
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
from .cddo_child         import CDDOChildEngine
from .cdo                import CDOEngine
from .cdo_chernobyl      import CDOChornobylEngine
from .cem                import CEMEngine
from .ceo_cosmic         import CEOCosmicEngine
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
from .deo_dolphin        import DEODolphinEngine
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
from .foa_fossa          import FOAFossaEngine
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
from .gja                import GJAEngine
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
from .gso_glider_snake   import GSOGliderSnakeEngine
from .gto                import GTOEngine
from .gwo                import GWOEngine
from .hba                import HBAEngine
from .hba_honey          import HBAHoneyEngine
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
from .iagwo              import IAGWOEngine
from .ica                import ICAEngine
from .ikoa               import IKOAEngine
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
from .loa_lyrebird       import LOALyrebirdEngine
from .lpo                import LPOEngine
from .lshade_cnepsin     import LSHADECnEpSinEngine
from .lso_spectrum       import LSOSpectrumEngine
from .mbo                import MBOEngine
from .memetic_a          import MEMETIC_AEngine
from .mfa                import MFAEngine
from .mfea               import MFEAEngine
from .mfea2              import MFEA2Engine
from .mgo                import MGOEngine
from .mgoa_market        import MGOAMarketEngine
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
from .nwoa               import NWOAEngine
from .ofa                import OFAEngine
from .ooa                import OOAEngine
from .parrot_o           import ParrotOEngine
from .pbil               import PBILEngine
from .pcx                import PCXEngine
from .pdo                import PDOEngine
from .pfa                import PFAEngine
from .pfa_polar_fox      import PFAPolarFoxEngine
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
from .shio_success       import SHIOSuccessEngine
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
    AAAEngine, ABCOEngine, ACGWOEngine, ACOEngine, ACOREngine, AEFAEngine, AEOEngine, AESSPSOEngine,
    AFSAEngine, AFTEngine, AGTOEngine, AHAEngine, AHOEngine, ALAEngine, ALOEngine, AOAEngine,
    AOEngine, APOEngine, ARCHOAEngine, AROEngine, ARSEngine, ASOAtomEngine, ASOEngine, AVOAEngine,
    AdamEngine, ArtemisininOEngine, AutoVEngine, BATAEngine, BBOAEngine, BBOEngine, BCOEngine,
    BEAEngine, BESEngine, BFGSEngine, BFOEngine, BKAEngine, BMOEngine, BOAEngine, BROEngine,
    BSPGAEngine, BSAEngine, BSOEngine, BWOEngine, BonobOEngine, CAEngine, CAT_SOEngine, CDDOChildEngine,
    CDDOEngine, CDOChornobylEngine, CDOEngine, CEOCosmicEngine, CEMEngine, CGOEngine, CHICKEN_SOEngine, CHIOEngine,
    CIRCLESAEngine, CLONALGEngine, CMAESEngine, COAEngine, COATI_OAEngine, COCKROACH_SOEngine, COOTEngine, CROEngine,
    CSAEngine, CSBOEngine, CSOEngine, CUCKOO_SEngine, CamelEngine, CapSAEngine, ChOAEngine, ChameleonSAEngine,
    CrayfishOAEngine, DAEngine, DBOEngine, DDAOEngine, DEEngine, DEODolphinEngine, DFOEngine, DMOAEngine,
    DOAEngine, DVBAEngine, DSOEngine, DandelionOEngine, EAOEngine, ECOEngine, ECPOEngine, EDOEngine,
    EFOEngine, EGOEngine, EHOEngine, EOEngine, EOAEngine, EPCEngine, EPEngine, ESCEngine,
    ESEngine, ESOAEngine, ESOEngine, ETOEngine, EVOEngine, EcologicalCycleOEngine, ElkHOEngine, FATAEngine,
    FBIOEngine, FDAEngine, FDOEngine, FEPEngine, FFAEngine, FFOEngine, FIREFLY_AEngine, FLAEngine,
    FOAEngine, FOAFossaEngine, FOXEngine, FPAEngine, FRCGEngine, FROFIEngine, FSSEngine, FloodAEngine,
    FWAEngine, GAEngine, GBOEngine, GCOEngine, GEAEngine, GGOEngine, GJAEngine, GJOEngine,
    GKSOEngine, GMOEngine, GNDOEngine, GOAEngine, GOGrowthEngine, GPSOEngine, GSAEngine, GSKAEngine,
    GSOEngine, GSOGliderSnakeEngine, GTOEngine, GazelleOAEngine, GWOEngine, HBAEngine, HBAHoneyEngine, HBOEngine,
    HCEngine, HCOEngine, HDEEngine, HEOAEngine, HGSEngine, HGSOEngine, HHOEngine, HSAEngine,
    HSABAEngine, HUSEngine, HikingOAEngine, HippoEngine, HorseOAEngine, IAGWOEngine, ICAEngine, IKOAEngine,
    ILSHADEEngine, IMODEEngine, INFOEngine, IVYAEngine, IWOEngine, I_GWOEngine, I_WOAEngine, JDEEngine,
    JSOEngine, JYEngine, KHAEngine, KMAEngine, L2SMEAEngine, LCAEngine, LCOEngine, LFDEngine,
    LOAEngine, LOALyrebirdEngine, LPOEngine, LSHADECnEpSinEngine, LSOSpectrumEngine, LiWOEngine, MBOEngine, MEMETIC_AEngine,
    MFEA2Engine, MFEAEngine, MFAEngine, MGOAMarketEngine, MGOEngine, MKEEngine, MPAEngine, MRFOEngine,
    MSAEngine, MSHOAEngine, MSOEngine, MTSEngine, MVOEngine, MVPAEngine, MiSACOEngine, MossGOEngine,
    NCAEngine, NGOEngine, NMMEngine, NMRAEngine, NNDREASOEngine, NOAEngine, NROEngine, NWOAEngine,
    OFAEngine, OOAEngine, PBILEngine, PCXEngine, PDOEngine, PFAEngine, PFAPolarFoxEngine, PKOEngine,
    PLBAEngine, PLOEngine, POAEngine, PROEngine, PSOEngine, PSSEngine, ParrotOEngine, PoliticalOEngine,
    PumaOEngine, QIOEngine, QSAEngine, RANDOM_SEngine, RBMOEngine, RFOEngine, RIMEEngine, RMSPropEngine,
    ROAEngine, RSAEngine, RSOEngine, RUNEngine, SADEAMSSEngine, SADEATDSCEngine, SADESammonEngine, SABAEngine,
    SACCEAMIIEngine, SACOSOEngine, SAMSOEngine, SAPOEngine, SAROEngine, SAEngine, SBOAEngine, SBOEngine,
    SCHOEngine, SCSoEngine, SDEngine, SERVALOAEngine, SFOAEngine, SFOEngine, SHADEEngine, SHIOSuccessEngine,
    SHIOEngine, SHOEngine, SINE_COSINE_AEngine, SLOEngine, SMAEngine, SMOEngine, SOAEngine, SOOEngine,
    SOSEngine, SPBOEngine, SQPEngine, SRSREngine, SSAEngine, SSDOEngine, SSIORLEngine, SSOEngine,
    SSPIDERAEngine, STOEngine, SeaHOEngine, SnakeOptimizerEngine, SnowOAEngine, SparrowSAEngine, SquirrelSAEngine, SuperbFOAEngine,
    SupplyDOEngine, TDOEngine, THROEngine, TLBOEngine, TLCOEngine, TOAEngine, TOCEngine, TPOEngine,
    TSAEngine, TSEngine, TSOEngine, TTAOEngine, TWOEngine, VCSEngine, WAOAEngine, WARSOEngine,
    WCAEngine, WDOEngine, WHOEngine, WOAEngine, WSOEngine, WUTPEngine, WaveOptEngine, YDSEEngine,
    ZOAEngine,
)

REGISTRY: dict[str, type[BaseEngine]] = {
    cls.algorithm_id: cls
    for cls in _ENGINE_CLASSES
}

_REGISTRY_ALIASES: dict[str, str] = {}

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
    'aaa', 'abco', 'acgwo', 'aco', 'acor', 'adam', 'aefa', 'aeo', 'aesspso', 'afsa', 'aft', 'agto', 'aha',
    'aho', 'ala', 'alo', 'ao', 'aoa', 'apo', 'arch_oa', 'aro', 'ars', 'artemisinin_o', 'aso', 'aso_atom',
    'autov', 'avoa', 'bat_a', 'bbo', 'bboa', 'bco', 'bea', 'bes', 'bfgs', 'bfo', 'bka', 'bmo', 'boa', 'bono',
    'bro', 'bsa', 'bso', 'bspga', 'bwo', 'ca', 'camel', 'capsa', 'cat_so', 'cddo', 'cddo_child', 'cdo',
    'cdo_chernobyl', 'cem', 'ceo_cosmic', 'cgo', 'chameleon_sa', 'chicken_so', 'chio', 'choa', 'circle_sa',
    'clonalg', 'cmaes', 'coa', 'coati_oa', 'cockroach_so', 'coot', 'crayfish_oa', 'cro', 'csa', 'csbo',
    'cso', 'cuckoo_s', 'da', 'dbo', 'ddao', 'de', 'deo_dolphin', 'dfo', 'dmoa', 'do_dandelion', 'doa', 'dso',
    'dvba', 'eao', 'eco', 'ecological_cycle_o', 'ecpo', 'edo', 'efo', 'ego', 'eho', 'elk_ho', 'eo', 'eoa',
    'ep', 'epc', 'es', 'esc', 'eso', 'esoa', 'eto', 'evo', 'fata', 'fbio', 'fda', 'fdo', 'fep', 'ffa', 'ffo',
    'firefly_a', 'fla', 'flood_a', 'foa', 'foa_fossa', 'fox', 'fpa', 'frcg', 'frofi', 'fss', 'fwa', 'ga',
    'gazelle_oa', 'gbo', 'gco', 'gea', 'ggo', 'gja', 'gjo', 'gkso', 'gmo', 'gndo', 'go_growth', 'goa',
    'gpso', 'gsa', 'gska', 'gso', 'gso_glider_snake',  'gto', 'gwo', 'hba', 'hba_honey', 'hbo', 'hc', 'hco', 'hde', 'heoa', 'hgs',
    'hgso', 'hho', 'hiking_oa', 'ho_hippo', 'horse_oa', 'hsa', 'hsaba', 'hus', 'iagwo',  'i_gwo', 'i_woa', 'ica',
    'ikoa', 'ilshade', 'imode', 'info', 'ivya', 'iwo', 'jde', 'jso', 'jy', 'kha', 'kma', 'l2smea', 'lca',
    'lco', 'lfd', 'liwo', 'loa', 'loa_lyrebird', 'lpo', 'lshade_cnepsin', 'lso_spectrum', 'mbo', 'memetic_a',
    'mfa', 'mfea', 'mfea2', 'mgo', 'mgoa_market', 'misaco', 'mke', 'moss_go', 'mpa', 'mrfo', 'msa_e',
    'mshoa', 'mso', 'mts', 'mvo', 'mvpa', 'nca', 'ngo', 'nmm', 'nmra', 'nndrea_so', 'noa', 'nro', 'nwoa',
    'ofa', 'ooa', 'parrot_o', 'pbil', 'pcx', 'pdo', 'pfa', 'pfa_polar_fox', 'pko', 'plba', 'plo', 'poa', 'political_o', 'pro',
    'pso', 'pss', 'puma_o', 'qio', 'qsa', 'random_s', 'rbmo', 'rfo', 'rime', 'rmsprop', 'roa', 'rsa', 'rso',
    'run', 'sa', 'saba', 'sacc_eam2', 'sacoso', 'sade_amss', 'sade_atdsc', 'sade_sammon', 'samso', 'sapo',
    'saro', 'sbo', 'sboa', 'scho', 'scso', 'sd', 'seaho', 'serval_oa', 'sfo', 'sfoa', 'shade', 'shio',
    'shio_success', 'sho', 'sine_cosine_a', 'slo', 'sma', 'smo', 'snow_oa', 'so_snake', 'soa', 'soo', 'sos',
    'sparrow_sa', 'spbo', 'sqp', 'squirrel_sa', 'srsr', 'ssa', 'ssdo', 'ssio_rl', 'sso', 'sspider_a', 'sto',
    'superb_foa', 'supply_do', 'tdo', 'thro', 'tlbo', 'tlco', 'toa', 'toc', 'tpo', 'ts', 'tsa', 'tso',
    'ttao', 'two', 'vcs', 'waoa', 'warso', 'wca', 'wdo', 'who', 'wo_wave', 'woa', 'wso', 'wutp', 'ydse',
    'zoa',
}

# Algorithms marked as population-based in the table.
_POPULATION_BASED: set[str] = {
    'aaa', 'abco', 'acgwo', 'aco', 'acor', 'aefa', 'aeo', 'aesspso', 'afsa', 'aft', 'agto', 'aha', 'aho', 'ala',
    'alo', 'ao', 'aoa', 'apo', 'arch_oa', 'aro', 'ars', 'artemisinin_o', 'aso', 'aso_atom', 'autov', 'avoa',
    'bat_a', 'bbo', 'bboa', 'bco', 'bea', 'bes', 'bfo', 'bka', 'bmo', 'boa', 'bono', 'bro', 'bsa', 'bso', 'bspga',
    'bwo', 'ca', 'camel', 'capsa', 'cat_so', 'cddo', 'cddo_child', 'cdo', 'cdo_chernobyl', 'cem', 'ceo_cosmic',
    'cgo', 'chameleon_sa', 'chicken_so', 'chio', 'choa', 'circle_sa', 'clonalg', 'cmaes', 'coa', 'coati_oa',
    'cockroach_so', 'coot', 'crayfish_oa', 'cro', 'csa', 'csbo', 'cso', 'cuckoo_s', 'da', 'dbo', 'ddao', 'de',
    'deo_dolphin', 'dfo', 'dmoa', 'do_dandelion', 'doa', 'dso', 'dvba', 'eao', 'eco', 'ecological_cycle_o',
    'ecpo', 'edo', 'efo', 'ego', 'eho', 'elk_ho', 'eo', 'eoa', 'ep', 'epc', 'es', 'esc', 'eso', 'esoa', 'eto',
    'evo', 'fata', 'fbio', 'fda', 'fdo', 'fep', 'ffa', 'ffo', 'firefly_a', 'fla', 'flood_a', 'foa', 'foa_fossa',
    'fox', 'fpa', 'frofi', 'fss', 'fwa', 'ga', 'gazelle_oa', 'gbo', 'gco', 'gea', 'ggo', 'gja', 'gjo', 'gkso',
    'gmo', 'gndo', 'go_growth', 'goa', 'gpso', 'gsa', 'gska', 'gso', 'gso_glider_snake', 'gto', 'gwo', 'hba', 'hba_honey', 'hbo',
    'hco', 'hde', 'heoa', 'hgs', 'hgso', 'hho', 'hiking_oa', 'ho_hippo', 'horse_oa', 'hsa', 'hsaba', 'hus', 'iagwo',
    'i_gwo', 'i_woa', 'ica', 'ikoa', 'ilshade', 'imode', 'info', 'ivya', 'iwo', 'jde', 'jso', 'jy', 'kha', 'kma',
    'l2smea', 'lca', 'lco', 'lfd', 'liwo', 'loa', 'loa_lyrebird', 'lpo', 'lshade_cnepsin', 'lso_spectrum', 'mbo',
    'memetic_a', 'mfa', 'mfea', 'mfea2', 'mgo', 'mgoa_market', 'misaco', 'mke', 'moss_go', 'mpa', 'mrfo', 'msa_e',
    'mshoa', 'mso', 'mts', 'mvo', 'mvpa', 'nca', 'ngo', 'nmm', 'nmra', 'nndrea_so', 'noa', 'nro', 'nwoa', 'ofa',
    'ooa', 'parrot_o', 'pcx', 'pdo', 'pfa', 'pfa_polar_fox', 'pko', 'plba', 'plo', 'poa', 'political_o', 'pro', 'pso', 'pss',
    'puma_o', 'qio', 'qsa', 'random_s', 'rbmo', 'rfo', 'rime', 'roa', 'rsa', 'rso', 'run', 'saba', 'sacc_eam2',
    'sacoso', 'sade_amss', 'sade_atdsc', 'sade_sammon', 'samso', 'sapo', 'saro', 'sbo', 'sboa', 'scho', 'scso',
    'seaho', 'serval_oa', 'sfo', 'sfoa', 'shade', 'shio', 'shio_success', 'sho', 'sine_cosine_a', 'slo', 'sma',
    'smo', 'snow_oa', 'so_snake', 'soa', 'soo', 'sos', 'sparrow_sa', 'spbo', 'squirrel_sa', 'srsr', 'ssa', 'ssdo',
    'ssio_rl', 'sso', 'sspider_a', 'sto', 'superb_foa', 'supply_do', 'tdo', 'thro', 'tlbo', 'tlco', 'toa', 'toc',
    'tpo', 'tsa', 'tso', 'ttao', 'two', 'vcs', 'waoa', 'warso', 'wca', 'wdo', 'who', 'wo_wave', 'woa', 'wso',
    'wutp', 'ydse', 'zoa'
}

# Algorithms marked as supporting native candidate injection in the table.
_INJECTION_ENABLED: set[str] = {
    'abco', 'acgwo', 'acor', 'aeo', 'aefa', 'aesspso', 'afsa', 'aft', 'agto', 'aha', 'aho', 'ala',
    'alo', 'ao', 'aoa', 'apo', 'arch_oa', 'aro', 'ars', 'artemisinin_o', 'aso', 'aso_atom', 'autov',
    'avoa', 'bat_a', 'bbo', 'bboa', 'bco', 'bea', 'bes', 'bfo', 'bka', 'bmo',
    'boa', 'bono', 'bro', 'bsa', 'bso', 'bspga', 'bwo', 'ca', 'camel', 'capsa', 'cat_so', 'cddo',
    'cddo_child', 'cdo', 'cdo_chernobyl', 'cem', 'ceo_cosmic', 'cgo', 'chameleon_sa', 'chio', 'choa',
    'circle_sa', 'clonalg', 'cmaes', 'coa', 'coati_oa', 'cockroach_so', 'coot', 'cro', 'csa', 'csbo',
    'cso', 'cuckoo_s', 'da', 'dbo', 'ddao', 'de', 'deo_dolphin', 'dfo', 'dmoa', 'do_dandelion', 'doa',
    'dso', 'dvba', 'eao', 'eco', 'ecological_cycle_o', 'ecpo', 'edo', 'efo', 'ego', 'eho', 'elk_ho',
    'eo', 'eoa', 'ep', 'epc', 'es', 'esc', 'eso', 'esoa', 'eto', 'evo', 'fata', 'fbio', 'fda', 'fdo',
    'fep', 'ffa', 'ffo', 'firefly_a', 'fla', 'flood_a', 'foa', 'foa_fossa', 'fox', 'fpa', 'frofi',
    'fss', 'fwa', 'ga', 'gazelle_oa', 'gbo', 'gco', 'gea', 'ggo', 'gja', 'gjo', 'gkso', 'gmo', 'gndo',
    'go_growth', 'goa', 'gpso', 'gsa', 'gska', 'gso', 'gto', 'gwo', 'hba', 'hba_honey', 'hbo', 'hco',
    'hde', 'heoa', 'hgs', 'hgso', 'hho', 'hiking_oa', 'ho_hippo', 'horse_oa', 'hsaba', 'hus', 'i_gwo',
    'i_woa', 'ica', 'ikoa', 'ilshade', 'imode', 'info', 'ivya', 'iwo', 'jde', 'jso', 'jy', 'kma',
    'l2smea', 'lca', 'lco', 'lfd', 'liwo', 'loa', 'loa_lyrebird', 'lpo', 'lshade_cnepsin',
    'lso_spectrum', 'mbo', 'memetic_a', 'mfa', 'mfea', 'mfea2', 'mgo', 'mgoa_market', 'misaco',
    'mke', 'moss_go', 'mpa', 'mrfo', 'msa_e', 'mshoa', 'mso', 'mts', 'mvo', 'mvpa', 'nca', 'ngo',
    'nmm', 'nmra', 'nndrea_so', 'noa', 'nro', 'nwoa', 'ofa', 'ooa', 'parrot_o', 'pcx', 'pdo', 'pfa',
    'pko', 'plba', 'plo', 'poa', 'political_o', 'pro', 'pso', 'pss', 'puma_o', 'qio', 'qsa',
    'random_s', 'rbmo', 'rfo', 'rime', 'roa', 'rsa', 'rso', 'run', 'sa', 'saba', 'sacc_eam2',
    'sacoso', 'sade_amss', 'sade_atdsc', 'sade_sammon', 'samso', 'sapo', 'saro', 'sbo', 'sboa',
    'scho', 'scso', 'seaho', 'serval_oa', 'sfo', 'sfoa', 'shade', 'shio', 'shio_success', 'sho',
    'sine_cosine_a', 'slo', 'sma', 'smo', 'snow_oa', 'so_snake', 'soa', 'soo', 'sos', 'sparrow_sa',
    'spbo', 'squirrel_sa', 'srsr', 'ssa', 'ssdo', 'ssio_rl', 'sso', 'sspider_a', 'sto', 'superb_foa',
    'supply_do', 'tdo', 'thro', 'tlbo', 'tlco', 'toa', 'toc', 'tpo', 'tsa', 'tso', 'ttao', 'two',
    'vcs', 'waoa', 'warso', 'wca', 'wdo', 'who', 'wo_wave', 'woa', 'wso', 'wutp', 'ydse', 'zoa',
}

# Algorithms marked as supporting restart in the table.
_RESTART_ENABLED: set[str] = {
    'sa',
}

# Algorithms marked as snapshot-fit compatible in the table.
_SNAPSHOT_FIT_ENABLED: set[str] = {
    'abco', 'acgwo', 'aco', 'acor', 'aeo', 'aefa', 'aesspso', 'afsa', 'aft', 'agto', 'aha', 'aho', 'ala',
    'alo', 'ao', 'aoa', 'apo', 'arch_oa', 'aro', 'ars', 'artemisinin_o', 'aso', 'aso_atom', 'autov',
    'avoa', 'bat_a', 'bbo', 'bboa', 'bco', 'bea', 'bes', 'bfo', 'bka', 'bmo',
    'boa', 'bono', 'bro', 'bsa', 'bso', 'bspga', 'bwo', 'ca', 'camel', 'capsa', 'cat_so', 'cddo',
    'cddo_child', 'cdo', 'cdo_chernobyl', 'cem', 'ceo_cosmic', 'cgo', 'chameleon_sa', 'chicken_so',
    'chio', 'choa', 'circle_sa', 'clonalg', 'cmaes', 'coa', 'coati_oa', 'cockroach_so', 'coot', 'cro',
    'csa', 'csbo', 'cso', 'cuckoo_s', 'da', 'dbo', 'ddao', 'de', 'deo_dolphin', 'dfo', 'dmoa',
    'do_dandelion', 'doa', 'dso', 'dvba', 'eao', 'eco', 'ecological_cycle_o', 'ecpo', 'edo', 'efo',
    'ego', 'eho', 'elk_ho', 'eo', 'eoa', 'ep', 'epc', 'es', 'esc', 'eso', 'esoa', 'eto', 'evo',
    'fata', 'fbio', 'fda', 'fdo', 'fep', 'ffa', 'ffo', 'firefly_a', 'fla', 'flood_a', 'foa',
    'foa_fossa', 'fox', 'fpa', 'frofi', 'fss', 'fwa', 'ga', 'gazelle_oa', 'gbo', 'gco', 'gea',
    'ggo', 'gja', 'gjo', 'gkso', 'gmo', 'gndo', 'go_growth', 'goa', 'gpso', 'gsa', 'gska', 'gso',
    'gto', 'gwo', 'hba', 'hba_honey', 'hbo', 'hco', 'hde', 'heoa', 'hgs', 'hgso', 'hho',
    'hiking_oa', 'ho_hippo', 'horse_oa', 'hsa', 'hsaba', 'hus', 'i_gwo', 'i_woa', 'ica', 'ikoa',
    'ilshade', 'imode', 'info', 'ivya', 'iwo', 'jde', 'jso', 'jy', 'kha', 'kma', 'l2smea', 'lca',
    'lco', 'lfd', 'liwo', 'loa', 'loa_lyrebird', 'lpo', 'lshade_cnepsin', 'lso_spectrum', 'mbo',
    'memetic_a', 'mfa', 'mfea', 'mfea2', 'mgo', 'mgoa_market', 'misaco', 'mke', 'moss_go', 'mpa',
    'mrfo', 'msa_e', 'mshoa', 'mso', 'mts', 'mvo', 'mvpa', 'nca', 'ngo', 'nmm', 'nmra',
    'nndrea_so', 'noa', 'nro', 'nwoa', 'ofa', 'ooa', 'parrot_o', 'pcx', 'pdo', 'pfa', 'pko',
    'plba', 'plo', 'poa', 'political_o', 'pro', 'pso', 'pss', 'puma_o', 'qio', 'qsa', 'random_s',
    'rbmo', 'rfo', 'rime', 'roa', 'rsa', 'rso', 'run', 'saba', 'sacc_eam2', 'sacoso', 'sade_amss',
    'sade_atdsc', 'sade_sammon', 'samso', 'sapo', 'saro', 'sbo', 'sboa', 'scho', 'scso', 'seaho',
    'serval_oa', 'sfo', 'sfoa', 'shade', 'shio', 'shio_success', 'sho', 'sine_cosine_a', 'slo',
    'sma', 'smo', 'snow_oa', 'so_snake', 'soa', 'soo', 'sos', 'sparrow_sa', 'spbo',
    'squirrel_sa', 'srsr', 'ssa', 'ssdo', 'ssio_rl', 'sso', 'sspider_a', 'sto', 'superb_foa',
    'supply_do', 'tdo', 'thro', 'tlbo', 'tlco', 'toa', 'toc', 'tpo', 'tsa', 'tso', 'ttao', 'two',
    'vcs', 'waoa', 'warso', 'wca', 'wdo', 'who', 'wo_wave', 'woa', 'wso', 'wutp', 'ydse', 'zoa',
}

# Optional descriptive metadata from the table.
_ALGORITHM_NAMES: dict[str, str] = {
    'aaa'               : 'Artificial Algae Algorithm',
    'abco'              : 'Artificial Bee Colony Optimization',
    'acgwo'             : 'Adaptive Chaotic Grey Wolf Optimizer',
    'aco'               : 'Ant Colony Optimization',
    'acor'              : 'Ant Colony Optimization (Continuous)',
    'adam'              : 'Adam (Adaptive Moment Estimation)',
    'aefa'              : 'Artificial Electric Field Algorithm',
    'aeo'               : 'Artificial Ecosystem Optimization',
    'aesspso'           : 'Adaptive Exploration State-Space Particle Swarm Optimization',
    'afsa'              : 'Artificial Fish Swarm Algorithm',
    'aft'               : 'Ali Baba and the Forty Thieves',
    'agto'              : 'Artificial Gorilla Troops Optimizer',
    'aha'               : 'Artificial Hummingbird Algorithm',
    'aho'               : 'Archerfish Hunting Optimizer',
    'ala'               : 'Artificial Lemming Algorithm',
    'alo'               : 'Ant Lion Optimizer',
    'ao'                : 'Aquila Optimizer',
    'aoa'               : 'Arithmetic Optimization Algorithm',
    'apo'               : 'Artificial Protozoa Optimizer',
    'arch_oa'           : 'Archimedes Optimization Algorithm',
    'aro'               : 'Artificial Rabbits Optimization',
    'ars'               : 'Adaptive Random Search',
    'artemisinin_o'     : 'Artemisinin Optimization',
    'aso'               : 'Anarchic Society Optimization',
    'aso_atom'          : 'Atom Search Optimization',
    'autov'             : 'Automated Design of Variation Operators',
    'avoa'              : 'African Vultures Optimization Algorithm',
    'bat_a'             : 'Bat Algorithm',
    'bbo'               : 'Biogeography-Based Optimization',
    'bboa'              : 'Brown-Bear Optimization Algorithm',
    'bco'               : 'Bacterial Chemotaxis Optimizer',
    'bea'               : 'Bees Algorithm',
    'bes'               : 'Bald Eagle Search',
    'bfgs'              : 'BFGS Quasi-Newton Method',
    'bfo'               : 'Bacterial Foraging Optimization',
    'bka'               : 'Black-winged Kite Algorithm',
    'bmo'               : 'Barnacles Mating Optimizer',
    'boa'               : 'Butterfly Optimization Algorithm',
    'bono'              : 'Bonobo Optimizer',
    'bro'               : 'Battle Royale Optimization',
    'bsa'               : 'Bird Swarm Algorithm',
    'bso'               : 'Brain Storm Optimization',
    'bspga'             : 'Binary Space Partition Tree Genetic Algorithm',
    'bwo'               : 'Black Widow Optimization',
    'ca'                : 'Cultural Algorithm',
    'camel'             : 'Camel Algorithm',
    'capsa'             : 'Capuchin Search Algorithm',
    'cat_so'            : 'Cat Swarm Optimization',
    'cddo'              : 'Cheetah Based Optimization',
    'cddo_child'        : 'Child Drawing Development Optimization Algorithm',
    'cdo'               : 'Cheetah Optimizer',
    'cdo_chernobyl'     : 'Chernobyl Disaster Optimizer',
    'cem'               : 'Cross Entropy Method',
    'ceo_cosmic'        : 'Cosmic Evolution Optimization',
    'cgo'               : 'Chaos Game Optimization',
    'chameleon_sa'      : 'Chameleon Swarm Algorithm',
    'chicken_so'        : 'Chicken Swarm Optimization',
    'chio'              : 'Coronavirus Herd Immunity Optimization',
    'choa'              : 'Chimp Optimization Algorithm',
    'circle_sa'         : 'Circle-Based Search Algorithm',
    'clonalg'           : 'Clonal Selection Algorithm',
    'cmaes'             : 'Covariance Matrix Adaptation Evolution Strategy',
    'coa'               : 'Coyote Optimization Algorithm',
    'coati_oa'          : 'Coati Optimization Algorithm',
    'cockroach_so'      : 'Cockroach Swarm Optimization',
    'coot'              : 'COOT Bird Optimization',
    'crayfish_oa'       : 'Crayfish Optimization Algorithm',
    'cro'               : 'Coral Reefs Optimization',
    'csa'               : 'Crow Search Algorithm',
    'csbo'              : 'Circulatory System Based Optimization',
    'cso'               : 'Competitive Swarm Optimizer',
    'cuckoo_s'          : 'Cuckoo Search',
    'da'                : 'Dragonfly Algorithm',
    'dbo'               : 'Dung Beetle Optimizer',
    'ddao'              : 'Dynamic Differential Annealed Optimization',
    'de'                : 'Differential Evolution',
    'deo_dolphin'       : 'Dolphin Echolocation Optimization',
    'dfo'               : 'Dispersive Fly Optimization',
    'dmoa'              : 'Dwarf Mongoose Optimization Algorithm',
    'do_dandelion'      : 'Dandelion Optimizer',
    'doa'               : 'Deer Hunting Optimization Algorithm',
    'dso'               : 'Deep Sleep Optimiser',
    'dvba'              : 'Dynamic Virtual Bats Algorithm',
    'eao'               : 'Enzyme Activity Optimizer',
    'eco'               : 'Educational Competition Optimizer',
    'ecological_cycle_o': 'Ecological Cycle Optimizer',
    'ecpo'              : 'Electric Charged Particles Optimization',
    'edo'               : 'Exponential Distribution Optimizer',
    'efo'               : 'Electromagnetic Field Optimization',
    'ego'               : 'Efficient Global Optimization',
    'eho'               : 'Elephant Herding Optimization',
    'elk_ho'            : 'Elk Herd Optimizer',
    'eo'                : 'Equilibrium Optimizer',
    'eoa'               : 'Earthworm Optimization Algorithm',
    'ep'                : 'Evolutionary Programming',
    'epc'               : 'Emperor Penguin Colony',
    'es'                : 'Evolution Strategy (mu + lambda)',
    'esc'               : 'Escape Algorithm',
    'eso'               : 'Electrical Storm Optimization',
    'esoa'              : 'Egret Swarm Optimization Algorithm',
    'eto'               : 'Exponential-Trigonometric Optimization',
    'evo'               : 'Energy Valley Optimizer',
    'fata'              : 'FATA Geophysics Optimizer',
    'fbio'              : 'Forensic-Based Investigation Optimization',
    'fda'               : 'Flow Direction Algorithm',
    'fdo'               : 'Fitness Dependent Optimizer',
    'fep'               : 'Fast Evolutionary Programming',
    'ffa'               : 'Fruit-Fly Algorithm',
    'ffo'               : 'Fennec Fox Optimizer',
    'firefly_a'         : 'Firefly Algorithm',
    'fla'               : "Fick's Law Algorithm",
    'flood_a'           : 'Flood Algorithm',
    'foa'               : 'Forest Optimization Algorithm',
    'foa_fossa'         : 'Fossa Optimization Algorithm',
    'fox'               : 'Fox Optimizer',
    'fpa'               : 'Flower Pollination Algorithm',
    'frcg'              : 'Fletcher-Reeves Conjugate Gradient',
    'frofi'             : 'Feasibility Rule with Objective Function Information',
    'fss'               : 'Fish School Search',
    'fwa'               : 'Fireworks Algorithm',
    'ga'                : 'Genetic Algorithm',
    'gazelle_oa'        : 'Gazelle Optimization Algorithm',
    'gbo'               : 'Gradient-Based Optimizer',
    'gco'               : 'Germinal Center Optimization',
    'gea'               : 'Geyser Inspired Algorithm',
    'ggo'               : 'Greylag Goose Optimization',
    'gja'               : 'Gekko Japonicus Algorithm',
    'gjo'               : 'Golden Jackal Optimizer',
    'gkso'              : 'Genghis Khan Shark Optimizer',
    'gmo'               : 'Geometric Mean Optimizer',
    'gndo'              : 'Generalized Normal Distribution Optimizer',
    'go_growth'         : 'Growth Optimizer',
    'goa'               : 'Grasshopper Optimization Algorithm',
    'gpso'              : 'Gradient-Based Particle Swarm Optimization',
    'gsa'               : 'Gravitational Search Algorithm',
    'gska'              : 'Gaining-Sharing Knowledge Algorithm',
    'gso'               : 'Glowworm Swarm Optimization',
    'gso_glider_snake'  : 'Glider Snake Optimization',
    'gto'               : 'Giant Trevally Optimizer',
    'gwo'               : 'Grey Wolf Optimizer',
    'hba'               : 'Hybrid Bat Algorithm',
    'hba_honey'         : 'Honey Badger Algorithm',
    'hbo'               : 'Heap-Based Optimizer',
    'hc'                : 'Hill Climb Algorithm',
    'hco'               : 'Human Conception Optimizer',
    'hde'               : 'Differential Evolution MTS',
    'heoa'              : 'Human Evolutionary Optimization Algorithm',
    'hgs'               : 'Hunger Games Search',
    'hgso'              : 'Henry Gas Solubility Optimization',
    'hho'               : 'Harris Hawks Optimization',
    'hiking_oa'         : 'Hiking Optimization Algorithm',
    'ho_hippo'          : 'Hippopotamus Optimization Algorithm',
    'horse_oa'          : 'Horse Herd Optimization Algorithm',
    'hsa'               : 'Harmony Search Algorithm',
    'hsaba'             : 'Hybrid Self-Adaptive Bat Algorithm',
    'hus'               : 'Hunting Search Algorithm',
    'iagwo'             : 'Improved Adaptive Grey Wolf Optimization',
    'i_gwo'             : 'Improved Grey Wolf Optimizer',
    'i_woa'             : 'Improved Whale Optimization Algorithm',
    'ica'               : 'Imperialist Competitive Algorithm',
    'ikoa'              : 'Improved Kepler Optimization Algorithm',
    'ilshade'           : 'Improved L-SHADE',
    'imode'             : 'Improved Multi-Operator Differential Evolution',
    'info'              : 'Weighting and Inertia Random Walk Optimizer',
    'ivya'              : 'Ivy Algorithm',
    'iwo'               : 'Invasive Weed Optimization',
    'jde'               : 'Self-Adaptive Differential Evolution',
    'jso'               : 'Jellyfish Search Optimizer',
    'jy'                : 'Jaya Algorithm',
    'kha'               : 'Krill Herd Algorithm',
    'kma'               : 'Komodo Mlipir Algorithm',
    'l2smea'            : 'Linear Subspace Surrogate Modeling Evolutionary Algorithm',
    'lca'               : 'Liver Cancer Algorithm',
    'lco'               : 'Life Choice-Based Optimizer',
    'lfd'               : 'Lévy Flight Distribution',
    'liwo'              : 'Leaf in Wind Optimization',
    'loa'               : 'Lion Optimization Algorithm',
    'loa_lyrebird'      : 'Lyrebird Optimization Algorithm',
    'lpo'               : 'Lungs Performance-Based Optimization',
    'lshade_cnepsin'    : 'LSHADE-cnEpSin',
    'lso_spectrum'      : 'Light Spectrum Optimizer',
    'mbo'               : 'Monarch Butterfly Optimization',
    'memetic_a'         : 'Memetic Algorithm',
    'mfa'               : 'Moth Flame Algorithm',
    'mfea'              : 'Multifactorial Evolutionary Algorithm',
    'mfea2'             : 'Multifactorial Evolutionary Algorithm II',
    'mgo'               : 'Mountain Gazelle Optimizer',
    'mgoa_market'       : 'Market Game Optimization Algorithm',
    'misaco'            : 'Multi-Surrogate-Assisted Ant Colony Optimization',
    'mke'               : 'Monkey King Evolution V1',
    'moss_go'           : 'Moss Growth Optimization',
    'mpa'               : 'Marine Predators Algorithm',
    'mrfo'              : 'Manta Ray Foraging Optimization',
    'msa_e'             : 'Moth Search Algorithm',
    'mshoa'             : 'Mantis Shrimp Optimization Algorithm',
    'mso'               : 'Mirage-Search Optimizer',
    'mts'               : 'Multiple Trajectory Search',
    'mvo'               : 'Multi-Verse Optimizer',
    'mvpa'              : 'Most Valuable Player Algorithm',
    'nca'               : 'Numeric Crunch Algorithm',
    'ngo'               : 'Northern Goshawk Optimization',
    'nmm'               : 'Nelder-Mead Method',
    'nmra'              : 'Naked Mole-Rat Algorithm',
    'nndrea_so'         : 'Neural Network-Based Dimensionality Reduction Evolutionary Algorithm (SO)',
    'noa'               : 'Nizar Optimization Algorithm',
    'nro'               : 'Nuclear Reaction Optimization',
    'nwoa'              : 'Narwhal Optimizer',
    'ofa'               : 'Optimal Foraging Algorithm',
    'ooa'               : 'Osprey Optimization Algorithm',
    'parrot_o'          : 'Parrot Optimizer',
    'pbil'              : 'Population-Based Incremental Learning',
    'pcx'               : 'Parent-Centric Crossover (G3-PCX style)',
    'pdo'               : 'Prairie Dog Optimization Algorithm',
    'pfa'               : 'Pathfinder Algorithm',
    'pfa_polar_fox'     : 'Polar Fox Optimization',
    'pko'               : 'Pied Kingfisher Optimizer',
    'plba'              : 'Parameter-Free Bat Algorithm',
    'plo'               : 'Polar Lights Optimizer',
    'poa'               : 'Pelican Optimization Algorithm',
    'political_o'       : 'Political Optimizer',
    'pro'               : 'Poor and Rich Optimization Algorithm',
    'pso'               : 'Particle Swarm Optimization',
    'pss'               : 'Pareto Sequential Sampling',
    'puma_o'            : 'Puma Optimizer',
    'qio'               : 'Quadratic Interpolation Optimization',
    'qsa'               : 'Queuing Search Algorithm',
    'random_s'          : 'Random Search',
    'rbmo'              : 'Red-billed Blue Magpie Optimizer',
    'rfo'               : "Rüppell's Fox Optimizer",
    'rime'              : 'RIME-ice Algorithm',
    'rmsprop'           : 'RMSProp',
    'roa'               : 'Remora Optimization Algorithm',
    'rsa'               : 'Reptile Search Algorithm',
    'rso'               : 'Rat Swarm Optimizer',
    'run'               : 'RUNge Kutta Optimizer',
    'sa'                : 'Simulated Annealing',
    'saba'              : 'Self-Adaptive Bat Algorithm',
    'sacc_eam2'         : 'Surrogate-Assisted Cooperative Co-Evolutionary Algorithm of Minamo II',
    'sacoso'            : 'Surrogate-Assisted Cooperative Swarm Optimization',
    'sade_amss'         : 'Surrogate-Assisted DE with Adaptive Multi-Subspace Search',
    'sade_atdsc'        : 'Surrogate-Assisted DE with Adaptive Training Data Selection Criterion',
    'sade_sammon'       : 'Sammon Mapping Assisted Differential Evolution',
    'samso'             : 'Multiswarm-Assisted Expensive Optimization',
    'sapo'              : 'Surrogate-Assisted Partial Optimization',
    'saro'              : 'Search And Rescue Optimization',
    'sbo'               : 'Satin Bowerbird Optimizer',
    'sboa'              : 'Secretary Bird Optimization Algorithm',
    'scho'              : 'Sinh Cosh Optimizer',
    'scso'              : 'Sand Cat Swarm Optimization',
    'sd'                : 'Steepest Descent',
    'seaho'             : 'Seahorse Optimizer',
    'serval_oa'         : 'Serval Optimization Algorithm',
    'sfo'               : 'Sailfish Optimizer',
    'sfoa'              : 'Starfish Optimization Algorithm',
    'shade'             : 'Success-History Adaptive Differential Evolution',
    'shio'              : 'Spotted Hyena Inspired Optimizer',
    'shio_success'      : 'Success History Intelligent Optimizer',
    'sho'               : 'Spotted Hyena Optimizer',
    'sine_cosine_a'     : 'Sine Cosine Algorithm',
    'slo'               : 'Sea Lion Optimization',
    'sma'               : 'Slime Mould Algorithm',
    'smo'               : 'Spider Monkey Optimization',
    'snow_oa'           : 'Snow Ablation Optimizer',
    'so_snake'          : 'Snake Optimizer',
    'soa'               : 'Seagull Optimization Algorithm',
    'soo'               : 'Stellar Oscillator Optimization',
    'sos'               : 'Symbiotic Organisms Search',
    'sparrow_sa'        : 'Sparrow Search Algorithm',
    'spbo'              : 'Student Psychology Based Optimization',
    'sqp'               : 'Sequential Quadratic Programming',
    'squirrel_sa'       : 'Squirrel Search Algorithm',
    'srsr'              : 'Shuffle-based Runner-Root Algorithm',
    'ssa'               : 'Salp Swarm Algorithm',
    'ssdo'              : 'Social Ski-Driver Optimization',
    'ssio_rl'           : 'Search Space Independent Operator Based Deep Reinforcement Learning',
    'sso'               : 'Social Spider Swarm Optimizer',
    'sspider_a'         : 'Social Spider Algorithm',
    'sto'               : 'Siberian Tiger Optimization',
    'superb_foa'        : 'Superb Fairy-wren Optimization Algorithm',
    'supply_do'         : 'Supply-Demand-Based Optimization',
    'tdo'               : 'Tasmanian Devil Optimization',
    'thro'              : 'Tianji Horse Racing Optimizer',
    'tlbo'              : 'Teaching Learning Based Optimization',
    'tlco'              : 'Termite Life Cycle Optimizer',
    'toa'               : 'Teamwork Optimization Algorithm',
    'toc'               : 'Tornado Optimizer with Coriolis Force',
    'tpo'               : 'Tree Physiology Optimization',
    'ts'                : 'Tabu Search',
    'tsa'               : 'Tunicate Swarm Algorithm',
    'tso'               : 'Tuna Swarm Optimization',
    'ttao'              : 'Triangulation Topology Aggregation Optimizer',
    'two'               : 'Tug of War Optimization',
    'vcs'               : 'Virus Colony Search',
    'waoa'              : 'Walrus Optimization Algorithm',
    'warso'             : 'War Strategy Optimization',
    'wca'               : 'Water Cycle Algorithm',
    'wdo'               : 'Wind Driven Optimization',
    'who'               : 'Wildebeest Herd Optimization',
    'wo_wave'           : 'Wave Optimization Algorithm',
    'woa'               : 'Whale Optimization Algorithm',
    'wso'               : 'White Shark Optimizer',
    'wutp'              : 'Water Uptake and Transport in Plants',
    'ydse'              : "Young's Double-Slit Experiment Optimizer",
    'zoa'               : 'Zebra Optimization Algorithm',
}

_ALGORITHM_FAMILIES: dict[str, str] = {
    'aaa'               : 'swarm',
    'abco'              : 'swarm',
    'acgwo'             : 'swarm',
    'aco'               : 'swarm',
    'acor'              : 'swarm',
    'adam'              : 'math',
    'aefa'              : 'physics',
    'aeo'               : 'human',
    'aesspso'           : 'swarm',
    'afsa'              : 'swarm',
    'aft'               : 'human',
    'agto'              : 'swarm',
    'aha'               : 'swarm',
    'aho'               : 'swarm',
    'ala'               : 'swarm',
    'alo'               : 'swarm',
    'ao'                : 'swarm',
    'aoa'               : 'swarm',
    'apo'               : 'swarm',
    'arch_oa'           : 'physics',
    'aro'               : 'swarm',
    'ars'               : 'trajectory',
    'artemisinin_o'     : 'nature',
    'aso'               : 'swarm',
    'aso_atom'          : 'physics',
    'autov'             : 'evolutionary',
    'avoa'              : 'swarm',
    'bat_a'             : 'swarm',
    'bbo'               : 'evolutionary',
    'bboa'              : 'swarm',
    'bco'               : 'nature',
    'bea'               : 'swarm',
    'bes'               : 'swarm',
    'bfgs'              : 'math',
    'bfo'               : 'swarm',
    'bka'               : 'swarm',
    'bmo'               : 'swarm',
    'boa'               : 'swarm',
    'bono'              : 'swarm',
    'bro'               : 'human',
    'bsa'               : 'swarm',
    'bso'               : 'human',
    'bspga'             : 'evolutionary',
    'bwo'               : 'evolutionary',
    'ca'                : 'evolutionary',
    'camel'             : 'swarm',
    'capsa'             : 'swarm',
    'cat_so'            : 'swarm',
    'cddo'              : 'swarm',
    'cddo_child'        : 'human',
    'cdo'               : 'swarm',
    'cdo_chernobyl'     : 'physics',
    'cem'               : 'distribution',
    'ceo_cosmic'        : 'physics',
    'cgo'               : 'math',
    'chameleon_sa'      : 'swarm',
    'chicken_so'        : 'swarm',
    'chio'              : 'human',
    'choa'              : 'swarm',
    'circle_sa'         : 'math',
    'clonalg'           : 'evolutionary',
    'cmaes'             : 'evolutionary',
    'coa'               : 'swarm',
    'coati_oa'          : 'swarm',
    'cockroach_so'      : 'swarm',
    'coot'              : 'swarm',
    'crayfish_oa'       : 'swarm',
    'cro'               : 'evolutionary',
    'csa'               : 'swarm',
    'csbo'              : 'swarm',
    'cso'               : 'swarm',
    'cuckoo_s'          : 'swarm',
    'da'                : 'swarm',
    'dbo'               : 'swarm',
    'ddao'              : 'physics',
    'de'                : 'evolutionary',
    'deo_dolphin'       : 'swarm',
    'dfo'               : 'swarm',
    'dmoa'              : 'swarm',
    'do_dandelion'      : 'physics',
    'doa'               : 'human',
    'dso'               : 'human',
    'dvba'              : 'swarm',
    'eao'               : 'nature',
    'eco'               : 'human',
    'ecological_cycle_o': 'swarm',
    'ecpo'              : 'physics',
    'edo'               : 'math',
    'efo'               : 'physics',
    'ego'               : 'distribution',
    'eho'               : 'swarm',
    'elk_ho'            : 'swarm',
    'eo'                : 'physics',
    'eoa'               : 'swarm',
    'ep'                : 'evolutionary',
    'epc'               : 'swarm',
    'es'                : 'evolutionary',
    'esc'               : 'human',
    'eso'               : 'physics',
    'esoa'              : 'swarm',
    'eto'               : 'math',
    'evo'               : 'physics',
    'fata'              : 'physics',
    'fbio'              : 'human',
    'fda'               : 'swarm',
    'fdo'               : 'swarm',
    'fep'               : 'evolutionary',
    'ffa'               : 'swarm',
    'ffo'               : 'swarm',
    'firefly_a'         : 'swarm',
    'fla'               : 'physics',
    'flood_a'           : 'physics',
    'foa'               : 'swarm',
    'foa_fossa'         : 'swarm',
    'fox'               : 'swarm',
    'fpa'               : 'swarm',
    'frcg'              : 'math',
    'frofi'             : 'evolutionary',
    'fss'               : 'swarm',
    'fwa'               : 'swarm',
    'ga'                : 'evolutionary',
    'gazelle_oa'        : 'swarm',
    'gbo'               : 'math',
    'gco'               : 'human',
    'gea'               : 'physics',
    'ggo'               : 'swarm',
    'gja'               : 'swarm',
    'gjo'               : 'swarm',
    'gkso'              : 'swarm',
    'gmo'               : 'swarm',
    'gndo'              : 'math',
    'go_growth'         : 'swarm',
    'goa'               : 'swarm',
    'gpso'              : 'swarm',
    'gsa'               : 'physics',
    'gska'              : 'human',
    'gso'               : 'swarm',
    'gso_glider_snake'  : 'swarm',
    'gto'               : 'swarm',
    'gwo'               : 'swarm',
    'hba'               : 'swarm',
    'hba_honey'         : 'swarm',
    'hbo'               : 'human',
    'hc'                : 'trajectory',
    'hco'               : 'human',
    'hde'               : 'evolutionary',
    'heoa'              : 'human',
    'hgs'               : 'swarm',
    'hgso'              : 'physics',
    'hho'               : 'swarm',
    'hiking_oa'         : 'human',
    'ho_hippo'          : 'swarm',
    'horse_oa'          : 'swarm',
    'hsa'               : 'trajectory',
    'hsaba'             : 'swarm',
    'hus'               : 'swarm',
    'iagwo'             : 'swarm',
    'i_gwo'             : 'swarm',
    'i_woa'             : 'swarm',
    'ica'               : 'human',
    'ikoa'              : 'physics',
    'ilshade'           : 'evolutionary',
    'imode'             : 'evolutionary',
    'info'              : 'math',
    'ivya'              : 'nature',
    'iwo'               : 'nature',
    'jde'               : 'evolutionary',
    'jso'               : 'swarm',
    'jy'                : 'swarm',
    'kha'               : 'swarm',
    'kma'               : 'swarm',
    'l2smea'            : 'evolutionary',
    'lca'               : 'nature',
    'lco'               : 'human',
    'lfd'               : 'swarm',
    'liwo'              : 'physics',
    'loa'               : 'swarm',
    'loa_lyrebird'      : 'swarm',
    'lpo'               : 'nature',
    'lshade_cnepsin'    : 'evolutionary',
    'lso_spectrum'      : 'physics',
    'mbo'               : 'swarm',
    'memetic_a'         : 'evolutionary',
    'mfa'               : 'swarm',
    'mfea'              : 'evolutionary',
    'mfea2'             : 'evolutionary',
    'mgo'               : 'swarm',
    'mgoa_market'       : 'human',
    'misaco'            : 'swarm',
    'mke'               : 'evolutionary',
    'moss_go'           : 'nature',
    'mpa'               : 'swarm',
    'mrfo'              : 'swarm',
    'msa_e'             : 'swarm',
    'mshoa'             : 'swarm',
    'mso'               : 'physics',
    'mts'               : 'trajectory',
    'mvo'               : 'swarm',
    'mvpa'              : 'human',
    'nca'               : 'math',
    'ngo'               : 'swarm',
    'nmm'               : 'trajectory',
    'nmra'              : 'swarm',
    'nndrea_so'         : 'evolutionary',
    'noa'               : 'math',
    'nro'               : 'physics',
    'nwoa'              : 'swarm',
    'ofa'               : 'swarm',
    'ooa'               : 'swarm',
    'parrot_o'          : 'swarm',
    'pbil'              : 'distribution',
    'pcx'               : 'evolutionary',
    'pdo'               : 'swarm',
    'pfa'               : 'swarm',
    'pfa_polar_fox'     : 'swarm',
    'pko'               : 'swarm',
    'plba'              : 'swarm',
    'plo'               : 'physics',
    'poa'               : 'swarm',
    'political_o'       : 'human',
    'pro'               : 'human',
    'pso'               : 'swarm',
    'pss'               : 'math',
    'puma_o'            : 'swarm',
    'qio'               : 'math',
    'qsa'               : 'human',
    'random_s'          : 'trajectory',
    'rbmo'              : 'swarm',
    'rfo'               : 'swarm',
    'rime'              : 'physics',
    'rmsprop'           : 'math',
    'roa'               : 'swarm',
    'rsa'               : 'swarm',
    'rso'               : 'swarm',
    'run'               : 'math',
    'sa'                : 'trajectory',
    'saba'              : 'swarm',
    'sacc_eam2'         : 'evolutionary',
    'sacoso'            : 'swarm',
    'sade_amss'         : 'evolutionary',
    'sade_atdsc'        : 'evolutionary',
    'sade_sammon'       : 'evolutionary',
    'samso'             : 'swarm',
    'sapo'              : 'evolutionary',
    'saro'              : 'human',
    'sbo'               : 'swarm',
    'sboa'              : 'swarm',
    'scho'              : 'math',
    'scso'              : 'swarm',
    'sd'                : 'math',
    'seaho'             : 'swarm',
    'serval_oa'         : 'swarm',
    'sfo'               : 'swarm',
    'sfoa'              : 'swarm',
    'shade'             : 'evolutionary',
    'shio'              : 'swarm',
    'shio_success'      : 'swarm',
    'sho'               : 'swarm',
    'sine_cosine_a'     : 'swarm',
    'slo'               : 'swarm',
    'sma'               : 'nature',
    'smo'               : 'swarm',
    'snow_oa'           : 'physics',
    'so_snake'          : 'swarm',
    'soa'               : 'swarm',
    'soo'               : 'physics',
    'sos'               : 'swarm',
    'sparrow_sa'        : 'swarm',
    'spbo'              : 'swarm',
    'sqp'               : 'math',
    'squirrel_sa'       : 'swarm',
    'srsr'              : 'swarm',
    'ssa'               : 'swarm',
    'ssdo'              : 'human',
    'ssio_rl'           : 'evolutionary',
    'sso'               : 'swarm',
    'sspider_a'         : 'swarm',
    'sto'               : 'swarm',
    'superb_foa'        : 'swarm',
    'supply_do'         : 'human',
    'tdo'               : 'swarm',
    'thro'              : 'human',
    'tlbo'              : 'swarm',
    'tlco'              : 'swarm',
    'toa'               : 'human',
    'toc'               : 'physics',
    'tpo'               : 'nature',
    'ts'                : 'trajectory',
    'tsa'               : 'swarm',
    'tso'               : 'swarm',
    'ttao'              : 'math',
    'two'               : 'physics',
    'vcs'               : 'swarm',
    'waoa'              : 'swarm',
    'warso'             : 'human',
    'wca'               : 'nature',
    'wdo'               : 'physics',
    'who'               : 'swarm',
    'wo_wave'           : 'physics',
    'woa'               : 'swarm',
    'wso'               : 'swarm',
    'wutp'              : 'nature',
    'ydse'              : 'physics',
    'zoa'               : 'swarm',
}

_ALGORITHM_DOIS: dict[str, str] = {
    'aaa'               : '10.1016/j.asoc.2015.03.003',
    'abco'              : '10.1007/s10898-007-9149-x',
    'acgwo'             : '10.1007/s42835-023-01621-w',
    'aco'               : '10.1109/3477.484436',
    'acor'              : '10.1016/j.ejor.2006.06.046',
    'adam'              : '10.48550/arXiv.1412.6980',
    'aefa'              : '10.1016/j.swevo.2019.03.013',
    'aeo'               : '10.1007/s00521-019-04452-x',
    'aesspso'           : '10.1016/j.swevo.2025.101868',
    'afsa'              : '10.1007/s10462-012-9342-2',
    'aft'               : '10.1007/s00521-021-06392-x',
    'agto'              : '10.1002/int.22535',
    'aha'               : '10.1016/j.cma.2021.114194',
    'aho'               : '10.1016/j.engappai.2024.108081',
    'ala'               : '10.1007/s10462-024-11023-7',
    'alo'               : '10.1016/j.advengsoft.2015.01.010',
    'ao'                : '10.1016/j.cie.2021.107250',
    'aoa'               : '10.1016/j.cma.2020.113609',
    'apo'               : '10.1016/j.knosys.2024.111737',
    'arch_oa'           : '10.1007/s10489-020-01893-z',
    'aro'               : '10.1016/j.engappai.2022.105082',
    'ars'               : '10.1002/nav.20422',
    'artemisinin_o'     : '10.1016/j.displa.2024.102740',
    'aso'               : '10.1109/CEC.2011.5949940',
    'aso_atom'          : '10.1016/j.knosys.2018.08.030',
    'autov'             : '10.1145/3712256.3726456',
    'avoa'              : '10.1016/j.cie.2021.107408',
    'bat_a'             : '10.1007/978-3-642-12538-6_6',
    'bbo'               : '10.1109/TEVC.2008.919004',
    'bboa'              : '10.1201/9781003337003-6',
    'bco'               : '10.1007/s13369-025-10749-y',
    'bea'               : '10.1016/B978-008045157-2/50081-X',
    'bes'               : '10.1007/s10462-019-09732-5',
    'bfgs'              : '10.1090/S0025-5718-1970-0274029-X',
    'bfo'               : '10.1109/MCS.2002.1004010',
    'bka'               : '10.1007/s10462-024-10723-4',
    'bmo'               : '10.1016/j.engappai.2019.103330',
    'boa'               : '10.1007/s00500-018-3102-4',
    'bono'              : '10.1007/s10489-021-02444-w',
    'bro'               : '10.1007/s00521-020-05004-4',
    'bsa'               : '10.1080/0952813X.2015.1042530',
    'bso'               : '10.1007/978-3-642-21515-5_36',
    'bspga'             : '10.1016/j.ins.2019.10.016',
    'bwo'               : '10.1016/j.engappai.2019.103249',
    'ca'                : '10.1080/00207160.2015.1067309',
    'camel'             : '10.13140/RG.2.2.21814.56649',
    'capsa'             : '10.1007/s00521-020-05145-6',
    'cat_so'            : '10.1007/978-3-540-36668-3_94',
    'cddo'              : '10.1038/s41598-022-14338-z',
    'cddo_child'        : '10.1016/j.knosys.2024.111558',
    'cdo'               : '10.1038/s41598-022-14338-z',
    'cdo_chernobyl'     : '10.1016/j.compstruc.2023.107488',
    'cem'               : '10.1007/978-1-4757-4321-0',
    'ceo_cosmic'        : '10.1007/s00521-025-11234-6',
    'cgo'               : '10.1007/s10462-020-09867-w',
    'chameleon_sa'      : '10.1016/j.eswa.2021.114685',
    'chicken_so'        : '10.1007/978-3-319-11857-4_10',
    'chio'              : '10.1007/s00521-020-05296-6',
    'choa'              : '10.1016/j.eswa.2020.113338',
    'circle_sa'         : '10.3390/math10101626',
    'clonalg'           : '10.1109/TEVC.2002.1011539',
    'cmaes'             : '10.1109/ICEC.1996.542381',
    'coa'               : '10.1109/CEC.2018.8477769',
    'coati_oa'          : '10.1016/j.knosys.2022.110011',
    'cockroach_so'      : '10.1109/ICCET.2010.5485993',
    'coot'              : '10.1016/j.eswa.2021.115352',
    'crayfish_oa'       : '10.1007/s10462-023-10567-4',
    'cro'               : '10.1155/2014/739768',
    'csa'               : '10.1016/j.compstruc.2016.03.001',
    'csbo'              : '10.1016/j.egyr.2025.04.007',
    'cso'               : '10.1016/j.swevo.2024.101543',
    'cuckoo_s'          : '10.1109/NABIC.2009.5393690',
    'da'                : '10.1007/s00521-015-1920-1',
    'dbo'               : '10.1007/s11227-022-04959-6',
    'ddao'              : '10.1016/j.asoc.2020.106392',
    'de'                : '10.1023/A:1008202821328',
    'deo_dolphin'       : '10.1016/j.advengsoft.2016.05.002',
    'dfo'               : '10.15439/2014F142',
    'dmoa'              : '10.1016/j.cma.2022.114570',
    'do_dandelion'      : '10.1016/j.engappai.2022.105075',
    'doa'               : '10.1093/comjnl/bxy133',
    'dso'               : '10.1109/ACCESS.2023.3298105',
    'dvba'              : '10.1109/INCoS.2014.40',
    'eao'               : '10.1007/s11227-025-07052-w',
    'eco'               : '10.3390/biomimetics10030176',
    'ecological_cycle_o': '10.48550/arXiv.2508.20458',
    'ecpo'              : '10.1007/s10462-020-09890-x',
    'edo'               : '10.1007/s10462-023-10403-9',
    'efo'               : '10.1016/j.swevo.2015.07.002',
    'ego'               : '10.1023/A:1008306431147',
    'eho'               : '10.1109/ISCBI.2015.8',
    'elk_ho'            : '10.1007/s10462-023-10680-4',
    'eo'                : '10.1016/j.knosys.2019.105190',
    'eoa'               : '10.1504/IJBIC.2015.10004283',
    'ep'                : '10.1007/BF00175356',
    'epc'               : '10.1016/j.knosys.2018.06.001',
    'es'                : '10.1023/A:1015059928466',
    'esc'               : '10.1007/s10462-024-11008-6',
    'eso'               : '10.3390/make7010024',
    'esoa'              : '10.3390/biomimetics7040144',
    'eto'               : '10.1016/j.cma.2024.117411',
    'evo'               : '10.1038/s41598-022-27344-y',
    'fata'              : '10.1016/j.neucom.2024.128289',
    'fbio'              : '10.1016/j.asoc.2020.106339',
    'fda'               : '10.1016/j.cie.2021.107224',
    'fdo'               : '10.1109/ACCESS.2019.2907012',
    'fep'               : '10.1109/4235.771163',
    'ffa'               : '10.1016/j.knosys.2011.07.001',
    'ffo'               : '10.1109/ACCESS.2022.3197745',
    'firefly_a'         : '10.1504/IJBIC.2010.032124',
    'fla'               : '10.1016/j.knosys.2022.110146',
    'flood_a'           : '10.1007/s11227-024-06291-7',
    'foa'               : '10.1016/j.eswa.2014.05.009',
    'foa_fossa'         : '10.1007/s10462-024-10953-0',
    'fox'               : '10.1007/s10489-022-03533-0',
    'fpa'               : '10.1007/978-3-642-32894-7_27',
    'frcg'              : '10.1002/er.8067',
    'frofi'             : '10.1109/TCYB.2015.2493239',
    'fss'               : '10.1109/ICSMC.2008.4811695',
    'fwa'               : '10.1016/j.asoc.2017.10.046',
    'ga'                : '10.7551/mitpress/1090.001.0001',
    'gazelle_oa'        : '10.1007/s00521-022-07854-6',
    'gbo'               : '10.1007/s11831-022-09872-y',
    'gco'               : '10.1016/j.ifacol.2018.07.300',
    'gea'               : '10.1007/s42235-023-00437-8',
    'ggo'               : '10.1016/j.eswa.2023.122147',
    'gja'               : '10.1016/j.eswa.2025.127982',
    'gjo'               : '10.1016/j.eswa.2022.116924',
    'gkso'              : '10.1016/j.aei.2023.102210',
    'gmo'               : '10.1007/s00500-023-08202-z',
    'gndo'              : '10.1016/j.enconman.2020.113301',
    'go_growth'         : '10.1016/j.knosys.2022.110206',
    'goa'               : '10.1016/j.advengsoft.2017.01.004',
    'gpso'              : '10.48550/arXiv.2312.09703',
    'gsa'               : '10.1016/j.ins.2009.03.004',
    'gska'              : '10.1007/s13042-019-01053-x',
    'gso'               : '10.1007/978-3-319-51595-3',
    'gso_glider_snake'  : '10.1007/s10462-026-11504-x',
    'gto'               : '10.1109/ACCESS.2022.3223388',
    'gwo'               : '10.1016/j.advengsoft.2013.12.007',
    'hba'               : '10.48550/arXiv.1303.6310',
    'hba_honey'         : '10.1016/j.matcom.2021.08.013',
    'hbo'               : '10.1016/j.eswa.2020.113702',
    'hc'                : '10.1007/978-3-540-75256-1_52',
    'hco'               : '10.1038/s41598-022-25031-6',
    'hde'               : '10.1109/CEC.2009.4983179',
    'heoa'              : '10.1016/j.eswa.2023.122638',
    'hgs'               : '10.1016/j.eswa.2021.114864',
    'hgso'              : '10.1016/j.future.2019.07.015',
    'hho'               : '10.1016/j.future.2019.02.028',
    'hiking_oa'         : '10.1016/j.knosys.2024.111880',
    'ho_hippo'          : '10.1038/s41598-024-54910-3',
    'horse_oa'          : '10.1016/j.knosys.2020.106711',
    'hsa'               : '10.1177/003754970107600201',
    'hsaba'             : '10.1155/2014/709738',
    'hus'               : '10.1109/ICSCCW.2009.5379451',
    'iagwo'             : '10.1007/s10462-024-10821-3',
    'i_gwo'             : '10.1016/j.eswa.2020.113917',
    'i_woa'             : '10.1016/j.jcde.2019.02.002',
    'ica'               : '10.1109/CEC.2007.4425083',
    'ikoa'              : '10.1016/j.eswa.2025.128216',
    'ilshade'           : '10.1109/CEC.2016.7743922',
    'imode'             : '10.1109/CEC48606.2020.9185577',
    'info'              : '10.1016/j.eswa.2022.116516',
    'ivya'              : '10.1016/j.knosys.2024.111850',
    'iwo'               : '10.1016/j.ecoinf.2006.07.003',
    'jde'               : '10.1109/TEVC.2006.872133',
    'jso'               : '10.1016/j.amc.2020.125535',
    'jy'                : '10.5267/j.ijiec.2015.8.004',
    'kha'               : '10.1016/j.asoc.2016.08.041',
    'kma'               : '10.1016/j.asoc.2021.108043',
    'l2smea'            : '10.1109/TEVC.2023.3319640',
    'lca'               : '10.1016/j.compbiomed.2023.107389',
    'lco'               : '10.1007/s00500-019-04443-z',
    'lfd'               : '10.1016/j.engappai.2020.103731',
    'liwo'              : '10.1109/ACCESS.2024.3390670',
    'loa'               : '10.1016/j.jcde.2015.06.003',
    'loa_lyrebird'      : '10.1016/j.cma.2023.116436',
    'lpo'               : '10.1016/j.cma.2023.116582',
    'lshade_cnepsin'    : '10.1109/CEC.2016.7744173',
    'lso_spectrum'      : '10.1016/j.asoc.2024.112318',
    'mbo'               : '10.1007/s00521-015-1923-y',
    'memetic_a'         : '10.1007/978-3-540-92910-9_29',
    'mfa'               : '10.1016/j.knosys.2015.07.006',
    'mfea'              : '10.1109/TEVC.2015.2458037',
    'mfea2'             : '10.1109/TEVC.2019.2906927',
    'mgo'               : '10.1016/j.advengsoft.2022.103282',
    'mgoa_market'       : '10.1016/j.asoc.2024.112466',
    'misaco'            : '10.1109/TCYB.2021.3064676',
    'mke'               : '10.1016/j.knosys.2016.01.009',
    'moss_go'           : '10.1093/jcde/qwae080',
    'mpa'               : '10.1016/j.eswa.2020.113377',
    'mrfo'              : '10.1016/j.engappai.2019.103300',
    'msa_e'             : '10.1007/s12293-016-0212-3',
    'mshoa'             : '10.3390/math13091500',
    'mso'               : '10.1016/j.advengsoft.2025.103883',
    'mts'               : '10.1109/CEC.2008.4631210',
    'mvo'               : '10.1007/s00521-015-1870-7',
    'mvpa'              : '10.1007/s12351-017-0320-y',
    'nca'               : '10.1007/s00500-023-08925-z',
    'ngo'               : '10.1109/ACCESS.2021.3133286',
    'nmm'               : '10.1093/comjnl/7.4.308',
    'nmra'              : '10.1007/s00521-019-04464-7',
    'nndrea_so'         : '10.1109/TEVC.2024.3400398',
    'noa'               : '10.1007/s11227-023-05579-4',
    'nro'               : '10.1109/ACCESS.2019.2918406',
    'nwoa'              : '10.1038/s41598-024-61278-8',
    'ofa'               : '10.1016/j.eswa.2022.117735',
    'ooa'               : '10.3389/fmech.2022.1126450',
    'parrot_o'          : '10.1016/j.compbiomed.2024.108064',
    'pbil'              : '10.1109/SSE62657.2024.00022',
    'pcx'               : '10.1109/CEC.2004.1331141',
    'pdo'               : '10.1007/s00521-022-07530-9',
    'pfa'               : '10.1016/j.asoc.2019.03.012',
    'pfa_polar_fox'     : '10.1007/s00521-024-10346-4',
    'pko'               : '10.1007/s00521-024-09879-5',
    'plba'              : 'https://www.iztok-jr-fister.eu/static/publications/124.pdf',
    'plo'               : '10.1016/j.neucom.2024.128427',
    'poa'               : '10.3390/s22030855',
    'political_o'       : '10.1016/j.knosys.2020.105709',
    'pro'               : '10.1016/j.engappai.2019.08.025',
    'pso'               : '10.1109/ICNN.1995.488968',
    'pss'               : '10.1007/s00500-021-05853-8',
    'puma_o'            : '10.1007/s10586-023-04221-5',
    'qio'               : '10.1016/j.cma.2023.116446',
    'qsa'               : '10.1007/s12652-020-02849-4',
    'random_s'          : '10.1016/j.advengsoft.2022.103141',
    'rbmo'              : '10.1007/s10462-024-10716-3',
    'rfo'               : '10.1007/s10586-024-04950-1',
    'rime'              : '10.1016/j.neucom.2023.02.010',
    'roa'               : '10.1016/j.eswa.2021.115665',
    'rsa'               : '10.1016/j.eswa.2021.116158',
    'rso'               : '10.1007/s12652-020-02580-0',
    'run'               : '10.1016/j.eswa.2021.115079',
    'sa'                : '10.1126/science.220.4598.671',
    'saba'              : '10.1155/2014/709738',
    'sacc_eam2'         : '10.1007/978-3-319-97773-7_4',
    'sacoso'            : '10.1109/TEVC.2017.2675628',
    'sade_amss'         : '10.1109/TEVC.2022.3226837',
    'sade_atdsc'        : '10.1109/SSCI51031.2022.10022105',
    'sade_sammon'       : '10.1016/j.petrol.2019.106633',
    'samso'             : '10.1109/TCYB.2020.2967553',
    'sapo'              : '10.1007/978-3-031-70068-2_24',
    'saro'              : '10.1155/2019/2482543',
    'sbo'               : '10.1016/j.engappai.2017.01.006',
    'sboa'              : '10.1007/s10462-024-10729-y',
    'scho'              : '10.1016/j.knosys.2023.111081',
    'scso'              : '10.1007/s00366-022-01604-x',
    'sd'                : '10.1006/hmat.1996.2146',
    'seaho'             : '10.1007/s10489-022-03994-3',
    'serval_oa'         : '10.3390/biomimetics7040204',
    'sfo'               : '10.1016/j.engappai.2019.01.001',
    'sfoa'              : '10.1007/s00521-024-10694-1',
    'shade'             : '10.1109/CEC.2014.6900380',
    'shio'              : '10.1016/j.advengsoft.2017.05.014',
    'shio_success'      : '10.1016/j.cma.2024.117272',
    'sho'               : '10.1016/j.advengsoft.2017.05.014',
    'sine_cosine_a'     : '10.1016/j.knosys.2015.12.022',
    'slo'               : '10.14569/IJACSA.2019.0100548',
    'sma'               : '10.1016/j.future.2020.03.055',
    'smo'               : '10.1007/s12293-013-0128-0',
    'snow_oa'           : '10.1016/j.eswa.2023.120069',
    'so_snake'          : '10.1016/j.knosys.2022.108320',
    'soa'               : '10.1016/j.knosys.2018.11.024',
    'soo'               : '10.1007/s10586-024-04976-5',
    'sos'               : '10.1016/j.compstruc.2014.03.007',
    'sparrow_sa'        : '10.1080/21642583.2019.1708830',
    'spbo'              : '10.1016/j.advengsoft.2020.102804',
    'sqp'               : '10.1017/S0962492900002518',
    'squirrel_sa'       : '10.1016/j.swevo.2018.02.013',
    'srsr'              : '10.1007/978-3-319-70139-4_16',
    'ssa'               : '10.1016/j.advengsoft.2017.07.002',
    'ssdo'              : '10.1007/s00521-019-04159-z',
    'ssio_rl'           : 'https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2025.125444',
    'sso'               : '10.1016/j.eswa.2013.05.041',
    'sspider_a'         : '10.1016/j.asoc.2015.02.014',
    'sto'               : '10.1109/ACCESS.2022.3229964',
    'superb_foa'        : '10.1007/s10586-024-04901-w',
    'supply_do'         : '10.1109/ACCESS.2019.2919408',
    'tdo'               : '10.1109/ACCESS.2022.3151641',
    'thro'              : '10.1007/s10462-025-11269-9',
    'tlbo'              : '10.1016/j.cad.2010.12.015',
    'tlco'              : '10.1016/j.eswa.2022.119211',
    'toa'               : '10.3390/s21134567',
    'toc'               : '10.1007/s10462-025-11118-9',
    'tpo'               : '10.1515/jisys-2017-0156',
    'ts'                : '10.1287/ijoc.1.3.190',
    'tsa'               : '10.1016/j.engappai.2020.103541',
    'tso'               : '10.1155/2021/9210050',
    'ttao'              : '10.1016/j.eswa.2023.121744',
    'two'               : '10.1007/978-3-030-04067-3_11',
    'vcs'               : '10.1016/j.advengsoft.2015.11.004',
    'waoa'              : '10.1038/s41598-023-35863-5',
    'warso'             : '10.1109/ACCESS.2022.3153493',
    'wca'               : '10.1016/j.compstruc.2012.07.010',
    'wdo'               : '10.1109/APS.2010.5562213',
    'who'               : '10.3233/JIFS-190495',
    'wo_wave'           : '10.1016/j.cor.2014.10.008',
    'woa'               : '10.1016/j.advengsoft.2016.01.008',
    'wso'               : '10.1016/j.knosys.2022.108457',
    'wutp'              : '10.1007/s00521-025-11228-z',
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
