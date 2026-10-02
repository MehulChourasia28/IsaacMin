"""Explicit outdoor biome recipes; recognition never implies qualification."""
from dataclasses import dataclass,asdict
from collections import Counter
import numpy as np


@dataclass(frozen=True)
class BiomeRecipe:
    family: str
    biomes: tuple[str,...]
    ground_material: str
    terrain_process: str
    canopy_species: tuple[str,...]
    ground_cover: str
    required_visual_features: tuple[str,...]


RECIPES=(
    BiomeRecipe('temperate_grassland',('plains','sunflower_plains','meadow'),'leafy_grass',
        'drainage_erosion_and_soil_dequantization',('oak','birch'),'temperate_meadow',
        ('patchy_short_and_tall_grasses','forbs','exposed_soil','worn_routes')),
    BiomeRecipe('deciduous_forest',('forest','flower_forest'),'forest_ground_04',
        'drainage_erosion_and_soil_dequantization',('oak','birch'),'woodland_herbs_and_litter',
        ('layered_canopy','leaf_litter','ferns','roots','deadwood')),
    BiomeRecipe('birch_forest',('birch_forest','old_growth_birch_forest'),'forest_ground_05',
        'drainage_erosion_and_soil_dequantization',('birch',),'light_woodland',
        ('white_bark','mixed_ages','filtered_understory','litter')),
    BiomeRecipe('dark_forest',('dark_forest','pale_garden'),'forest_ground_04',
        'drainage_erosion_and_soil_dequantization',('dark_oak','pale_oak'),'shaded_woodland',
        ('dense_broadleaf_canopy','root_buttresses','shaded_understory')),
    BiomeRecipe('conifer_forest',('taiga','old_growth_pine_taiga','old_growth_spruce_taiga'),
        'forest_ground_05','drainage_and_talus',('spruce','pine'),'conifer_understory',
        ('needle_litter','conifer_silhouettes','dead_branches','moss')),
    BiomeRecipe('snowy_forest',('snowy_taiga','grove'),'snow_02','snow_cover_over_preserved_relief',
        ('spruce','pine'),'snow_and_sparse_shrubs',('snow_drifts','conifers','snow_ground_contact')),
    BiomeRecipe('rocky_mountain',('stony_peaks','windswept_hills','windswept_gravelly_hills',
        'windswept_forest','stony_shore'),'rocky_gravel','mountain_drainage_and_talus',
        ('oak','spruce'),'altitude_and_slope_limited',('rock_outcrops','scree','valley_drainage','preserved_skyline')),
    BiomeRecipe('alpine',('frozen_peaks','jagged_peaks','snowy_slopes','snowy_plains','ice_spikes'),
        'snow_02','snow_cover_over_preserved_relief',(),'sparse_alpine',
        ('rock_snow_boundaries','windward_leeward_snow','preserved_skyline')),
    BiomeRecipe('desert',('desert',),'sand_03','source_relief_with_wind_aligned_sand_transport',(),
        'sparse_arid',('sand_ripples','drought_shrubs','rocky_exposures')),
    BiomeRecipe('coastal_sand',('beach','snowy_beach'),'sand_03','shoreline_preserving_sediment',(),
        'sparse_coastal',('wet_dry_sand','consistent_waterline','shore_debris')),
    BiomeRecipe('dry_grassland',('savanna','savanna_plateau','windswept_savanna'),'withered_grass',
        'dry_soil_and_runoff',('acacia',),'dry_grass_and_shrubs',('dry_grass','acacia_silhouettes','bare_soil')),
    BiomeRecipe('badlands',('badlands','eroded_badlands','wooded_badlands'),'cliff_side',
        'layered_sediment_weathering',('oak',),'sparse_arid',('strata','eroded_buttes','scree','arid_plants')),
    BiomeRecipe('wetland',('swamp','mangrove_swamp'),'mud_forest','water_level_preserving_banks',
        ('oak','mangrove'),'wetland_vegetation',('wet_soil','reeds','water_plants','species_specific_roots')),
    BiomeRecipe('tropical_forest',('jungle','sparse_jungle','bamboo_jungle'),'mud_forest',
        'humid_soil_and_drainage',('jungle','bamboo'),'tropical_understory',
        ('broadleaf_tropical_canopy','vines','large_herbs','humid_litter')),
    BiomeRecipe('flowering_woodland',('cherry_grove',),'forest_ground_05','soil_and_drainage',
        ('cherry',),'flowering_woodland',('cherry_crowns','petal_litter','understory')),
    BiomeRecipe('surface_water',('river','frozen_river','ocean','deep_ocean','cold_ocean',
        'deep_cold_ocean','lukewarm_ocean','deep_lukewarm_ocean','warm_ocean','frozen_ocean','deep_frozen_ocean'),
        'sand_03','preserve_water_surface_and_bed',(),'water_edge_only',('shoreline','water_depth','bank_transition')),
)
BY_BIOME={name:recipe for recipe in RECIPES for name in recipe.biomes}


def biome_inventory(biomes,valid,*,available_materials=(),qualified_families=()):
    counts=Counter(np.asarray(biomes)[np.asarray(valid,bool)].tolist());families={};unknown={}
    for name,count in sorted(counts.items()):
        recipe=BY_BIOME.get(name.removeprefix('minecraft:')) if name.startswith('minecraft:') else None
        if recipe is None:unknown[name]=count;continue
        row=families.setdefault(recipe.family,dict(recipe=asdict(recipe),source_biomes={},samples=0,
            material_available=recipe.ground_material in available_materials,
            qualification='automated_visual_qualified' if recipe.family in qualified_families else 'not_run'))
        row['samples']+=count;row['source_biomes'][name]=count
    return dict(families=families,unknown_biomes=unknown,
        policy='Each biome needs distinct applicable geometry, materials and ecology; missing families block qualification, never generic substitution')
