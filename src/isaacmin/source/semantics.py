"""Conservative semantic policy: unrecognized blocks remain unknown, never air."""
from __future__ import annotations

AIR = {"air", "cave_air", "void_air"}
SOIL = {"dirt", "grass_block", "coarse_dirt", "rooted_dirt", "podzol", "mycelium", "mud", "clay", "farmland", "dirt_path", "moss_block", "pale_moss_block"}
SEDIMENT = {"sand", "red_sand", "gravel", "suspicious_sand", "suspicious_gravel", "soul_sand", "soul_soil"}
ROCK = {"stone", "granite", "diorite", "andesite", "deepslate", "tuff", "calcite", "dripstone_block", "bedrock", "obsidian", "crying_obsidian", "magma_block", "netherrack", "basalt", "smooth_basalt", "blackstone", "end_stone", "sandstone", "red_sandstone", "amethyst_block", "budding_amethyst", "terracotta"}
VEGETATION = {"short_grass", "grass", "tall_grass", "fern", "large_fern", "dead_bush", "vine", "glow_lichen", "moss_carpet", "pale_moss_carpet", "hanging_roots", "azalea", "flowering_azalea", "spore_blossom", "cave_vines", "cave_vines_plant", "seagrass", "tall_seagrass", "kelp", "kelp_plant", "lily_pad", "cactus", "sugar_cane", "bamboo", "bamboo_sapling", "brown_mushroom", "red_mushroom", "brown_mushroom_block", "red_mushroom_block", "mushroom_stem", "pumpkin", "melon", "sweet_berry_bush", "small_dripleaf", "big_dripleaf", "big_dripleaf_stem", "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "oxeye_daisy", "cornflower", "lily_of_the_valley", "wither_rose", "sunflower", "lilac", "rose_bush", "peony", "pink_petals", "wildflowers", "leaf_litter", "bush", "firefly_bush", "short_dry_grass", "tall_dry_grass", "cactus_flower", "mangrove_roots", "muddy_mangrove_roots", "resin_clump", "pale_hanging_moss"}
STRUCTURES = {"chest", "trapped_chest", "barrel", "crafting_table", "furnace", "blast_furnace", "smoker", "spawner", "trial_spawner", "vault", "bookshelf", "cobweb", "torch", "wall_torch", "soul_torch", "soul_wall_torch", "ladder", "chain", "lantern", "soul_lantern", "iron_bars", "rail", "powered_rail", "detector_rail", "activator_rail", "cobblestone", "mossy_cobblestone", "bricks", "stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks", "chiseled_stone_bricks", "glass", "glass_pane", "redstone_wire", "redstone_torch", "redstone_wall_torch", "lever", "tripwire", "tripwire_hook", "water_cauldron", "cauldron", "flower_pot", "bee_nest", "beehive", "hay_block", "bell", "campfire", "soul_campfire", "composter", "fletching_table", "smithing_table", "cartography_table", "stonecutter", "grindstone", "brewing_stand", "enchanting_table", "jukebox", "note_block"}


def classify(name: str) -> str:
    if not name.startswith("minecraft:"):
        return "unknown"
    block = name.split(":", 1)[1]
    if block in AIR:
        return "air"
    if block in {"water", "bubble_column"}:
        return "water"
    if block == "lava":
        return "lava"
    if block in SOIL:
        return "soil"
    if block in SEDIMENT:
        return "sediment"
    if block in ROCK or block in {"raw_copper_block", "raw_iron_block", "raw_gold_block", "bone_block"} or block.endswith("_ore") or block.startswith("infested_") or block.endswith("_terracotta") and not block.endswith("glazed_terracotta"):
        return "rock"
    if block in {"ice", "packed_ice", "blue_ice", "frosted_ice"}:
        return "ice"
    if block == "snow_block":
        return "snow_volume"
    if block in {"snow", "powder_snow"}:
        return "snow_cover"
    if block == "pointed_dripstone" or block.endswith("_amethyst_bud") or block == "amethyst_cluster":
        return "mineral_detail"
    if block in VEGETATION or block in {"cocoa", "mangrove_propagule", "sea_pickle"} or block.endswith(("_leaves", "_log", "_wood", "_sapling", "_tulip", "_coral", "_coral_fan", "_coral_wall_fan", "_coral_block")):
        return "vegetation"
    if block in STRUCTURES or block.endswith(("_stairs", "_slab", "_fence", "_fence_gate", "_planks", "_door", "_trapdoor", "_wall", "_button", "_pressure_plate", "_bed", "_wool", "_carpet", "_banner", "_sign", "_hanging_sign", "_candle", "_concrete", "_concrete_powder", "_stained_glass", "_stained_glass_pane", "_glazed_terracotta", "_tiles", "_bricks")) or block.startswith("potted_"):
        return "structure"
    return "unknown"


GROUND_CLASSES = {"soil", "sediment", "rock", "ice", "snow_volume"}


def biome_family(biome: str) -> str:
    name = biome.split(":")[-1]
    if biome.startswith("minecraft:"):
        if any(v in name for v in ("forest", "taiga", "grove", "jungle", "cherry")):
            return "forest"
        if any(v in name for v in ("peak", "mountain", "hills", "slopes", "stony", "badlands")):
            return "rocky_mountain"
        if name in ("desert", "beach"):
            return "sand_terrain_dunes_not_inferred"
        if "ocean" in name or name in ("river", "frozen_river"):
            return "water"
        if "cave" in name or name == "deep_dark":
            return "subsurface"
        if name in ("plains", "sunflower_plains", "meadow", "savanna", "savanna_plateau", "snowy_plains"):
            return "grassland"
        if "swamp" in name:
            return "wetland"
    return "unsupported"
