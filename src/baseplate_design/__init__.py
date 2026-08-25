"""baseplate_design: pinned-base column baseplates for an optimized frame.

Runs off the back of `frame_optimizer`: the finalized gravity member design
supplies each column's section and its base reactions, and this module turns
them into ONE baseplate detail used at every column base -- the way an
industrial building is actually fabricated.

    from frame_optimizer import optimize_layout
    from baseplate_design import (BaseplateConfig, design_uniform_baseplate,
                                  write_baseplate_configuration_json)

    result = optimize_layout(config)
    design = design_uniform_baseplate(result, BaseplateConfig())
    print(design.summary())
    write_baseplate_configuration_json(
        design, "output/baseplate_configuration.json")

Design basis: AISC 360-22 + AISC Design Guide 1 (2nd Ed.), LRFD. Compression
bearing, plate flexure and anchor rod shear on a pinned base. Interface units
are SI (mm, kN, MPa); the design math runs in AISC-native kips and inches.
"""
from .baseplate_design import (BaseplateCheck, ColumnDemand, PlateGeometry,
                               check_plate, design, design_plate,
                               effective_A2, required_A1)
from .config import BaseplateConfig
from .export import baseplate_configuration, write_baseplate_configuration_json
from .uniform_design import (UniformBaseplateDesign, demands_from_inputs,
                             design_uniform_baseplate)

__all__ = [
    "BaseplateCheck",
    "BaseplateConfig",
    "ColumnDemand",
    "PlateGeometry",
    "UniformBaseplateDesign",
    "baseplate_configuration",
    "check_plate",
    "demands_from_inputs",
    "design",
    "design_plate",
    "design_uniform_baseplate",
    "effective_A2",
    "required_A1",
    "write_baseplate_configuration_json",
]
