import os
import pandas as pd
from modelarts_worker.physics.BrusEngine import BrusEngine
from modelarts_worker.physics.PbSEngine import PbSEngine
from modelarts_worker.physics.PerovskiteEngine import PerovskiteEngine

CONFIRMED_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "db", "confirmed_materials.csv")


class MaterialSpec:
    """
    MaterialSpec: Declarative description of a demo-selectable material.

    Bridges a material name to its physics engine and to the label/unit the
    frontend should use when reporting the layer's physical control variable
    (radius, diameter, composition, ...) instead of the raw [0, 1] gene.
    """

    def __init__(self, name, engine_factory, control_label, control_unit):
        self.name = name
        self.engine_factory = engine_factory
        self.control_label = control_label
        self.control_unit = control_unit

    def build_engine(self):
        return self.engine_factory()


def _confirmed_row(material_name):
    df = pd.read_csv(CONFIRMED_CSV_PATH)
    return df.set_index("Material").loc[material_name].to_dict()


def _build_cdse_engine():
    row = _confirmed_row("CdSe")
    return BrusEngine(
        bandgap=row["Eg_0K_eV"],
        alpha=row["Alpha_evK"],
        beta=row["Beta_K"],
        me_eff=row["me_eff"],
        mh_eff=row["mh_eff"],
        eps_r=row["epsilon_r"],
    )


DEMO_MATERIALS = {
    "CdSe": MaterialSpec("CdSe", _build_cdse_engine, "Radio del punto cuántico", "nm"),
    "PbS": MaterialSpec("PbS", PbSEngine, "Diámetro del punto cuántico", "nm"),
    "Perovskita yoduro bromuro": MaterialSpec(
        "Perovskita yoduro bromuro", PerovskiteEngine, "Composición (fracción de Br)", ""
    ),
}


def available_materials():
    return list(DEMO_MATERIALS.keys())


def is_valid_material(name):
    return name in DEMO_MATERIALS
