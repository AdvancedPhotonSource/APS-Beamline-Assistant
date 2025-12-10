#!/usr/bin/env python3
"""
Fetch crystallographic data from Materials Project API

Requirements:
    pip install mp-api

Usage:
    1. Get API key from https://materialsproject.org/api
    2. Set environment variable: export MP_API_KEY="your_key_here"
    3. Run: python fetch_materials_from_mp.py

This will populate knowledge_base/data/materials.json with authoritative data.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False
    print("Warning: mp-api not installed. Install with: pip install mp-api")


def fetch_material_from_mp(formula: str, api_key: str) -> Dict[str, Any]:
    """Fetch material data from Materials Project"""
    with MPRester(api_key) as mpr:
        # Search for the material
        docs = mpr.materials.summary.search(formula=formula, num_chunks=1)

        if not docs:
            print(f"  ⚠️  Material {formula} not found in Materials Project")
            return None

        # Get the most stable structure
        doc = docs[0]
        structure = doc.structure

        return {
            "name": formula,
            "formula": doc.formula_pretty,
            "mp_id": doc.material_id,
            "space_group": doc.symmetry.number,
            "space_group_symbol": doc.symmetry.symbol,
            "crystal_system": doc.symmetry.crystal_system,
            "lattice_parameters": {
                "a": structure.lattice.a,
                "b": structure.lattice.b,
                "c": structure.lattice.c,
                "alpha": structure.lattice.alpha,
                "beta": structure.lattice.beta,
                "gamma": structure.lattice.gamma
            },
            "density_g_cm3": doc.density,
            "volume_angstrom3": structure.lattice.volume,
            "formation_energy_per_atom": doc.formation_energy_per_atom if hasattr(doc, 'formation_energy_per_atom') else None,
            "band_gap_ev": doc.band_gap if hasattr(doc, 'band_gap') else None,
            "is_stable": doc.is_stable if hasattr(doc, 'is_stable') else None,
            "source": "Materials Project",
            "source_url": f"https://materialsproject.org/materials/{doc.material_id}"
        }


def add_hedm_metadata(material_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Add HEDM-specific metadata (calibration use, typical rings, etc.)"""
    material_data.update(metadata)
    return material_data


# Common materials for HEDM
COMMON_MATERIALS = {
    # Calibration standards
    "CeO2": {
        "common_use": "calibration_standard",
        "typical_rings": 24,
        "nist_srm": "674b",
        "advantages": ["Strong diffraction", "Chemically stable", "Many rings"],
        "notes": "Excellent general-purpose calibrant"
    },
    "LaB6": {
        "common_use": "calibration_standard",
        "typical_rings": 15,
        "nist_srm": "660c",
        "advantages": ["NIST standard", "High precision", "Low thermal expansion"],
        "notes": "Best for high-precision calibrations"
    },
    "Si": {
        "common_use": "calibration_standard",
        "typical_rings": 12,
        "nist_srm": "640e",
        "advantages": ["Low absorption", "Good for high energies"],
        "notes": "Preferred for >80 keV beams"
    },
    "Al2O3": {
        "common_use": "calibration_standard",
        "typical_rings": 20,
        "nist_srm": "676a",
        "advantages": ["Chemically inert", "Non-cubic system"],
        "notes": "Useful for trigonal calibration"
    },

    # Common sample materials
    "Fe": {
        "common_use": "sample_material",
        "typical_rings": 8,
        "crystal_structure_notes": "BCC at room temperature",
        "phase_info": "Alpha-Fe (ferrite) below 912°C"
    },
    "Ti": {
        "common_use": "sample_material",
        "typical_rings": 12,
        "crystal_structure_notes": "HCP at room temperature",
        "phase_info": "Alpha-Ti (HCP) below 882°C"
    },
    "Ni": {
        "common_use": "sample_material",
        "typical_rings": 8,
        "crystal_structure_notes": "FCC structure",
        "applications": ["Superalloys", "Catalysts"]
    },
    "Al": {
        "common_use": "sample_material",
        "typical_rings": 8,
        "crystal_structure_notes": "FCC structure",
        "applications": ["Aerospace", "Automotive"]
    },
}


def main():
    """Main function to fetch and save materials data"""
    api_key = os.getenv("MP_API_KEY")

    if not api_key:
        print("❌ Error: MP_API_KEY environment variable not set")
        print("   Get your API key from https://materialsproject.org/api")
        print("   Then run: export MP_API_KEY='your_key_here'")
        return

    if not HAS_MP_API:
        print("❌ Error: mp-api not installed")
        print("   Install with: pip install mp-api")
        return

    print("🔍 Fetching materials from Materials Project...")
    print()

    materials_db = {
        "_metadata": {
            "description": "Materials database for HEDM calibration and analysis",
            "source": "Materials Project API + HEDM-specific metadata",
            "last_updated": "auto-generated",
            "total_materials": 0
        }
    }

    for formula, metadata in COMMON_MATERIALS.items():
        print(f"  Fetching: {formula}...")
        try:
            mp_data = fetch_material_from_mp(formula, api_key)
            if mp_data:
                # Combine Materials Project data with HEDM metadata
                combined_data = add_hedm_metadata(mp_data, metadata)
                materials_db[formula] = combined_data
                print(f"  ✓ {formula}: {combined_data.get('space_group_symbol')} (SG {combined_data.get('space_group')})")
            else:
                # Fallback: keep metadata only
                materials_db[formula] = {
                    "name": formula,
                    "source": "manual_entry",
                    **metadata
                }
                print(f"  ⚠️  {formula}: Using manual entry (not found in MP)")
        except Exception as e:
            print(f"  ❌ {formula}: Error - {e}")
            materials_db[formula] = {
                "name": formula,
                "source": "manual_entry",
                "error": str(e),
                **metadata
            }

    # Update metadata
    materials_db["_metadata"]["total_materials"] = len(materials_db) - 1  # Exclude metadata itself

    # Save to file
    output_path = Path(__file__).parent / "data" / "materials.json"
    with open(output_path, 'w') as f:
        json.dump(materials_db, f, indent=2)

    print()
    print(f"✅ Saved {materials_db['_metadata']['total_materials']} materials to {output_path}")
    print()
    print("Example materials:")
    for formula in list(COMMON_MATERIALS.keys())[:3]:
        if formula in materials_db:
            mat = materials_db[formula]
            sg = mat.get('space_group', 'N/A')
            use = mat.get('common_use', 'N/A')
            print(f"  • {formula}: SG {sg}, {use}")


if __name__ == "__main__":
    main()
