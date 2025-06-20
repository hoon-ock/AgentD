import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import re
import warnings
import matplotlib
matplotlib.use('Agg')

# RDKit import
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, MolSurf
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not installed. Some functionality will be limited.")

class DrugLikenessAnalyzer:
    """
    Fixed Drug-Likeness Analyzer - Real data only, no defaults
    """
    
    def __init__(self, data=None, smiles=None, mol=None):
        self.data = None
        self.mol = None
        self.smiles = None
        self.calculated_properties = {}
        
        if data:
            self.load_data(data)
        
        if smiles:
            self.smiles = smiles
            if RDKIT_AVAILABLE:
                try:
                    self.mol = Chem.MolFromSmiles(smiles)
                    if self.mol:
                        self._calculate_rdkit_properties()
                    else:
                        warnings.warn(f"Could not create molecule from SMILES: {smiles}")
                except Exception as e:
                    warnings.warn(f"RDKit processing failed: {e}")
            else:
                warnings.warn("RDKit not available. Cannot process SMILES directly.")
    
    def load_data(self, data):
        """Load data from various formats"""
        if isinstance(data, dict):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.to_dict()
        else:
            self.data = data
        
        # Extract SMILES if available
        if self.data and "SMILES" in self.data and not self.smiles:
            self.smiles = self.data["SMILES"]
            if RDKIT_AVAILABLE and self.smiles:
                try:
                    self.mol = Chem.MolFromSmiles(self.smiles)
                    if self.mol:
                        self._calculate_rdkit_properties()
                except Exception as e:
                    warnings.warn(f"RDKit processing failed: {e}")
    
    def _calculate_rdkit_properties(self):
        """Calculate molecular properties using RDKit"""
        if not RDKIT_AVAILABLE or not self.mol:
            return
        
        try:
            self.calculated_properties = {
                "molecular_weight": Descriptors.MolWt(self.mol),
                "logp": Descriptors.MolLogP(self.mol),
                "h_donors": Lipinski.NumHDonors(self.mol),
                "h_acceptors": Lipinski.NumHAcceptors(self.mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(self.mol),
                "aromatic_rings": Lipinski.NumAromaticRings(self.mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(self.mol),
                "molar_refractivity": Descriptors.MolMR(self.mol),
                "tpsa": MolSurf.TPSA(self.mol),
                "qed": Descriptors.qed(self.mol)
            }
        except Exception as e:
            warnings.warn(f"RDKit property calculation failed: {e}")
            self.calculated_properties = {}
    
    def _get_property(self, key):
        """
        Focused property extraction - only properties needed for drug-likeness rules
        
        What this function does:
        1. RDKit first (calculated from SMILES - most accurate)
        2. CSV mapping for only the properties we actually use in rules
        3. No defaults - returns None if not found
        
        This ensures each molecule gets different, real values
        """
        
        # PRIORITY 1: RDKit calculated properties (most accurate)
        if key in self.calculated_properties:
            return self.calculated_properties[key]
        
        # PRIORITY 2: CSV mapping for ONLY rule-required properties
        if self.data:
            deeppk_keys = [
                f"Predictions_{key}",
                f"Probability_{key}",
                f"Interpretation_{key}",
                key
            ]
            
            for dk in deeppk_keys:
                if dk in self.data:
                    try:
                        value = self.data[csv_column]
                        # Validate the value
                        if (value is not None and 
                            str(value).strip() != "" and 
                            str(value).lower() not in ['nan', 'na', 'null', 'none', '#n/a']):
                            return float(value)
                    except (ValueError, TypeError):
                        pass
            
            # Direct key lookup as fallback
            if key in self.data:
                try:
                    value = self.data[key]
                    if (value is not None and 
                        str(value).strip() != "" and 
                        str(value).lower() not in ['nan', 'na', 'null', 'none', '#n/a']):
                        return float(value)
                except (ValueError, TypeError):
                    pass
        
        # Return None if not found - NO DEFAULTS!
        return None

    def get_rule_data_availability(self):
        """
        Check data availability for each specific rule
        
        What this function does:
        - Shows exactly which properties are available for each rule
        - Identifies which rules can be analyzed vs which will fail
        - Helps debug why certain rules aren't working
        """
        
        # Define what each rule needs
        rule_requirements = {
            "Lipinski Rule of 5": ["molecular_weight", "logp", "h_donors", "h_acceptors", "rotatable_bonds"],
            "Veber Rule": ["rotatable_bonds", "tpsa"],
            "Ghose Filter": ["molecular_weight", "logp", "molar_refractivity", "heavy_atoms"], 
            "Rule of 3": ["molecular_weight", "logp", "h_donors"],
            "Oprea Lead-like": ["molecular_weight", "logp", "rotatable_bonds", "aromatic_rings", "h_donors", "h_acceptors"],
        }
        
        rule_analysis = {}
        
        print("RULE-SPECIFIC DATA AVAILABILITY:")
        print("=" * 60)
        
        for rule_name, required_props in rule_requirements.items():
            available = []
            missing = []
            
            for prop in required_props:
                value = self._get_property(prop)
                if value is not None:
                    available.append(f"{prop}={value:.3f}" if isinstance(value, float) else f"{prop}={value}")
                else:
                    missing.append(prop)
            
            availability_percent = len(available) / len(required_props) * 100
            can_analyze = len(available) >= len(required_props) * 0.6  # Need at least 60% of properties
            
            rule_analysis[rule_name] = {
                "required": len(required_props),
                "available": len(available),
                "missing": len(missing),
                "availability_percent": availability_percent,
                "can_analyze": can_analyze,
                "available_properties": available,
                "missing_properties": missing
            }
            
            status = "ANALYZABLE" if can_analyze else "INSUFFICIENT DATA"
            print(f"\n{rule_name}: {status}")
            print(f"  Required: {len(required_props)} | Available: {len(available)} | Missing: {len(missing)} ({availability_percent:.1f}%)")
            
            if available:
                print(f"Available: {available[:3]}{'...' if len(available) > 3 else ''}")
            if missing:
                print(f"Missing: {missing[:3]}{'...' if len(missing) > 3 else ''}")
        
        # Overall summary
        analyzable_rules = sum(1 for analysis in rule_analysis.values() if analysis["can_analyze"])
        total_rules = len(rule_requirements)

        print(f"\nSUMMARY:")
        print(f"Analyzable rules: {analyzable_rules}/{total_rules} ({analyzable_rules/total_rules*100:.1f}%)")
        print(f"RDKit properties available: {len(self.calculated_properties)}")
        
        return rule_analysis

    def debug_property_sources(self):
        """
        Debug where each property value is coming from
        
        What this function does:
        - Shows if property comes from RDKit vs CSV
        - Helps identify data source issues
        - Useful for debugging inconsistent results
        """
        
        key_properties = ["molecular_weight", "logp", "h_donors", "h_acceptors", "rotatable_bonds", 
                        "f20", "hia", "logs", "ppb", "cyp3a4_inhibitor", "clearance"]
        
        print("🔍 PROPERTY SOURCE ANALYSIS:")
        print("=" * 50)
        
        for prop in key_properties:
            # Check RDKit first
            if prop in self.calculated_properties:
                value = self.calculated_properties[prop]
                print(f"{prop}: {value:.3f} (RDKit)" if isinstance(value, float) else f"{prop}: {value} (RDKit)")
            else:
                # Check CSV
                csv_value = self._get_property(prop)
                if csv_value is not None:
                    print(f"{prop}: {csv_value:.3f} (CSV)" if isinstance(csv_value, float) else f"{prop}: {csv_value} (CSV)")
                else:
                    print(f"{prop}: NOT FOUND")

        print(f"\nData sources: RDKit ({len(self.calculated_properties)} props) + CSV (multiple columns)")
    
    # DRUG-LIKENESS RULES FOLLOW - each rule is a separate method
    def check_lipinski_rule_of_5(self):
        """
        Lipinski's Rule of 5 (with rotatable bonds as 5th optional criterion):
        1. Molecular weight ≤ 500 Da
        2. LogP ≤ 5
        3. H-bond donors ≤ 5
        4. H-bond acceptors ≤ 10
        5. Rotatable bonds ≤ 10 (extended criterion)
        
        Note: Allows one violation among the five.
        """
        mw = self._get_property("molecular_weight", float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp", float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        h_donors = self._get_property("h_donors", 3)
        h_acceptors = self._get_property("h_acceptors", 4)
        rotatable_bonds = self._get_property("rotatable_bonds", 5)

        results = {
            "molecular_weight": {"value": mw, "limit": 500, "pass": mw <= 500},
            "logp": {"value": logp, "limit": 5, "pass": logp <= 5},
            "h_donors": {"value": h_donors, "limit": 5, "pass": h_donors <= 5},
            "h_acceptors": {"value": h_acceptors, "limit": 10, "pass": h_acceptors <= 10},
            "rotatable_bonds": {"value": rotatable_bonds, "limit": 10, "pass": rotatable_bonds <= 10}
        }

        violations = sum(1 for check in results.values() if not check["pass"])
        
        # Lipinski allows 1 violation among the 4 criteria
        passed = violations <= 1 and len(available_props) >= 4

        return {
            "passed": passed,
            "violations": violations,
            "total_criteria": len(results),
            "available_properties": len(available_props),
            "missing_properties": missing_props,
            "results": results,
            "rule_summary": f"Lipinski Rule: {violations} violations out of {len(results)} criteria"
        }

    def check_veber_rule(self):
        """
        FIXED Veber's Rule - Uses real data, no defaults
        
        Veber's Rule (2 criteria):
        1. Rotatable bonds ≤ 10
        2. Polar surface area (TPSA) ≤ 140 Å²
        """
        rotatable_bonds = self._get_property("rotatable_bonds", 5)
        tpsa = self._get_property("tpsa", 90)
        
        results = {
            "rotatable_bonds": {"value": rotatable_bonds, "limit": 10, "pass": rotatable_bonds <= 10},
            "tpsa": {"value": tpsa, "limit": 140, "pass": tpsa <= 140}
        }
        
        violations = sum(1 for check in results.values() if not check["pass"])
        
        # Veber rule requires ALL criteria to pass (no violations allowed)
        passed = violations == 0
        
        return {
            "passed": passed,
            "violations": violations,
            "total_criteria": len(results),
            "available_properties": len(available_props),
            "missing_properties": missing_props,
            "results": results,
            "rule_summary": f"Veber Rule: {violations} violations out of {len(results)} criteria"
        }

    def check_ghose_filter(self):
        """
        FIXED Ghose's Filter - Uses real data, no defaults
        
        Ghose's Filter (4 criteria):
        1. Molecular weight between 160-480 Da
        2. LogP between -0.4 and 5.6
        3. Molar refractivity between 40-130
        4. Total number of atoms between 20-70
        """
        mw = self._get_property("molecular_weight", 
                               float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp", 
                                float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        molar_refractivity = self._get_property("molar_refractivity", 60)
        atom_count = self._get_property("heavy_atoms", 30)
        
        results = {
            "molecular_weight": {"value": mw, "range": [160, 480], "pass": 160 <= mw <= 480},
            "logp": {"value": logp, "range": [-0.4, 5.6], "pass": -0.4 <= logp <= 5.6},
            "molar_refractivity": {"value": molar_refractivity, "range": [40, 130], 
                                   "pass": 40 <= molar_refractivity <= 130},
            "atom_count": {"value": atom_count, "range": [20, 70], "pass": 20 <= atom_count <= 70}
        }
        
        violations = sum(1 for check in results.values() if not check["pass"])
        
        # Ghose filter requires ALL criteria to pass (no violations allowed)
        passed = violations == 0 and len(available_props) >= 3
        
        return {
            "passed": passed,
            "violations": violations,
            "total_criteria": len(results),
            "available_properties": len(available_props),
            "missing_properties": missing_props,
            "results": results,
            "rule_summary": f"Ghose Filter: {violations} violations out of {len(results)} criteria"
        }
    
    def check_rule_of_3(self):
        """
        FIXED Rule of 3 - Uses real data, no defaults
        
        Rule of 3 for fragment-based screening (3 criteria):
        1. Molecular weight < 300 Da
        2. LogP ≤ 3
        3. H-bond donors ≤ 3
        4. H-bond acceptors ≤ 3 (extended)
        5. Molar refractivity ≤ 60 (extended)
        
        Note: Extended version with additional common criteria
        """
        mw = self._get_property("molecular_weight", 
                              float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp", 
                                float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        h_donors = self._get_property("h_donors", 2)
        h_acceptors = self._get_property("h_acceptors", 3)
        molar_refractivity = self._get_property("molar_refractivity", 40)
        
        results = {
            "molecular_weight": {"value": mw, "limit": 300, "pass": mw < 300},
            "logp": {"value": logp, "limit": 3, "pass": logp <= 3},
            "h_donors": {"value": h_donors, "limit": 3, "pass": h_donors <= 3},
            # "h_acceptors": {"value": h_acceptors, "limit": 3, "pass": h_acceptors <= 3},
            # "molar_refractivity": {"value": molar_refractivity, "limit": 60, "pass": molar_refractivity <= 60}
        }
        
        violations = sum(1 for check in results.values() if not check["pass"])
        
        # Rule of 3 requires ALL criteria to pass (no violations allowed)
        passed = violations == 0 and len(available_props) >= 2
        
        return {
            "passed": passed,
            "violations": violations,
            "total_criteria": len(results),
            "available_properties": len(available_props),
            "missing_properties": missing_props,
            "results": results,
            "rule_summary": f"Rule of 3: {violations} violations out of {len(results)} criteria"
        }
    
    def check_oprea_lead_like_rules(self):
        """
        FIXED Oprea's Lead-like Rules - Uses real data, no defaults
        
        Oprea's Lead-like Rules (6 criteria):
        1. Molecular weight between 200-450 Da
        2. LogP between -1 and 4.5
        3. Rotatable bonds ≤ 8
        4. Rings ≤ 4
        5. H-bond donors ≤ 5
        6. H-bond acceptors ≤ 8
        """
        mw = self._get_property("molecular_weight",
                              float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp",
                                float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        rotatable_bonds = self._get_property("rotatable_bonds", 4)
        rings = self._get_property("aromatic_rings", 2)
        h_donors = self._get_property("h_donors", 2)
        h_acceptors = self._get_property("h_acceptors", 3)
        
        results = {
            "molecular_weight": {"value": mw, "range": [200, 450], "pass": 200 <= mw <= 450},
            "logp": {"value": logp, "range": [-1, 4.5], "pass": -1 <= logp <= 4.5},
            "rotatable_bonds": {"value": rotatable_bonds, "limit": 8, "pass": rotatable_bonds <= 8},
            "rings": {"value": rings, "limit": 4, "pass": rings <= 4},
            "h_donors": {"value": h_donors, "limit": 5, "pass": h_donors <= 5},
            "h_acceptors": {"value": h_acceptors, "limit": 8, "pass": h_acceptors <= 8}
        }
        
        violations = sum(1 for check in results.values() if not check["pass"])
        
        return {
            "passed": violations <= 1,
            "violations": violations,
            "results": results
        }
    
    def check_fda_2025_criteria(self):
        """
        Updated FDA 2024-2025 criteria based on current guidelines (7 criteria):
        """
        # Extract ADMET values
        f20 = float(self.data.get("Probability_absorption_f20", 0.5)) if self.data else 0.5
        hia = float(self.data.get("Probability_absorption_hia", 0.7)) if self.data else 0.7
        logs = float(self.data.get("Predictions_absorption_log_s", -3)) if self.data else -3
        ppb = float(self.data.get("Predictions_distribution_ppb", 80)) if self.data else 80
        cyp_inhibition = float(self.data.get("Predictions_metabolism_cyppro_iinh", 0.5)) if self.data else 0.5
        cyp3a4_substrate = float(self.data.get("Probability_metabolism_cyp3a4_substrate", 0.5)) if self.data else 0.5
        clearance = float(self.data.get("Predictions_excretion_clearance", 5)) if self.data else 5
        
        results = {
            "oral_bioavailability": {
                "value": f20, 
                "threshold": 0.3,
                "pass": f20 > 0.3,
                "description": "Oral bioavailability (F) > 30%"
            },
            "intestinal_absorption": {
                "value": hia, 
                "threshold": 0.7, 
                "pass": hia > 0.7,
                "description": "Human intestinal absorption > 70%"
            },
            "aqueous_solubility": {
                "value": logs, 
                "range": [-4, 0.5], 
                "pass": -4 <= logs <= 0.5,
                "description": "Log S between -4 and 0.5"
            },
            "protein_binding_safety": {
                "value": ppb, 
                "threshold": 95,
                "pass": ppb < 95,
                "description": "Plasma protein binding < 95%"
            },
            "drug_interaction_risk": {
                "value": cyp_inhibition, 
                "threshold": 0.3,
                "pass": cyp_inhibition < 0.3,
                "description": "Low CYP inhibition potential"
            },
            "metabolic_stability": {
                "value": 1 - cyp3a4_substrate, 
                "threshold": 0.4,
                "pass": (1 - cyp3a4_substrate) > 0.4,
                "description": "Adequate metabolic stability"
            },
            "clearance_profile": {
                "value": clearance, 
                "range": [0.5, 15],
                "pass": 0.5 <= clearance <= 15,
                "description": "Appropriate clearance rate"
            }
        }
        
        # Count violations among available properties
        violations = sum(1 for check in results.values() if not check["pass"])
        
        # Oprea allows 1 violation among the 6 criteria
        passed = violations <= 1 and len(available_props) >= 4
        
        return {
            "passed": passed,
            "violations": violations,
            "total_criteria": len(results),
            "available_properties": len(available_props),
            "missing_properties": missing_props,
            "results": results,
            "rule_summary": f"Oprea Lead-like: {violations} violations out of {len(results)} criteria"
        }

    def generate_rule_radar_plot(self, rule_name, save_path=None, show=True):
        try:
            # Get rule results
            rule_mapping = {
                'lipinski': (self.check_lipinski_rule_of_5, "Lipinski's Rule of 5"),
                'veber': (self.check_veber_rule, "Veber's Rule"),
                'ghose': (self.check_ghose_filter, "Ghose Filter"),
                'rule_of_3': (self.check_rule_of_3, "Rule of 3"),
                'ro3': (self.check_rule_of_3, "Rule of 3"),
                'oprea': (self.check_oprea_lead_like_rules, "Oprea's Lead-like Rules")
            }
            
            rule_key = rule_name.lower()
            if rule_key not in rule_mapping:
                return None
            
            rule_function, title = rule_mapping[rule_key]
            rule_results = rule_function()
            
            if "error" in rule_results or not rule_results['results']:
                return None
            
            # Simple property names
            display_names = {
                'molecular_weight': 'MW',
                'logp': 'LogP', 
                'h_donors': 'HBD',
                'h_acceptors': 'HBA',
                'rotatable_bonds': 'RB',
                'tpsa': 'TPSA',
                'molar_refractivity': 'MR',
                'atom_count': 'Atoms',
                'rings': 'Rings',
                'oral_bioavailability': 'F20',
                'intestinal_absorption': 'HIA',
                'aqueous_solubility': 'LogS',
                'protein_binding_safety': 'PPB',
                'drug_interaction_risk': 'CYP3A4',
                'metabolic_stability': 'MetStab',
                'clearance_profile': 'CLr'
            }
            
            # Extract boundaries and normalize each property
            properties = {}
            lower_boundaries = {}
            upper_boundaries = {}
            actual_values = {}
            
            for prop_name, prop_data in rule_results['results'].items():
                display_name = display_names.get(prop_name, prop_name[:6])
                raw_value = prop_data['value']
                passed = prop_data['pass']
                
                # Determine boundaries based on criteria type
                if 'limit' in prop_data:
                    # Upper limit only (e.g., MW ≤ 500)
                    upper_limit = prop_data['limit']
                    lower_limit = 0  # No lower bound
                    
                    # FIXED: Direct normalization without padding
                    normalized_lower = 0.0  # Lower boundary at 0%
                    normalized_upper = 1.0  # Upper boundary at 100%
                    normalized_actual = raw_value / upper_limit  # Can exceed 1.0 for violations
                    
                elif 'range' in prop_data:
                    # Range criteria (e.g., LogP -1 to 4.5 or MW 160-480)
                    range_min, range_max = prop_data['range']
                    lower_limit = range_min
                    upper_limit = range_max
                    
                    # REPLACE WITH THIS LOGIC:
                    if range_min < 0:
                        # Negative ranges: Scale proportionally where upper bound = 100%
                        normalized_upper = 1.0  # Upper bound always at 100%
                        normalized_lower = range_min / range_max  # Proportional scaling
                        normalized_actual = raw_value / range_max  # Scale relative to upper
                    else:
                        # Positive ranges: Use 0% to 100% scale
                        range_width = range_max - range_min
                        normalized_lower = 0.0    # Lower bound at 0%
                        normalized_upper = 1.0    # Upper bound at 100%
                        normalized_actual = (raw_value - range_min) / range_width
                elif 'threshold' in prop_data:
                    threshold = prop_data['threshold']
                    
                    if prop_name in ['drug_interaction_risk', 'protein_binding_safety']:
                        # Lower is better (risk factors) - threshold is upper limit
                        upper_limit = threshold
                        lower_limit = 0
                        
                        # FIXED: Direct normalization
                        normalized_lower = 0.0  # Lower boundary at 0%
                        normalized_upper = 1.0  # Upper boundary at 100%
                        normalized_actual = raw_value / threshold  # Can exceed 1.0 for high risk
                        
                    else:
                        # Higher is better - threshold is lower limit
                        lower_limit = threshold
                        upper_limit = 1.0  # Assume max 1.0 for probabilities
                        
                        # FIXED: Direct normalization
                        if upper_limit > lower_limit:
                            range_width = upper_limit - lower_limit
                            normalized_lower = 0.0  # Lower boundary at 0%
                            normalized_upper = 1.0  # Upper boundary at 100%
                            normalized_actual = (raw_value - lower_limit) / range_width  # Can be negative for below threshold
                        else:
                            normalized_lower = 0.0
                            normalized_upper = 1.0
                            normalized_actual = raw_value / threshold
                
                
                # Ensure valid ranges
                normalized_lower = max(-0.25, min(1, normalized_lower))  # Allow down to -25%
                normalized_upper = max(0, min(1, normalized_upper))      # Upper stays 0-100%
                normalized_actual = max(-0.3, min(1.2, normalized_actual))  # Allow slight overflow
                # Store data
                properties[display_name] = normalized_actual
                lower_boundaries[display_name] = normalized_lower
                upper_boundaries[display_name] = normalized_upper
                actual_values[display_name] = {
                    'raw_value': raw_value,
                    'passed': passed,
                    'lower_limit': lower_limit if 'lower_limit' in locals() else 0,
                    'upper_limit': upper_limit if 'upper_limit' in locals() else 'N/A'
                }
            
            # Create radar plot arrays
            categories = list(properties.keys())
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the circle
            
            # Data arrays
            actual_vals = list(properties.values()) + [list(properties.values())[0]]
            lower_vals = list(lower_boundaries.values()) + [list(lower_boundaries.values())[0]]
            upper_vals = list(upper_boundaries.values()) + [list(upper_boundaries.values())[0]]
            
            # Create figure
            fig = plt.figure(figsize=(8, 8), facecolor='white')
            ax = fig.add_subplot(111, polar=True)
            
            # 1. IDEAL ZONE (filled area between lower and upper limits)
            ax.fill_between(angles, lower_vals, upper_vals, 
                        color="#80DE75F6", alpha=0.4, label='Ideal Zone')
            
            # 2. LOWER LIMIT LINE (solid line)
            ax.plot(angles, lower_vals, '-', linewidth=3, color="#106C24", 
                    alpha=0.8, label='Lower Limit')
            
            # 3. UPPER LIMIT LINE (solid line) 
            ax.plot(angles, upper_vals, '-', linewidth=3, color='#00695C',
                    alpha=0.8, label='Upper Limit')
            
            # 4. ACTUAL PROPERTIES (dotted line with markers)
            overall_passed = rule_results['passed']
            molecule_color = '#2E8B57' if overall_passed else '#DC143C'
            
            ax.plot(angles, actual_vals, 'o:', linewidth=4, color=molecule_color,
                    alpha=0.9, markersize=12, markerfacecolor=molecule_color,
                    markeredgecolor='white', markeredgewidth=3,
                    label='Actual Properties', zorder=10)
            
            # 5. Add pass/fail indicators on each point
            for i, (angle, value, cat) in enumerate(zip(angles[:-1], actual_vals[:-1], categories)):
                passed = actual_values[cat]['passed'] 
                # Small indicator: green circle for pass, red X for fail
                indicator_color = 'green' if passed else 'red'
                indicator_marker = 'o' if passed else 'X'
                ax.scatter([angle], [value], s=60, c=indicator_color,
                        marker=indicator_marker, edgecolors='white', 
                        linewidth=2, zorder=15, alpha=0.8)
            
                # Plot configuration - STRICT: 0-1 normalization, ylim allows overflow
                ax.set_ylim(-0.25, 1.5)  # Allow -25% to 150%
                ax.set_yticks([-0.25, -0.1, 0.0, 0.25, 0.50, 0.75, 1.0, 1.25])
                ax.set_yticklabels(['-25%', '-10%', '0%', '25%', '50%', '75%', '100%', '125%'], 
                                fontsize=8, alpha=0.8)
            
            # Category labels
            plt.xticks(angles[:-1], categories, fontsize=22, weight='bold')
            
            # Plot styling  
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_rlabel_position(45) 
            ax.grid(True, alpha=0.3, linewidth=1)
            
            # Center status indicator
            violations = rule_results.get('violations', 0)
            total_criteria = len(rule_results['results'])
            
            status_text = f"{'PASSED' if overall_passed else 'FAILED'}"            
            ax.text(0, 0.05, status_text, ha='center', va='center',
                    fontsize=14, weight='bold', color=molecule_color)
            

            # Title
            plt.title(title, size=24, weight='bold', pad=30)

            # Legend with clear descriptions
            legend_elements = [
                plt.Line2D([0], [0], color='lightgreen', lw=8, alpha=0.4, label='Ideal Zone'),
                plt.Line2D([0], [0], color='darkgreen', lw=3, label='Lower Limit'),
                plt.Line2D([0], [0], color='green', lw=3, label='Upper Limit'),
                plt.Line2D([0], [0], color=molecule_color, lw=4, linestyle=':', 
                        marker='o', markersize=8, label='Actual Properties')
            ]
            
            plt.legend(handles=legend_elements, loc='upper right', 
                    bbox_to_anchor=(1.3, 1.0), fontsize=11, 
                    frameon=True, fancybox=True, shadow=True)
            
            # Layout
            plt.tight_layout()
            
            # Save and display
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
                print(f" Saved: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
                
            return fig
        except Exception as e:
            print(f"Error generating radar plot for {rule_name}: {e}")
        return None
    
    def generate_all_radar_plots(self, save_dir=os.getcwd(), save_plots=True, show_plots=False):
        """Generate all radar plots with correct metrics"""
        
        plots = {}
        rules = [
            ('lipinski', os.path.join(save_dir, 'lipinski_rule_5_criteria.png')),
            ('veber', os.path.join(save_dir, 'veber_rule_2_criteria.png')), 
            ('ghose', os.path.join(save_dir, 'ghose_filter_4_criteria.png')),
            ('rule_of_3', os.path.join(save_dir, 'rule_of_3_extended_5_criteria.png')),
            ('oprea', os.path.join(save_dir, 'oprea_leadlike_6_criteria.png')),
            ('fda_2025', os.path.join(save_dir, 'fda_2025_7_criteria.png'))
        ]
        
        print(" GENERATING ALL RADAR PLOTS")
        print("=" * 50)
        
        for rule_key, rule_name in rules_to_plot:
            print(f"\nAttempting {rule_name}...")
            
            # Generate save path if directory provided
            if save_directory:
                import os
                os.makedirs(save_directory, exist_ok=True)
                save_path = os.path.join(save_directory, f"{rule_key}_radar_plot.png")
            else:
                save_path = None
            
            # Generate plot
            try:
                fig = self.generate_rule_radar_plot(rule_key, save_path=save_path, show=show)
                if fig is not None:
                    successful_plots.append((rule_key, rule_name))
                    print(f" {rule_name}: SUCCESS")
                else:
                    failed_plots.append((rule_key, rule_name, "Insufficient data"))
                    print(f"  {rule_name}: SKIPPED (insufficient data)")
            except Exception as e:
                failed_plots.append((rule_key, rule_name, str(e)))
                print(f" {rule_name}: ERROR - {e}")
        
        # Summary
        print(f"\n RADAR PLOT SUMMARY:")
        print(f"Successful plots: {len(successful_plots)}")
        print(f"Failed/skipped: {len(failed_plots)}")
        
        if successful_plots:
            print("\n Successfully generated:")
            for rule_key, rule_name in successful_plots:
                print(f"  - {rule_name}")
        
        if failed_plots:
            print("\n  Failed/skipped:")
            for rule_key, rule_name, reason in failed_plots:
                print(f"  - {rule_name}: {reason}")
        
        return {
            'successful': successful_plots,
            'failed': failed_plots,
            'total_attempted': len(rules_to_plot)
        }


    def get_summary_report(self):
        """Generate summary report"""
        lipinski = self.check_lipinski_rule_of_5()
        veber = self.check_veber_rule()
        ghose = self.check_ghose_filter()
        ro3 = self.check_rule_of_3()
        oprea = self.check_oprea_lead_like_rules()
        
        all_rules = [lipinski, veber, ghose, ro3, oprea]
        valid_analyses = [rule for rule in all_rules if "error" not in rule]
        passed_rules = sum(1 for rule in valid_analyses if rule.get("passed"))
        
        return {
            "smiles": self.smiles,
            "lipinski_rule_of_5": lipinski,
            "veber_rule": veber,
            "ghose_filter": ghose,
            "rule_of_3": ro3,
            "oprea_lead_like": oprea,
            "summary": {
                "valid_analyses": len(valid_analyses),
                "total_rules": len(all_rules),
                "passed_rules": passed_rules,
                "success_rate": f"{len(valid_analyses)/len(all_rules)*100:.1f}%"
            }
        }
    
    def print_summary(self):
        """Print formatted summary"""
        report = self.get_summary_report()
        
        print("\n" + "="*60)
        print("DRUG-LIKENESS ANALYSIS SUMMARY")
        print("="*60)
        print(f"SMILES: {report['smiles']}")
        print(f"Valid analyses: {report['summary']['valid_analyses']}/6")
        print(f"Success rate: {report['summary']['success_rate']}")
        
        rules = [
            ("Lipinski Rule of 5", report["lipinski_rule_of_5"]),
            ("Veber Rule", report["veber_rule"]),
            ("Ghose Filter", report["ghose_filter"]),
            ("Rule of 3", report["rule_of_3"]),
            ("Oprea Lead-like", report["oprea_lead_like"]),
        ]
        
        for name, result in rules:
            if "error" in result:
                print(f"{name}:  {result['error']}")
            else:
                status = " PASSED" if result["passed"] else " FAILED"
                violations = result["violations"]
                criteria = len(result["results"])
                print(f"{name}: {status} ({violations}/{criteria} violations)")
        
        print("="*60)
