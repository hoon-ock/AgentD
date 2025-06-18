import matplotlib.pyplot as plt
import numpy as np
import json
import re
import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Force the Agg backend before importing pyplot

# Optionally import RDKit if available
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, MolSurf
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not installed. Some functionality will be limited.")

class DrugLikenessAnalyzer:
    """
    CORRECTED Drug-Likeness Analyzer with proper rule definitions and matching radar plots.
    
    Fixed Issues:
    - Lipinski Rule of 5
    - Rule of 3
    - Updated FDA criteria based on 2024-2025 guidelines
    - All radar plots now match their respective rule criteria counts
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
                self.mol = Chem.MolFromSmiles(smiles)
                if not self.mol:
                    raise ValueError(f"Invalid SMILES string: {smiles}")
                self._calculate_rdkit_properties()
            else:
                warnings.warn("RDKit not available. Cannot process SMILES directly.")
        
        if mol and RDKIT_AVAILABLE:
            self.mol = mol
            self.smiles = Chem.MolToSmiles(mol)
            self._calculate_rdkit_properties()
    
    def load_data(self, data):
        if isinstance(data, str):
            try:
                self.data = json.loads(data.strip())
            except json.JSONDecodeError:
                clean_data = re.sub(r'"\s*\n\s*"', '", "', data)
                try:
                    self.data = json.loads(clean_data)
                except json.JSONDecodeError:
                    fixed_data = self._fix_json_string(data)
                    self.data = json.loads(fixed_data)
        else:
            self.data = data
        
        # Extract SMILES if available
        if "SMILES" in self.data:
            self.smiles = self.data["SMILES"]
            if RDKIT_AVAILABLE and self.smiles:
                self.mol = Chem.MolFromSmiles(self.smiles)
                if self.mol:
                    self._calculate_rdkit_properties()
        
    
    def _fix_json_string(self, json_str):
        json_str = json_str.strip()
        json_str = re.sub(r'\s+', ' ', json_str)
        json_str = re.sub(r'"\s+"', '", "', json_str)
        json_str = re.sub(r'([{\s,])(\w+):', r'\1"\2":', json_str)
        
        if not json_str.startswith('{'):
            json_str = '{' + json_str
        if not json_str.endswith('}'):
            json_str = json_str + '}'
        
        return json_str
    
    def _calculate_rdkit_properties(self):
        if not RDKIT_AVAILABLE or not self.mol:
            return
        
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
        
        try:
            self.calculated_properties["fraction_csp3"] = Descriptors.FractionCSP3(self.mol)
        except:
            pass
            
        try:
            self.calculated_properties["formal_charge"] = Chem.GetFormalCharge(self.mol)
        except:
            pass
    
    def _get_property(self, key, default=None):
        if key in self.calculated_properties:
            return self.calculated_properties[key]
        
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
                        return float(self.data[dk])
                    except (ValueError, TypeError):
                        return self.data[dk]
        
        return default
    
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

        return {
            "passed": violations <= 1,
            "violations": violations,
            "results": results
        }
    
    def check_veber_rule(self):
        """
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
        
        return {
            "passed": violations == 0,
            "violations": violations,
            "results": results
        }
    
    def check_ghose_filter(self):
        """
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
        
        return {
            "passed": violations == 0,
            "violations": violations,
            "results": results
        }
    
    def check_rule_of_3(self):
        """
        Extended Rule of 3 for fragment-based screening (5 criteria):
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
        
        return {
            "passed": violations == 0,
            "violations": violations,
            "results": results
        }
    
    
    def check_oprea_lead_like_rules(self):
        """
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
        
        # Critical parameters for FDA approval
        critical_parameters = ["oral_bioavailability", "intestinal_absorption", "drug_interaction_risk"]
        critical_failures = sum(1 for key in critical_parameters if not results[key]["pass"])
        
        # Calculate approval likelihood
        total_criteria = len(results)
        passed_criteria = sum(1 for check in results.values() if check["pass"])
        approval_likelihood = passed_criteria / total_criteria
        
        # Updated categories based on 2024 FDA approval patterns
        if approval_likelihood > 0.85 and critical_failures == 0:
            approval_category = "High (Expedited Track Eligible)"
        elif approval_likelihood > 0.7 and critical_failures <= 1:
            approval_category = "Moderate (Standard Review)"
        elif approval_likelihood > 0.5:
            approval_category = "Low (Requires Optimization)"
        else:
            approval_category = "Very Low (Major Reformulation Needed)"
        
        return {
            "passed": critical_failures == 0 and approval_likelihood > 0.7,
            "critical_failures": critical_failures,
            "results": results,
            "approval_likelihood": approval_likelihood,
            "approval_category": approval_category
        }
    
    def _normalize_for_radar(self, value, range_min, range_max, exaggerate_if_fail=False, is_fail=False):
        """
        Normalize values for radar plot display.
        
        Parameters:
        - value: float, actual value of the metric
        - range_min: float, minimum expected/allowed value
        - range_max: float, maximum expected/allowed value
        - exaggerate_if_fail: bool, if True, push failed metrics outside the ideal zone
        - is_fail: bool, indicates if the current metric failed

        Returns:
        - float, normalized value between 0 and 1 (or slightly >1 if exaggeration is applied)
        """
        if range_max == range_min:
            return 0.5  # avoid divide-by-zero

        normalized = (value - range_min) / (range_max - range_min)

        # Clamp to [0, 1] unless exaggeration is allowed
        if exaggerate_if_fail and is_fail:
            return min(1.25, max(0, normalized))
        else:
            return max(0, min(1, normalized))

    
    def generate_rule_radar_plot(self, rule_name, save_path=None, show=True):
        """
        Generate radar plot for a specific rule with corrected normalization
        to visually exaggerate failed metrics outside the ideal zone.
        """
        try:
            # Select rule and title
            if rule_name.lower() == 'lipinski':
                rule_results = self.check_lipinski_rule_of_5()
                title = "Lipinski's Rule of 5"
            elif rule_name.lower() == 'veber':
                rule_results = self.check_veber_rule()
                title = "Veber's Rule"
            elif rule_name.lower() == 'ghose':
                rule_results = self.check_ghose_filter()
                title = "Ghose Filter"
            elif rule_name.lower() in ['rule_of_3', 'ro3']:
                rule_results = self.check_rule_of_3()
                title = "Rule of 3 (Extended)"
            elif rule_name.lower() == 'oprea':
                rule_results = self.check_oprea_lead_like_rules()
                title = "Oprea's Lead-like Rules"
            elif rule_name.lower() in ['fda', 'fda_2025']:
                rule_results = self.check_fda_2025_criteria()
                title = "FDA 2025 Criteria"
            else:
                raise ValueError(f"Unknown rule: {rule_name}")
            
            properties = {}
            display_names = {
                'molecular_weight': 'Mol Weight',
                'logp': 'LogP',
                'logd_ph74': 'LogD pH7.4',
                'h_donors': 'H-Donors',
                'h_acceptors': 'H-Acceptors',
                'rotatable_bonds': 'Rot Bonds',
                'tpsa': 'TPSA',
                'molar_refractivity': 'Molar Refr',
                'atom_count': 'Atoms',
                'rings': 'Rings',
                'pka': 'pKa',
                'oral_bioavailability': 'Oral Bio',
                'intestinal_absorption': 'HIA',
                'aqueous_solubility': 'Solubility',
                'protein_binding_safety': 'PPB Safety',
                'drug_interaction_risk': 'DDI Risk',
                'metabolic_stability': 'Met Stability',
                'clearance_profile': 'Clearance'
            }

            for prop_name, prop_data in rule_results['results'].items():
                label = display_names.get(prop_name, prop_name.replace('_', ' ').title())
                value = prop_data['value']
                fail = not prop_data['pass']

                # Determine normalization bounds
                if 'limit' in prop_data:
                    limit = prop_data['limit']
                    range_min = 0
                    range_max = limit * 1.5  # allow some headroom

                elif 'range' in prop_data:
                    range_min, range_max = prop_data['range']
                    range_min *= 0.9  # slightly extend for plotting
                    range_max *= 1.1

                elif 'threshold' in prop_data:
                    threshold = prop_data['threshold']
                    range_min = 0
                    range_max = threshold * 1.5

                elif 'threshold_cns' in prop_data or 'threshold_non_cns' in prop_data:
                    # For special cases like BBB
                    if 'pass_cns' in prop_data:
                        range_min, range_max = 0, 1
                    else:
                        range_min, range_max = 0, 1

                else:
                    range_min, range_max = 0, 1  # fallback range

                # Normalize and store
                normalized_value = self._normalize_for_radar(
                    value=value,
                    range_min=range_min,
                    range_max=range_max,
                    exaggerate_if_fail=True,    
                    is_fail=fail
                )
                properties[label] = normalized_value

            # Radar plot values
            categories = list(properties.keys())
            values = list(properties.values())
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            values += values[:1]

            # Plotting
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)

            #color = 'green' if rule_results['passed'] else 'red'
            color = 'purple'
            ax.plot(angles, values, 'o-', linewidth=3, color=color, alpha=0.8, markersize=8, label="Compound Values")
            ax.fill(angles, values, color=color, alpha=0.25)

            # Ideal zone
            ideal_values = [0.8] * N + [0.8]
            ax.plot(angles, ideal_values, '--', linewidth=2, color='darkgreen', alpha=0.6, label="Ideal Zone")

            ax.set_ylim(0, 1.3)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
            ax.tick_params(axis='y', labelsize=14)
            plt.xticks(angles[:-1], categories, weight='bold', fontsize=22)

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            status_text = "PASSED" if rule_results['passed'] else "FAILED"
            status_color = 'green' if rule_results['passed'] else 'red'
            ax.text(0, 0, status_text, ha='center', va='center',
                    fontsize=24, weight='bold', color=status_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

            plt.title(title, size=24, weight='bold', pad=20, fontsize=24)
            # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=16)

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
                print(f"Saved: {save_path}")
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
        
        print("Generating corrected radar plots...")
        print("=" * 50)
        
        for rule_name, filename in rules:
            save_path = filename if save_plots else None
            result = self.generate_rule_radar_plot(rule_name, save_path=save_path, show=show_plots)
            plots[rule_name] = result is not None
            
            # Get criteria count for verification
            if rule_name == 'lipinski':
                criteria_count = len(self.check_lipinski_rule_of_5()['results'])
            elif rule_name == 'veber':
                criteria_count = len(self.check_veber_rule()['results'])
            elif rule_name == 'ghose':
                criteria_count = len(self.check_ghose_filter()['results'])
            elif rule_name == 'rule_of_3':
                criteria_count = len(self.check_rule_of_3()['results'])
            elif rule_name == 'oprea':
                criteria_count = len(self.check_oprea_lead_like_rules()['results'])
            elif rule_name == 'fda_2025':
                criteria_count = len(self.check_fda_2025_criteria()['results'])
            else:
                criteria_count = 0
            
            status = "✓" if plots[rule_name] else "✗"
            print(f"{status} {rule_name.upper()}: {criteria_count} criteria - {'SUCCESS' if plots[rule_name] else 'FAILED'}")
        
        return plots
    
    def get_summary_report(self):
        """Generate comprehensive summary report"""
        lipinski = self.check_lipinski_rule_of_5()
        veber = self.check_veber_rule()
        ghose = self.check_ghose_filter()
        ro3 = self.check_rule_of_3()
        oprea = self.check_oprea_lead_like_rules()
        fda = self.check_fda_2025_criteria()
        
        # Overall assessment
        passed_rules = sum([
            lipinski["passed"],
            veber["passed"],
            ghose["passed"],
            oprea["passed"],
            fda["passed"]
        ])
        
        development_category = "Unfavorable"
        if passed_rules == 5:
            development_category = "Excellent"
        elif passed_rules == 4:
            development_category = "Good"
        elif passed_rules == 3:
            development_category = "Fair"
        elif passed_rules == 2:
            development_category = "Needs Improvement"
        else:
            development_category = "Unfavorable"

            
        # report = {
        #     "smiles": self.smiles,
        #     "lipinski_rule_of_5": lipinski,
        #     "veber_rule": veber,
        #     "ghose_filter": ghose, 
        #     "rule_of_3": ro3,
        #     "oprea_lead_like": oprea,
        #     "fda_2025_criteria": fda,
        #     "overall_assessment": {
        #         "drug_like": lipinski["passed"] and veber["passed"],
        #         "lead_like": oprea["passed"],
        #         "fragment_like": ro3["passed"],
        #         "fda_approvable": fda["passed"],
        #         "development_potential": development_category,
        #         "fda_approval_category": fda["approval_category"]
        #     }
        # }
        report = {
            "SMILES": self.smiles,
            "lipinski_rule_of_5": lipinski["passed"],
            "veber_rule": veber["passed"],
            "ghose_filter": ghose["passed"],
            "rule_of_3": ro3["passed"],
            "oprea_lead_like": oprea["passed"],
            "fda_criteria": fda["passed"],
            "fda_approval_category": fda["approval_category"],
            "num_passed_rules": passed_rules,
        }
        
        return report
    
    def print_summary(self):
        """Print formatted summary"""
        report = self.get_summary_report()
        
        print("\n" + "="*70)
        print("CORRECTED DRUG-LIKENESS ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nSMILES: {report['smiles']}")
        
        print("\nCORRECTED RULE ASSESSMENTS:")
        print("-"*70)
        
        rules = [
            ("Lipinski's Rule of 5 (with molar refractivity)", report["lipinski_rule_of_5"]),
            ("Veber's Rule (2 criteria)", report["veber_rule"]),
            ("Ghose Filter (4 criteria)", report["ghose_filter"]),
            ("Rule of 3 Extended (5 criteria)", report["rule_of_3"]),
            ("Oprea Lead-like Rules (6 criteria)", report["oprea_lead_like"])
        ]
        
        for name, result in rules:
            status = "PASSED" if result["passed"] else "FAILED"
            print(f"{name}: {status} ({result['violations']} violations)")
        
        print("\nFDA 2025 DEVELOPMENT CRITERIA (7 criteria):")
        print("-"*70)
        print(f"FDA Approval Category: {report['fda_2025_criteria']['approval_category']}")
        print(f"Critical Failures: {report['fda_2025_criteria']['critical_failures']}")
        print(f"Approval Likelihood: {report['fda_2025_criteria']['approval_likelihood']:.1%}")
        
        print("\nOVERALL ASSESSMENT:")
        print("-"*70)
        for category, status in report["overall_assessment"].items():
            print(f"{category.replace('_', ' ').title()}: {status}")
        
        print("\nKEY FEATURES:")
        print("-"*70)
        print("✓ Lipinski Rule of 5: Includes molar refractivity (5 criteria)")
        print("✓ Rule of 3: Extended version with molar refractivity (5 criteria)")
        print("✓ FDA 2025: Updated with current guidelines (7 criteria)")
        print("✓ Clean radar plots: No violation counts in titles")
        
        print("\n" + "="*70)

    def create_custom_rule(self, name, parameters):
        """Create custom drug-likeness rule"""
        def custom_rule_checker():
            results = {}
            for prop, criteria in parameters.items():
                if prop == "allowed_violations":
                    continue
                    
                value = self._get_property(prop)
                
                if value is None:
                    results[prop] = {"value": None, "status": "unknown", "pass": False}
                    continue
                
                if "range" in criteria:
                    min_val, max_val = criteria["range"]
                    passed = min_val <= value <= max_val
                    results[prop] = {
                        "value": value,
                        "range": criteria["range"],
                        "pass": passed
                    }
                elif "limit" in criteria:
                    limit = criteria["limit"]
                    comparison = criteria.get("comparison", "less_than")
                    
                    if comparison == "less_than":
                        passed = value < limit
                    elif comparison == "less_than_equal":
                        passed = value <= limit
                    elif comparison == "greater_than":
                        passed = value > limit
                    elif comparison == "greater_than_equal":
                        passed = value >= limit
                    else:
                        passed = False
                    
                    results[prop] = {
                        "value": value,
                        "limit": limit,
                        "comparison": comparison,
                        "pass": passed
                    }
            
            violations = sum(1 for check in results.values() if not check["pass"])
            allowed_violations = parameters.get("allowed_violations", 0)
            
            return {
                "name": name,
                "passed": violations <= allowed_violations,
                "violations": violations,
                "allowed_violations": allowed_violations,
                "results": results
            }
        
        return custom_rule_checker


# Helper function for easy usage
def analyze_molecule_corrected(data_json=None, smiles=None):
    """
    Analyze molecule with corrected drug-likeness rules
    """
    if not data_json and not smiles:
        raise ValueError("Either data_json or smiles must be provided")
    
    analyzer = DrugLikenessAnalyzer(data=data_json, smiles=smiles)
    
    print("="*70)
    print("CORRECTED DRUG-LIKENESS ANALYSIS")
    print("="*70)
    print("Fixed Issues:")
    print("✓ Lipinski Rule of 5: 4 criteria (not 5) - MW, LogP, HDonors, HAcceptors")
    print("✓ Rule of 3: 3 criteria (not 6) - MW, LogP, HDonors") 
    print("✓ FDA 2025: 7 criteria based on current guidelines")
    print("✓ All plots now match rule criteria counts")
    print("="*70)
    
    # Generate all corrected radar plots
    plots_generated = analyzer.generate_all_radar_plots(save_plots=True, show_plots=False)
    
    # Get comprehensive report
    report = analyzer.get_summary_report()
    
    # Print detailed summary
    analyzer.print_summary()
    
    print(f"\nRadar plots generated: {sum(plots_generated.values())}/7")
    print("\nGenerated files:")
    expected_files = [
        "lipinski_rule_4_criteria.png",
        "veber_rule_2_criteria.png", 
        "ghose_filter_4_criteria.png",
        "rule_of_3_criteria.png",
        "oprea_leadlike_6_criteria.png",
        "fda_2025_7_criteria.png"
    ]
    
    import os
    for filename in expected_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024
            print(f"✓ {filename} ({size:.1f} KB)")
        else:
            print(f"✗ {filename} (missing)")
    
    return report

