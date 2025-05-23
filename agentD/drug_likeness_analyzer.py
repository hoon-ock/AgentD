import matplotlib.pyplot as plt
import numpy as np
import json
import re
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
    Analyzes drug-likeness properties of molecules based on DeepPK data.
    Implements various filtering rules and generates radar plots for visualization.
    
    This class provides methods to:
    1. Evaluate molecules against common drug-likeness rules (Lipinski, Veber, Ghose, etc.)
    2. Generate radar plots for visualizing pharmacokinetic properties
    3. Create custom rules for filtering potential drug candidates
    4. Generate FDA-focused plots for drug development assessment
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
            # Parse JSON string
            try:
                self.data = json.loads(data.strip())
            except json.JSONDecodeError:
                # Try to clean up the JSON string if it's malformatted
                clean_data = re.sub(r'"\s*\n\s*"', '", "', data)
                try:
                    self.data = json.loads(clean_data)
                except json.JSONDecodeError:
                    # Last resort: try to manually fix the JSON
                    fixed_data = self._fix_json_string(data)
                    self.data = json.loads(fixed_data)
        else:
            # Assume it's already a dictionary
            self.data = data
        
        # Extract SMILES if available
        if "SMILES" in self.data:
            self.smiles = self.data["SMILES"]
            if RDKIT_AVAILABLE and self.smiles:
                self.mol = Chem.MolFromSmiles(self.smiles)
                if self.mol:
                    self._calculate_rdkit_properties()
        
#         # Flatten the data structure if needed
# `        if len(self.data.keys()) == 1 and "0" in self.data:
#             self.data = self.data["0"]`
    
    def _fix_json_string(self, json_str):
        
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()
        
        # Remove newlines and extra spaces
        json_str = re.sub(r'\s+', ' ', json_str)
        
        # Fix missing commas between key-value pairs
        json_str = re.sub(r'"\s+"', '", "', json_str)
        
        # Add missing quotes around keys
        json_str = re.sub(r'([{\s,])(\w+):', r'\1"\2":', json_str)
        
        # Ensure proper JSON structure
        if not json_str.startswith('{'):
            json_str = '{' + json_str
        if not json_str.endswith('}'):
            json_str = json_str + '}'
        
        return json_str
    
    def _calculate_rdkit_properties(self):
        """
        Calculate molecular properties using RDKit
        Requires RDKit to be installed
        """
        if not RDKIT_AVAILABLE or not self.mol:
            return
        
        # Calculate basic properties
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
        
        # Add a few more if available in the specific RDKit version
        try:
            self.calculated_properties["fraction_csp3"] = Descriptors.FractionCSP3(self.mol)
        except:
            pass
            
        try:
            self.calculated_properties["formal_charge"] = Chem.GetFormalCharge(self.mol)
        except:
            pass
    
    def _get_property(self, key, default=None):
        """
        Get a property value, prioritizing calculated properties over DeepPK data
        
        Parameters:
        -----------
        key : str
            Property name
        default : any
            Default value if property is not found
            
        Returns:
        --------
        value : any
            Property value
        """
        # First check if it's a calculated property
        if key in self.calculated_properties:
            return self.calculated_properties[key]
        
        # Then check DeepPK data
        if self.data:
            # Try standard DeepPK keys
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
        Check Lipinski's Rule of 5:
        - Molecular weight < 500 Da
        - LogP < 5
        - H-bond donors ≤ 5
        - H-bond acceptors ≤ 10
        
        Returns:
        --------
        dict
            Dictionary containing rule check results, violations count, and overall assessment
        """
        # Extract relevant properties
        mw = self._get_property("molecular_weight", 
                               float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp", 
                                float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        h_donors = self._get_property("h_donors", 3)  # Default if not available
        h_acceptors = self._get_property("h_acceptors", 4)  # Default if not available
        
        # Check rules
        results = {
            "molecular_weight": {"value": mw, "limit": 500, "pass": mw < 500},
            "logp": {"value": logp, "limit": 5, "pass": logp < 5},
            "h_donors": {"value": h_donors, "limit": 5, "pass": h_donors <= 5},
            "h_acceptors": {"value": h_acceptors, "limit": 10, "pass": h_acceptors <= 10}
        }
        
        violations = sum(1 for check in results.values() if not check["pass"])
        
        return {
            "passed": violations <= 1,  # Allow up to 1 violation
            "violations": violations,
            "results": results
        }
    
    def check_veber_rule(self):
        """
        Check Veber's Rule:
        - Rotatable bonds ≤ 10
        - Polar surface area (TPSA) ≤ 140 Å²
        
        Returns:
        --------
        dict
            Dictionary containing rule check results, violations count, and overall assessment
        """
        # Extract properties
        rotatable_bonds = self._get_property("rotatable_bonds", 5)  # Default if not available
        tpsa = self._get_property("tpsa", 90)  # Default if not available
        
        results = {
            "rotatable_bonds": {"value": rotatable_bonds, "limit": 10, "pass": rotatable_bonds <= 10},
            "tpsa": {"value": tpsa, "limit": 140, "pass": tpsa <= 140}
        }
        
        violations = sum(1 for check in results.values() if not check["pass"])
        
        return {
            "passed": violations == 0,  # No violations allowed
            "violations": violations,
            "results": results
        }
    
    def check_ghose_filter(self):
        """
        Check Ghose's Filter:
        - Molecular weight between 160-480 Da
        - LogP between -0.4 and 5.6
        - Molar refractivity between 40-130
        - Total number of atoms between 20-70
        
        Returns:
        --------
        dict
            Dictionary containing rule check results, violations count, and overall assessment
        """
        # Extract properties
        mw = self._get_property("molecular_weight", 
                               float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp", 
                                float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        molar_refractivity = self._get_property("molar_refractivity", 60)  # Default if not available
        atom_count = self._get_property("heavy_atoms", 30)  # Default if not available
        
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
        Check Rule of 3 (for fragment-based screening):
        - Molecular weight < 300 Da
        - LogP ≤ 3
        - H-bond donors ≤ 3
        - H-bond acceptors ≤ 3
        - Rotatable bonds ≤ 3
        - Polar surface area (TPSA) ≤ 60 Å²
        
        Returns:
        --------
        dict
            Dictionary containing rule check results, violations count, and overall assessment
        """
        # Extract properties
        mw = self._get_property("molecular_weight", 
                              float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp", 
                                float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        h_donors = self._get_property("h_donors", 2)  # Default if not available
        h_acceptors = self._get_property("h_acceptors", 3)  # Default if not available
        rotatable_bonds = self._get_property("rotatable_bonds", 2)  # Default if not available
        tpsa = self._get_property("tpsa", 40)  # Default if not available
        
        results = {
            "molecular_weight": {"value": mw, "limit": 300, "pass": mw < 300},
            "logp": {"value": logp, "limit": 3, "pass": logp <= 3},
            "h_donors": {"value": h_donors, "limit": 3, "pass": h_donors <= 3},
            "h_acceptors": {"value": h_acceptors, "limit": 3, "pass": h_acceptors <= 3},
            "rotatable_bonds": {"value": rotatable_bonds, "limit": 3, "pass": rotatable_bonds <= 3},
            "tpsa": {"value": tpsa, "limit": 60, "pass": tpsa <= 60}
        }
        
        violations = sum(1 for check in results.values() if not check["pass"])
        
        return {
            "passed": violations == 0,
            "violations": violations,
            "results": results
        }
    
    def check_pfizer_rule_of_4(self):
        """
        Check Pfizer's Rule of 4 (predicting CNS drug-likeness):
        - Molecular weight ≤ 400 Da
        - cLogP ≤ 4
        - H-bond donors ≤ 4
        - TPSA ≤ 90 Å²
        - pKa > 4
        
        Returns:
        --------
        dict
            Dictionary containing rule check results, violations count, and overall assessment
        """
        # Extract properties
        mw = self._get_property("molecular_weight",
                              float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp",
                                float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        h_donors = self._get_property("h_donors", 2)  # Default if not available
        tpsa = self._get_property("tpsa", 70)  # Default if not available
        pka = float(self.data.get("Predictions_general_properties_pka_acid", 6)) if self.data else 6
        
        results = {
            "molecular_weight": {"value": mw, "limit": 400, "pass": mw <= 400},
            "logp": {"value": logp, "limit": 4, "pass": logp <= 4},
            "h_donors": {"value": h_donors, "limit": 4, "pass": h_donors <= 4},
            "tpsa": {"value": tpsa, "limit": 90, "pass": tpsa <= 90},
            "pka": {"value": pka, "limit": 4, "pass": pka > 4}
        }
        
        violations = sum(1 for check in results.values() if not check["pass"])
        
        return {
            "passed": violations <= 1,  # Allow 1 violation
            "violations": violations,
            "results": results
        }
    
    def check_oprea_lead_like_rules(self):
        """
        Check Oprea's Lead-like Rules:
        - Molecular weight between 200-450 Da
        - logP between -1 and 4.5
        - Rotatable bonds ≤ 8
        - Rings ≤ 4
        - H-bond donors ≤ 5
        - H-bond acceptors ≤ 8
        
        Returns:
        --------
        dict
            Dictionary containing rule check results, violations count, and overall assessment
        """
        # Extract properties
        mw = self._get_property("molecular_weight",
                              float(self.data.get("Predictions_general_properties_bp", 0)) if self.data else 0)
        logp = self._get_property("logp",
                                float(self.data.get("Predictions_general_properties_log_p", 0)) if self.data else 0)
        rotatable_bonds = self._get_property("rotatable_bonds", 4)  # Default if not available
        rings = self._get_property("aromatic_rings", 2)  # Default if not available
        h_donors = self._get_property("h_donors", 2)  # Default if not available
        h_acceptors = self._get_property("h_acceptors", 3)  # Default if not available
        
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
            "passed": violations <= 1,  # Allow 1 violation
            "violations": violations,
            "results": results
        }
    
    def check_fda_criteria(self):
        """
        Check FDA-based criteria for drug development assessment
        
        Returns:
        --------
        dict
            Dictionary containing FDA criteria assessment, critical failures, and detailed results
        """
        # Extract absorption values
        f20 = float(self.data.get("Probability_absorption_f20", 0.5)) if self.data else 0.5
        hia = float(self.data.get("Probability_absorption_hia", 0.7)) if self.data else 0.7
        logs = float(self.data.get("Predictions_absorption_log_s", -3)) if self.data else -3
        
        # Extract distribution values
        bbb = float(self.data.get("Probability_distribution_bbb", 0.1)) if self.data else 0.1
        ppb = float(self.data.get("Predictions_distribution_ppb", 80)) if self.data else 80
        
        # Extract metabolism values
        cyp_inhibition = float(self.data.get("Predictions_metabolism_cyppro_iinh", 0.5)) if self.data else 0.5
        cyp3a4_substrate = float(self.data.get("Probability_metabolism_cyp3a4_substrate", 0.5)) if self.data else 0.5
        
        # Extract excretion values
        clearance = float(self.data.get("Predictions_excretion_clearance", 5)) if self.data else 5
        
        # Define criteria based on FDA guidelines
        results = {
            "bioavailability": {
                "value": f20, 
                "threshold": 0.5, 
                "pass": f20 > 0.5,
                "description": "Oral bioavailability (F) > 20%"
            },
            "intestinal_absorption": {
                "value": hia, 
                "threshold": 0.7, 
                "pass": hia > 0.7,
                "description": "Human intestinal absorption > 70%"
            },
            "solubility": {
                "value": logs, 
                "range": [-4, 0.5], 
                "pass": -4 <= logs <= 0.5,
                "description": "Log S between -4 and 0.5 (soluble)"
            },
            "blood_brain_barrier": {
                "value": bbb, 
                "threshold_cns": 0.5,
                "threshold_non_cns": 0.1,
                "pass_cns": bbb > 0.5,  # For CNS drugs
                "pass_non_cns": bbb < 0.1,  # For non-CNS drugs
                "description": "BBB penetration appropriate for indication"
            },
            "plasma_protein_binding": {
                "value": ppb, 
                "threshold": 90, 
                "pass": ppb < 90,
                "description": "Plasma protein binding < 90%"
            },
            "cyp_inhibition": {
                "value": cyp_inhibition, 
                "threshold": 0.5, 
                "pass": cyp_inhibition < 0.5,
                "description": "Low CYP inhibition potential"
            },
            "metabolic_stability": {
                "value": 1 - cyp3a4_substrate, 
                "threshold": 0.5, 
                "pass": (1 - cyp3a4_substrate) > 0.5,
                "description": "Good metabolic stability"
            },
            "clearance": {
                "value": clearance, 
                "range": [1, 10], 
                "pass": 1 <= clearance <= 10,
                "description": "Appropriate clearance rate"
            }
        }
        
        # Overall assessment depends on specific requirements
        critical_parameters = ["bioavailability", "intestinal_absorption", "solubility", "cyp_inhibition"]
        critical_failures = sum(1 for key in critical_parameters if key in results and not results[key]["pass"])
        
        # FDA approval likelihood estimation (simplified)
        total_criteria = len(results)
        passed_criteria = sum(1 for check in results.values() if "pass" in check and check["pass"])
        approval_likelihood = passed_criteria / total_criteria
        
        approval_category = "Low"
        if approval_likelihood > 0.8:
            approval_category = "High"
        elif approval_likelihood > 0.6:
            approval_category = "Moderate"
        
        return {
            "passed": critical_failures == 0,
            "critical_failures": critical_failures,
            "results": results,
            "approval_likelihood": approval_likelihood,
            "approval_category": approval_category
        }
    
    
    def _normalize(self, value, min_val, max_val):
        """
        Normalize a value to [0,1] range
        
        Parameters:
        -----------
        value : float
            Value to normalize
        min_val : float
            Minimum value in range
        max_val : float
            Maximum value in range
            
        Returns:
        --------
        float
            Normalized value between 0 and 1
        """
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(1, normalized))
    
    def _extract_radar_properties(self):
        """
        Extract and normalize properties for radar plot
        
        Returns:
        --------
        dict
            Dictionary of normalized properties for radar chart
        """
        # Define properties to display with their normalization ranges
        property_ranges = {
            "LogP": {
                "key": "Predictions_general_properties_log_p",
                "range": [0, 5],
                "invert": False
            },
            "LogS": {
                "key": "Predictions_absorption_log_s",
                "range": [-6, 0],
                "invert": False
            },
            "HIA": {
                "key": "Probability_absorption_hia",
                "range": [0, 1],
                "invert": False
            },
            "BBB": {
                "key": "Probability_distribution_bbb",
                "range": [0, 1],
                "invert": True  # Invert for non-CNS drugs
            },
            "PPB": {
                "key": "Predictions_distribution_ppb",
                "range": [0, 100],
                "invert": True  # Lower is better
            },
            "CYP1A2": {
                "key": "Probability_metabolism_cyp1a2_inhibitor",
                "range": [0, 1],
                "invert": True  # Lower inhibition is better
            },
            "CYP3A4": {
                "key": "Probability_metabolism_cyp3a4_substrate",
                "range": [0, 1],
                "invert": True  # Lower substrate activity is better
            },
            "Bioavailability": {
                "key": "Probability_absorption_f20",
                "range": [0, 1],
                "invert": False
            },
            "Clearance": {
                "key": "Predictions_excretion_clearance",
                "range": [0, 15],
                "invert": False  # Appropriate range is better
            }
        }
        
        # Extract and normalize values
        properties = {}
        for display_name, config in property_ranges.items():
            key = config["key"]
            if self.data and key in self.data:
                try:
                    value = float(self.data[key])
                    normalized = self._normalize(value, config["range"][0], config["range"][1])
                    if config.get("invert", False):
                        normalized = 1 - normalized
                    properties[display_name] = normalized
                except (ValueError, TypeError):
                    properties[display_name] = 0.5  # Default if conversion fails
            else:
                properties[display_name] = 0.5  # Default if not found
        
        return properties
    
    def generate_radar_plot(self, properties=None, save_path=None, show=True, title="Drug-likeness Radar Plot"):
        """
        Generate a radar plot visualizing key drug-likeness properties
        
        Parameters:
        -----------
        properties : dict, optional
            Dictionary of properties to plot (if None, uses default properties)
        save_path : str, optional
            Path to save the plot
        show : bool, default=True
            Whether to display the plot
        title : str, default="Drug-likeness Radar Plot"
            Title for the plot
            
        Returns:
        --------
        fig
            Matplotlib figure object
        """
        # Use provided properties or extract default ones
        if properties is None:
            properties = self._extract_radar_properties()
        
        # Number of variables
        categories = list(properties.keys())
        N = len(categories)
        
        # Values to plot
        values = list(properties.values())
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add values for the plot
        values += values[:1]  # Close the loop
        
        # Draw the plot
        ax.plot(angles, values, 'o-', linewidth=2, color='purple', alpha=0.8, label="Analyzed Compound")
        ax.fill(angles, values, color='purple', alpha=0.2)
        
        # Set y-limit
        ax.set_ylim(0, 1)
        
        # Draw axis lines for each angle and label
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw y-axis circles
        ax.set_yticklabels([])  # Remove default labels
        
        # Add custom y-axis circles and labels
        for i in [0.25, 0.5, 0.75]:
            ax.text(np.pi/4, i, f"{i:.2f}", ha='center', va='center', fontsize=10)
        
        # Move 0 to top (north)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(title, size=20, pad=40)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Show or close
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    def generate_fda_radar_plot(self, save_path=None, show=True):
        """
        Generate FDA-focused radar plot for drug approval criteria
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        fig
            Matplotlib figure object
        """
        # FDA-focused properties
        fda_properties = {
            "Oral Bioavailability": float(self.data.get("Probability_absorption_f20", 0.5)) if self.data else 0.5,
            "Intestinal Absorption": float(self.data.get("Probability_absorption_hia", 0.7)) if self.data else 0.7,
            "Solubility": self._normalize(
                float(self.data.get("Predictions_absorption_log_s", -3)) if self.data else -3, 
                -6, 0
            ),
            "Metabolism Stability": 1 - float(self.data.get("Probability_metabolism_cyp3a4_substrate", 0.5)) 
                                if self.data else 0.5,
            "Drug Interactions": 1 - float(self.data.get("Probability_metabolism_cyp1a2_inhibitor", 0.5)) 
                            if self.data else 0.5,
            "BBB Penetration": float(self.data.get("Probability_distribution_bbb", 0.1)) if self.data else 0.1,
            "Protein Binding": 1 - self._normalize(
                float(self.data.get("Predictions_distribution_ppb", 80)) if self.data else 80, 
                0, 100
            ),
            "Clearance": self._normalize(
                float(self.data.get("Predictions_excretion_clearance", 5)) if self.data else 5, 
                0, 15
            )
        }
        
        return self.generate_radar_plot(
            properties=fda_properties,
            save_path=save_path,
            show=show,
            title="FDA Drug Development Assessment"
        )

    def generate_lipinski_radar_plot(self, save_path=None, show=True):
        """Generate a radar plot for Lipinski's Rule of 5"""
        return self.generate_rule_radar_plot('lipinski', save_path, show)

    def generate_veber_radar_plot(self, save_path=None, show=True):
        """Generate a radar plot for Veber's Rule"""
        return self.generate_rule_radar_plot('veber', save_path, show)

    def generate_ghose_radar_plot(self, save_path=None, show=True):
        """Generate a radar plot for Ghose Filter"""
        return self.generate_rule_radar_plot('ghose', save_path, show)

    def generate_rule_of_3_radar_plot(self, save_path=None, show=True):
        """Generate a radar plot for Rule of 3"""
        return self.generate_rule_radar_plot('rule_of_3', save_path, show)

    def generate_pfizer_rule_of_4_radar_plot(self, save_path=None, show=True):
        """Generate a radar plot for Pfizer's Rule of 4"""
        return self.generate_rule_radar_plot('pfizer', save_path, show)

    def generate_oprea_radar_plot(self, save_path=None, show=True):
        """Generate a radar plot for Oprea's Lead-like Rules"""
        return self.generate_rule_radar_plot('oprea', save_path, show)
    
    def generate_rule_radar_plot(self, rule_name, save_path=None, show=True):
        """
        Generate a radar plot for a specific drug-likeness rule
        
        Parameters:
        -----------
        rule_name : str
            Name of the rule to visualize ('lipinski', 'veber', 'ghose', 'rule_of_3', 'pfizer', 'oprea')
        save_path : str, optional
            Path to save the plot
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        fig
            Matplotlib figure object
        """
        try:
            # Get rule check results
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
                title = "Rule of 3"
            elif rule_name.lower() in ['pfizer', 'rule_of_4', 'ro4']:
                rule_results = self.check_pfizer_rule_of_4()
                title = "Pfizer's Rule of 4"
            elif rule_name.lower() == 'oprea':
                rule_results = self.check_oprea_lead_like_rules()
                title = "Oprea's Lead-like Rules"
            else:
                raise ValueError(f"Unknown rule: {rule_name}")
            
            # Extract property values and limits from rule results
            properties = {}
            normalized_values = {}
            ideal_values = []
            
            # Create nice display names for properties
            display_names = {
                'molecular_weight': 'Mol Weight',
                'logp': 'LogP',
                'h_donors': 'H-Donors',
                'h_acceptors': 'H-Acceptors',
                'rotatable_bonds': 'Rot Bonds',
                'tpsa': 'TPSA',
                'molar_refractivity': 'Molar Refr',
                'atom_count': 'Atoms',
                'rings': 'Rings',
                'pka': 'pKa'
            }
            
            # Process each property in the rule
            for prop_name, prop_data in rule_results['results'].items():
                # Get actual value
                value = prop_data['value']
                display_name = display_names.get(prop_name, prop_name.replace('_', ' ').title())
                
                # Normalize value based on rule type
                if 'limit' in prop_data:
                    if prop_data.get('comparison', 'less_than') in ['less_than', 'less_than_equal']:
                        # Upper limit (lower is better)
                        min_val = 0
                        max_val = prop_data['limit'] * 1.5  # Extend a bit beyond limit for display
                        ideal = prop_data['limit'] * 0.7  # Ideal is 70% of limit
                        normalized = 1 - (value / max_val)  # Invert so lower values = higher on plot
                    else:
                        # Lower limit (higher is better)
                        min_val = prop_data['limit'] * 0.5  # Half of limit
                        max_val = prop_data['limit'] * 1.5  # 1.5x limit
                        ideal = prop_data['limit'] * 1.2  # Ideal is 120% of limit
                        normalized = (value - min_val) / (max_val - min_val)
                elif 'range' in prop_data:
                    # Range with min and max
                    min_val, max_val = prop_data['range']
                    range_mid = (min_val + max_val) / 2
                    ideal = range_mid  # Ideal is middle of range
                    
                    # Normalize to 0-1 with ideal value at 0.8
                    if value < min_val:
                        normalized = 0.4 * (value / min_val)  # Below range, scale from 0 to 0.4
                    elif value > max_val:
                        normalized = 0.8 + 0.2 * ((value - max_val) / max_val)  # Above range
                    else:
                        # Within range, scale from 0.4 to 0.8
                        normalized = 0.4 + 0.4 * ((value - min_val) / (max_val - min_val))
                else:
                    # Fallback normalization
                    normalized = 0.5
                    ideal = 0.5
                
                # Clamp to 0-1 range
                normalized = max(0, min(1, normalized))
                
                # Add to properties and normalized values
                properties[display_name] = value
                normalized_values[display_name] = normalized
                ideal_values.append(0.7)  # Use 0.7 as ideal value on the radar plot
            
            # Create plot data
            categories = list(normalized_values.keys())
            values = list(normalized_values.values())
            
            # Number of variables
            N = len(categories)
            
            # Create figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add values for the plot
            values += values[:1]  # Close the loop
            ideal_values += ideal_values[:1]  # Close the loop
            
            # Draw the plot
            ax.plot(angles, values, 'o-', linewidth=2, color='purple', alpha=0.8, label="Compound Values")
            ax.fill(angles, values, color='purple', alpha=0.2)
            
            # Draw ideal zone
            ax.plot(angles, ideal_values, '--', linewidth=1, color='green', alpha=0.6, label="Ideal Zone")
            
            # Set y-limit
            ax.set_ylim(0, 1)
            
            # Draw axis lines for each angle and label
            plt.xticks(angles[:-1], categories, size=12)
            
            # Draw y-axis circles
            ax.set_yticklabels([])  # Remove default labels
            
            # Move 0 to top (north)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Add violation count to title
            title += f" ({rule_results['violations']} violations)"
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Add title
            plt.title(title, size=20, pad=40)
            
            # Add passed/failed indicator
            passed_text = "PASSED" if rule_results['passed'] else "FAILED"
            ax.text(0, 0, passed_text, ha='center', va='center', fontsize=24, 
                    color='green' if rule_results['passed'] else 'red')
            
            # Save with explicit settings if requested
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            # Show or close
            if show:
                plt.show()
            else:
                plt.close()
            
            return fig
            
        except Exception as e:
            print(f"Error generating rule radar plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    # Add this function to your drug_likeness_analyzer.py file
    # Add this function to your drug_likeness_analyzer.py file

    def get_summary_report(self):
        """
        Generate a comprehensive summary report of all drug-likeness rules
        
        Returns:
        --------
        dict
            Dictionary containing rule check results and overall assessment
        """
        lipinski = self.check_lipinski_rule_of_5()
        veber = self.check_veber_rule()
        ghose = self.check_ghose_filter()
        ro3 = self.check_rule_of_3()
        pfizer = self.check_pfizer_rule_of_4()
        oprea = self.check_oprea_lead_like_rules()
        fda = self.check_fda_criteria()
        
        # Generate an overall drug development category
        passed_rules = sum([
            lipinski["passed"],
            veber["passed"],
            ghose["passed"],
            pfizer["passed"],
            oprea["passed"],
            fda["passed"]
        ])
        
        development_category = "Unfavorable"
        if passed_rules >= 5:
            development_category = "Excellent"
        elif passed_rules >= 4:
            development_category = "Good"
        elif passed_rules >= 3:
            development_category = "Moderate"
        elif passed_rules >= 2:
            development_category = "Poor"
            
        # report = {
        #     "smiles": self.smiles,
        #     "lipinski_rule_of_5": lipinski,
        #     "veber_rule": veber,
        #     "ghose_filter": ghose, 
        #     "rule_of_3": ro3,
        #     "pfizer_rule_of_4": pfizer,
        #     "oprea_lead_like": oprea,
        #     "fda_criteria": fda,
        #     "overall_assessment": {
        #         "drug_like": lipinski["passed"] and veber["passed"],
        #         "lead_like": oprea["passed"],
        #         "fragment_like": ro3["passed"],
        #         "cns_drug_like": pfizer["passed"],
        #         "development_potential": development_category,
        #         "fda_approval_category": fda["approval_category"]
        #     }
        # }

        report = {
            "smiles": self.smiles,
            "drug_like": lipinski["passed"] and veber["passed"],
            "lead_like": oprea["passed"],
            "fragment_like": ro3["passed"],
            "cns_drug_like": pfizer["passed"],
            "development_potential": development_category,
            "fda_approval_category": fda["approval_category"]
            
        }


        report = {
            "smiles": self.smiles,
            "lipinski_rule_of_5": lipinski["passed"],
            "veber_rule": veber["passed"],
            "ghose_filter": ghose["passed"],
            "rule_of_3": ro3["passed"],
            "pfizer_rule_of_4": pfizer["passed"],
            "oprea_lead_like": oprea["passed"],
            "fda_criteria": fda["passed"],
            "fda_approval_category": fda["approval_category"]
        }
        
        return report
    
    def print_summary(self):
        """
        Print a formatted summary of drug-likeness results
        """
        report = self.get_summary_report()
        
        print("\n" + "="*50)
        print("DRUG-LIKENESS ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"\nSMILES: {report['smiles']}")
        
        print("\nRULE ASSESSMENTS:")
        print("-"*50)
        rules = [
            ("Lipinski's Rule of 5", report["lipinski_rule_of_5"]),
            ("Veber's Rule", report["veber_rule"]),
            ("Ghose Filter", report["ghose_filter"]),
            ("Rule of 3 (Fragment)", report["rule_of_3"]),
            ("Pfizer Rule of 4 (CNS)", report["pfizer_rule_of_4"]),
            ("Oprea Lead-like Rules", report["oprea_lead_like"])
        ]
        
        for name, result in rules:
            status = "PASSED" if result["passed"] else "FAILED"
            print(f"{name}: {status} ({result['violations']} violations)")
        
        print("\nFDA DEVELOPMENT CRITERIA:")
        print("-"*50)
        print(f"FDA Approval Category: {report['fda_criteria']['approval_category']}")
        print(f"Critical Failures: {report['fda_criteria']['critical_failures']}")
        
        print("\nOVERALL ASSESSMENT:")
        print("-"*50)
        for category, status in report["overall_assessment"].items():
            print(f"{category.replace('_', ' ').title()}: {status}")
        
        print("\n" + "="*50)