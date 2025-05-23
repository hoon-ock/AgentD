import ast
import importlib.util
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Force the Agg backend before importing pyplot
import matplotlib.pyplot as plt

# cwd = os.getcwd()
# sys.path.append(cwd)
# home_dir = os.path.dirname(os.path.dirname(cwd))
# sys.path.append(home_dir)
from .drug_likeness_analyzer import DrugLikenessAnalyzer
# breakpoint()

def get_tool_decorated_functions(filepath):
    # Step 1: Parse the AST and find decorated functions
    with open(filepath, "r", encoding="utf-8") as f:
        node = ast.parse(f.read(), filename=filepath)

    decorated_func_names = []
    for n in ast.walk(node):
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                if isinstance(d, ast.Name) and d.id == "tool":
                    decorated_func_names.append(n.name)
                elif isinstance(d, ast.Call) and getattr(d.func, "id", "") == "tool":
                    decorated_func_names.append(n.name)

    # Step 2: Import the module dynamically
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Step 3: Extract function objects
    tools = [getattr(module, name) for name in decorated_func_names]

    return tools


def custom_serializer(obj):
    try:
        return str(obj)
    except:
        return f"<<Non-serializable: {type(obj).__name__}>>"
    

#!/usr/bin/env python
"""
Enhanced usage example with rule-based radar plots
"""



def rule_based_evaluation(data_string):
    """
    Example of using the DrugLikefnessAnalyzer class with DeepPK data
    """
    # Load DeepPK data (in this case from a string, but could be from a file)
    
    # Create analyzer with data
    analyzer = DrugLikenessAnalyzer(data=data_string)
    
    # Check various drug-likeness rules
    # lipinski_results = analyzer.check_lipinski_rule_of_5()
    # print(f"Lipinski's Rule of 5: {'Passed' if lipinski_results['passed'] else 'Failed'}")
    
    # veber_results = analyzer.check_veber_rule()
    # print(f"Veber's Rule: {'Passed' if veber_results['passed'] else 'Failed'}")
    
    # # Generate standard radar plots
    # print("\nGenerating standard radar plots...")
    # analyzer.generate_radar_plot(save_path="drug_likeness_radar.png", show=False)
    # analyzer.generate_fda_radar_plot(save_path="fda_assessment_radar.png", show=False)
    
    # # Generate rule-specific radar plots
    # print("\nGenerating rule-specific radar plots...")
    # analyzer.generate_rule_radar_plot('lipinski', save_path="lipinski_rule.png", show=False)
    # analyzer.generate_rule_radar_plot('veber', save_path="veber_rule.png", show=False)
    
    # Check more rules and generate their radar plots
    # print("\nChecking additional rules and generating plots...")
    # rule_of_3_results = analyzer.check_rule_of_3()
    # print(f"Rule of 3: {'Passed' if rule_of_3_results['passed'] else 'Failed'}")
    # analyzer.generate_rule_radar_plot('rule_of_3', save_path="rule_of_3.png", show=False)
    
    # pfizer_results = analyzer.check_pfizer_rule_of_4()
    # print(f"Pfizer's Rule of 4: {'Passed' if pfizer_results['passed'] else 'Failed'}")
    # analyzer.generate_rule_radar_plot('pfizer', save_path="pfizer_rule.png", show=False)
    
    # Print detailed summary
    report =  analyzer.get_summary_report() #.print_summary()
    
    return report
    
    # Example of creating a custom rule
    # custom_rule = analyzer.create_custom_rule(
    #     name="My Custom Rule",
    #     parameters={
    #         "logp": {"limit": 3, "comparison": "less_than"},
    #         "molecular_weight": {"limit": 450, "comparison": "less_than"},
    #         "h_donors": {"limit": 4, "comparison": "less_than_equal"},
    #         "allowed_violations": 1  # Allow one violation
    #     }
    # )
    
    # # Check custom rule
    # custom_results = custom_rule()
    # print(f"\nCustom Rule: {'Passed' if custom_results['passed'] else 'Failed'}")
    # return custom_results

