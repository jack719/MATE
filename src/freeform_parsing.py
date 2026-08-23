# import re
# import sys
# import pandas as pd
# import torch
# from datetime import datetime
# from typing import List, Dict, Optional

# from src.logging import Logger

# # 全局变量
# all_data = []
# exceptions = []
# SYSTEM_PROMPT=""

# # ==========================================
# # 第一阶段:样品抽取 PROMPT (unchanged)
# # ==========================================
# STAGE1_SYSTEM_PROMPT = '''
# You are an expert Materials Science Information Extraction engine based on the GoLLIE (Guideline-following) architecture.

# ## STAGE 1: SAMPLE IDENTIFICATION

# Your task is to identify all distinct **samples** mentioned in the paper and instantiate the following Python class:

# # ===== Entity Types =====
# class TitaniumAlloy:
#     """Represents a titanium alloy sample with its chemical formula/composition.
    
#     GUIDELINES for STAGE 1:
#     - Each sample is a distinct experimental specimen
#     - Samples may differ by: processing conditions, heat treatment, composition variations
#     - Extract explicit labels from text (e.g., "Sample A", "S1", "As-received", "Heat-treated")
#     - If no explicit label exists, create descriptive labels (e.g., "sample_1", "sample_annealed")
#     """
#     def __init__(self, name: str, formula: str, label: str, description: str = None):
#         """
#          Args:
#              name: Common name of the titanium alloy (e.g., "Ti-5Al-2.5Sn", "Ti-10V-2Fe-3Al")
#              formula: Chemical formula representation (e.g., "Ti-5Al-2.5Sn" or "Ti-10V-2Fe-3Al")
#              label: Unique identifier for this sample (e.g., "S1", "Sample-A", "as-cast", "annealed")
#              description: Brief description of sample condition (e.g., "annealed at 850C for 3h", "as-received")
#         """
#         self.name = name
#         self.formula = formula
#         self.label = label
#         self.description = description


# ### GLOBAL RULES
# # 1. **NO Class Definitions**: Class is already defined. DO NOT redefine it.
# # 2. **Keyword Arguments ONLY**: Use keyword arguments explicitly.
# # 3. **Unique Variable Names**: Assign every object to a distinct variable (e.g., `alloy_1`, `alloy_2`).
# # 4. **Format**: Output valid Python code wrapped in ```python and ``` markers.
# # 5. **One sample = One TitaniumAlloy instance**: Each distinct sample gets its own alloy object.
# # 6. **CRITICAL**: If the paper contains NO titanium alloy samples, or if the input is empty/invalid, output ONLY: ```python\n# NULL\n```

# ### ONE-SHOT EXAMPLE(YOU SHOULD NOT ADD THIS EXAMPLE INTO RESULT!):

# Input Text:
# """
# We investigated Ti-5Al-2.5Sn alloy under two different processing routes:
# - Specimen X: as-fabricated condition
# - Specimen Y: solution treated at 850°C for 3 hours

# Table 2 presents mechanical test data for both specimens.
# """

# Output Code:
# ```python
# # Define Specimen X
# alloy_1 = TitaniumAlloy(
#     name="Ti-5Al-2.5Sn", 
#     formula="Ti-5Al-2.5Sn", 
#     label="Specimen X", 
#     description="as-fabricated condition"
# )

# # Define Specimen Y
# alloy_2 = TitaniumAlloy(
#     name="Ti-5Al-2.5Sn", 
#     formula="Ti-5Al-2.5Sn", 
#     label="Specimen Y", 
#     description="solution treated at 850°C for 3 hours"
# )
# ```

# ### EXAMPLE: No Samples Found

# Input Text:
# """
# This paper reviews computational methods for predicting crystal structures.
# """

# Output Code:
# ```python
# # NULL
# ```

# ### YOUR TASK
# Identify all samples in the user's input paper and generate Python code wrapped in ```python and ```.
# If no samples exist, output: ```python\n# NULL\n```
# '''


# # ==========================================
# # 第二阶段:样品属性抽取 PROMPT (UPDATED)
# # ==========================================
# STAGE2_SYSTEM_PROMPT = '''
# You are an expert Materials Science Information Extraction engine based on the GoLLIE (Guideline-following) architecture.

# ## STAGE 2: SAMPLE PROPERTIES EXTRACTION

# You will receive:
# 1. The original paper text
# 2. A list of identified samples from Stage 1 with their labels

# Your task is to extract properties (element composition, heat treatment, mechanical properties) for each sample.

# **CRITICAL OUTPUT RULES**: 
# 1. Output ONLY valid Python code wrapped in ```python and ``` markers.
# 2. NO explanations. NO markdown headers. NO conversational filler. 
# 3. Start directly with class instantiations inside the code block.
# 4. If Stage 1 returned NULL (no samples found), output: ```python\n# NULL\n```
# 5. If you cannot find any properties for the samples, output: ```python\n# NULL\n```

# # ===== Entity Types =====
# class TitaniumAlloy:
#     """Represents a titanium alloy sample (already identified in Stage 1)."""
#     def __init__(self, name: str, formula: str, label: str, description: str = None):
#         self.name = name
#         self.formula = formula
#         self.label = label
#         self.description = description
#         self.elements = []
#         self.properties = []
#         self.treatment_steps = []


# class ElementComposition:
#     """Represents a single element in the alloy composition.
    
#     GUIDELINES:
#     - Extract measured values if available; otherwise use nominal values.
#     - Associate with the correct sample label
#     """
#     def __init__(self, 
#                  element: str,
#                  content: float = None,
#                  unit: str = "wt%"):
#         self.element = element
#         self.content = content
#         self.unit = unit


# class MechanicalProperty:
#     """Represents mechanical properties of a titanium alloy."""
#     def __init__(self,
#                  UTS_MPa: float = None,
#                  elongation_percent: float = None,
#                  youngs_modulus_GPa: float = None,
#                  vickers_hardness_HV: float = None):
#         """
#         CRITICAL EXTRACTION GUIDELINES:
#         1. **Scope**: Extract the following properties:
#            - Ultimate Tensile Strength (UTS) from Tensile Tests
#            - Elongation from Tensile Tests
#            - Young's Modulus (E) from Tensile Tests or other mechanical testing
#            - Vickers Hardness (HV) from Hardness Tests
           
#         2. **Exclusions**: 
#            - DO NOT extract Yield Strength (YS).
#            - DO NOT extract values from Compression tests or Shear tests for UTS/elongation.
#            - DO NOT extract 'Reduction of Area' (RA) for elongation.
#            - DO NOT extract other hardness scales (Rockwell, Brinell) for Vickers hardness.
           
#         3. **Data Source**: 
#            - Extract ONLY data belonging to "This Work" (experimental results of the authors).
#            - IGNORE values cited from references or literature reviews.
#            - IGNORE values derived from Simulations (FEM/DFT).
        
#         4. **Numeric Handling (CRITICAL)**:
#            - **Ranges**: If a range is provided (e.g., "950-1000 MPa"), calculate and extract the **AVERAGE** (975). Do NOT return a string.
#            - **Statistics**: If "Mean ± SD" is provided (e.g., "960 ± 15 MPa"), extract the **MEAN** (960). Ignore the standard deviation.
#            - **Qualifiers**: Remove symbols like "approx.", "~", ">", "<". Extract only the numeric value.
           
#         5. **Units**:
#            - UTS_MPa: Always convert to MPa
#            - elongation_percent: Always express as percentage (%)
#            - youngs_modulus_GPa: Always convert to GPa (if given in MPa, divide by 1000)
#            - vickers_hardness_HV: Extract the HV value directly (no unit conversion needed)
#         """
#         self.UTS_MPa = UTS_MPa
#         self.elongation_percent = elongation_percent
#         self.youngs_modulus_GPa = youngs_modulus_GPa
#         self.vickers_hardness_HV = vickers_hardness_HV


# class HeatTreatmentStep:
#     """Represents a single heat treatment step.
    
#     GUIDELINES:
#     1. **Sequence**: Maintain the chronological order of steps.
#     2. **Units**: 
#        - 'temperature_unit' must be explicitly extracted as 'C' (Celsius) or 'K' (Kelvin).
#        - 'time_unit' must be explicitly extracted (e.g., 'h', 'min', 's'). Do NOT convert units yourself.
#     3. **Cooling**: Distinguish between 'air_cool', 'furnace_cool', 'water_quench'. If unknown, use 'unspecified'.
#     """
#     def __init__(self,
#                  step_type: str,
#                  temperature: float = None,
#                  temperature_unit: str = None,
#                  time_value: float = None,
#                  time_unit: str = None,
#                  cooling_method: str = None,
#                  cooling_medium: str = None,
#                  cooling_temperature: float = None,
#                  cooling_temperature_unit: str = None,
#                  atmosphere: str = None,
#                  other_parameters: str = None):
#         self.step_type = step_type
#         self.temperature = temperature
#         self.temperature_unit = temperature_unit
#         self.time_value = time_value
#         self.time_unit = time_unit
#         self.cooling_method = cooling_method
#         self.cooling_medium = cooling_medium
#         self.cooling_temperature = cooling_temperature
#         self.cooling_temperature_unit = cooling_temperature_unit
#         self.atmosphere = atmosphere
#         self.other_parameters = other_parameters
#         self.next_step = None


# # ===== Relations =====
# def has_element(alloy: TitaniumAlloy, element: ElementComposition):
#     """Link a titanium alloy with one of its elemental compositions."""
#     alloy.elements.append(element)


# def has_property(alloy: TitaniumAlloy, prop: MechanicalProperty):
#     """Link a titanium alloy with its mechanical properties."""
#     alloy.properties.append(prop)


# def has_treatment_step(alloy: TitaniumAlloy, step: HeatTreatmentStep):
#     """Add one heat treatment step to the alloy's processing route."""
#     alloy.treatment_steps.append(step)


# def next_step(previous_step: HeatTreatmentStep, next_step_obj: HeatTreatmentStep):
#     """Define the processing order: next_step happens after previous_step."""
#     previous_step.next_step = next_step_obj


# ### GLOBAL RULES
# # 1. **NO Class Definitions**: Classes are already defined. DO NOT redefine them.
# # 2. **Keyword Arguments ONLY**: Use keyword arguments explicitly.
# # 3. **Unique Variable Names**: Assign every object to a distinct variable.
# # 4. **Relations as Functions**: Use provided relation functions to link entities.
# # 5. **Format**: Output valid Python code wrapped in ```python and ``` markers.
# # 6. **NEVER** merge properties from different samples into a single object.
# # 7. **Match Sample Labels**: Use the EXACT sample labels provided from Stage 1.
# # 8. **If Stage 1 is NULL or no properties found**: Output ```python\n# NULL\n```

# ### ONE-SHOT EXAMPLE(YOU SHOULD NOT ADD THIS EXAMPLE INTO RESULT!):

# Stage 1 Output (Samples identified):
# """
# ```python
# alloy_1 = TitaniumAlloy(name="Ti-5Al-2.5Sn", formula="Ti-5Al-2.5Sn", label="Specimen X", description="as-fabricated condition")
# alloy_2 = TitaniumAlloy(name="Ti-5Al-2.5Sn", formula="Ti-5Al-2.5Sn", label="Specimen Y", description="solution treated at 850°C for 3 hours")
# ```
# """

# Input Text:
# """
# Table 1: Chemical composition of Ti-5Al-2.5Sn (wt%).
# | Element | Ti   | Al  | Sn  |
# |---------|------|-----|-----|
# | Content | 92.5 | 5.0 | 2.5 |

# Table 2: Mechanical properties
# | Sample | UTS (MPa) | Elongation (%) | Young's Modulus (GPa) | Vickers Hardness (HV) |
# |--------|-----------|----------------|----------------------|----------------------|
# | Specimen X | 880 | 10 | 110 | 320 |
# | Specimen Y | 920 | 13 | 115 | 340 |

# Specimen X was tested in as-fabricated condition.
# Specimen Y was solution treated at 850°C for 3 hours and air cooled.
# """

# Output Code:
# ```python
# # Recreate Specimen X (from Stage 1)
# alloy_1 = TitaniumAlloy(
#     name="Ti-5Al-2.5Sn", 
#     formula="Ti-5Al-2.5Sn", 
#     label="Specimen X", 
#     description="as-fabricated condition"
# )

# # Define Elements for Specimen X
# elem_1_1 = ElementComposition(element="Ti", content=92.5, unit="wt%")
# elem_1_2 = ElementComposition(element="Al", content=5.0, unit="wt%")
# elem_1_3 = ElementComposition(element="Sn", content=2.5, unit="wt%")
# has_element(alloy=alloy_1, element=elem_1_1)
# has_element(alloy=alloy_1, element=elem_1_2)
# has_element(alloy=alloy_1, element=elem_1_3)

# # Define Properties for Specimen X
# prop_1 = MechanicalProperty(
#     UTS_MPa=880.0, 
#     elongation_percent=10.0,
#     youngs_modulus_GPa=110.0,
#     vickers_hardness_HV=320.0
# )
# has_property(alloy=alloy_1, prop=prop_1)

# # Recreate Specimen Y (from Stage 1)
# alloy_2 = TitaniumAlloy(
#     name="Ti-5Al-2.5Sn", 
#     formula="Ti-5Al-2.5Sn", 
#     label="Specimen Y", 
#     description="solution treated at 850°C for 3 hours"
# )

# # Define Elements for Specimen Y
# elem_2_1 = ElementComposition(element="Ti", content=92.5, unit="wt%")
# elem_2_2 = ElementComposition(element="Al", content=5.0, unit="wt%")
# elem_2_3 = ElementComposition(element="Sn", content=2.5, unit="wt%")
# has_element(alloy=alloy_2, element=elem_2_1)
# has_element(alloy=alloy_2, element=elem_2_2)
# has_element(alloy=alloy_2, element=elem_2_3)

# # Define Heat Treatment for Specimen Y
# step_2 = HeatTreatmentStep(
#     step_type="solution_treatment",
#     temperature=850.0,
#     temperature_unit="C",
#     time_value=3.0,
#     time_unit="h",
#     cooling_method="air_cool"
# )
# has_treatment_step(alloy=alloy_2, step=step_2)

# # Define Properties for Specimen Y
# prop_2 = MechanicalProperty(
#     UTS_MPa=920.0, 
#     elongation_percent=13.0,
#     youngs_modulus_GPa=115.0,
#     vickers_hardness_HV=340.0
# )
# has_property(alloy=alloy_2, prop=prop_2)
# ```

# ### EXAMPLE: Stage 1 returned NULL

# Stage 1 Output:
# """
# ```python
# # NULL
# ```
# """

# Output Code:
# ```python
# # NULL
# ```

# ### YOUR TASK
# Extract properties for each sample identified in Stage 1. Generate Python code wrapped in ```python and ```.
# If Stage 1 is NULL or no properties found, output: ```python\n# NULL\n```
# '''


# # ==========================================
# # Pipeline Functions (unchanged)
# # ==========================================

# def extract_code_from_markdown(text: str, logger: Logger) -> Optional[str]:
#     """Extract code from markdown code blocks."""
#     if not text or len(text.strip()) == 0:
#         logger.log("⚠️  extract_code_from_markdown: Empty input")
#         return None
    
#     if "# NULL" in text:
#         logger.log("🔍 Found NULL marker in generated code")
#         return None
    
#     pattern = r"```python\s*(.*?)\s*```"
#     match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
#     if match:
#         code = match.group(1).strip()
#         logger.log(f"✅ Extracted code from ```python block ({len(code)} chars)")
        
#         if code.strip() == "# NULL" or code.strip() == "#NULL":
#             logger.log("🔍 Code block contains only NULL marker")
#             return None
        
#         return code
    
#     pattern_generic = r"```\s*(.*?)\s*```"
#     match_generic = re.search(pattern_generic, text, re.DOTALL)
    
#     if match_generic:
#         code = match_generic.group(1).strip()
#         logger.log(f"✅ Extracted code from generic ``` block ({len(code)} chars)")
        
#         if code.strip() == "# NULL" or code.strip() == "#NULL":
#             logger.log("🔍 Code block contains only NULL marker")
#             return None
        
#         return code
    
#     logger.log("⚠️  No code blocks found, using raw text as fallback")
#     return text.strip()


# def parse_paper_stage1(model, tokenizer, paper_text: str, logger: Logger) -> Optional[str]:
#     """Stage 1: Extract samples"""
    
#     logger.log("=" * 80)
#     logger.log("🔍 DEBUG STAGE 1 - INPUT CHECK")
#     logger.log("=" * 80)
    
#     if not paper_text or len(paper_text.strip()) == 0:
#         logger.log("❌ ERROR: paper_text is EMPTY in parse_paper_stage1!")
#         return None
    
#     logger.log(f"✅ Stage 1 - Paper text received")
#     logger.log(f"   Length: {len(paper_text)} characters")
#     logger.log(f"   First 100 chars: '{paper_text[:100]}'")
#     logger.log(f"   Last 100 chars: '{paper_text[-100:]}'")
#     logger.log("=" * 80)
    
#     messages = [
#         {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
#         {"role": "user", "content": paper_text}
#     ]
    
#     raw_output = _generate_code(model, tokenizer, messages, logger, stage="STAGE1")
    
#     if raw_output is None:
#         return None
    
#     extracted_code = extract_code_from_markdown(raw_output, logger)
    
#     return extracted_code


# def parse_paper_stage2(model, tokenizer, paper_text: str, stage1_samples: str, logger: Logger) -> Optional[str]:
#     """Stage 2: Extract properties for identified samples"""
    
#     logger.log("=" * 80)
#     logger.log("🔍 DEBUG STAGE 2 - INPUT CHECK")
#     logger.log("=" * 80)
    
#     if stage1_samples is None:
#         logger.log("⚠️  Stage 1 returned NULL, skipping Stage 2")
#         return None
    
#     if not paper_text or len(paper_text.strip()) == 0:
#         logger.log("❌ ERROR: paper_text is EMPTY in parse_paper_stage2!")
#         return None
#     else:
#         logger.log(f"✅ Stage 2 - Paper text received")
#         logger.log(f"   Length: {len(paper_text)} characters")
#         logger.log(f"   First 100 chars: '{paper_text[:100]}'")
    
#     if len(stage1_samples.strip()) == 0:
#         logger.log("❌ ERROR: stage1_samples is EMPTY in parse_paper_stage2!")
#         return None
#     else:
#         logger.log(f"✅ Stage 2 - Stage 1 samples received")
#         logger.log(f"   Length: {len(stage1_samples)} characters")
#         logger.log(f"   First 100 chars: '{stage1_samples[:100]}'")
    
#     logger.log("=" * 80)
    
#     combined_input = f"""## Stage 1 Identified Samples:
# ```python
# {stage1_samples}
# ```

# ## Original Paper Text:
# {paper_text}
# """
    
#     logger.log(f"📦 Combined input for Stage 2:")
#     logger.log(f"   Total length: {len(combined_input)} characters")
#     logger.log(f"   First 200 chars: '{combined_input[:200]}'")
#     logger.log("=" * 80)
    
#     messages = [
#         {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
#         {"role": "user", "content": combined_input}
#     ]
    
#     raw_output = _generate_code(model, tokenizer, messages, logger, stage="STAGE2")
    
#     if raw_output is None:
#         return None
    
#     extracted_code = extract_code_from_markdown(raw_output, logger)
    
#     return extracted_code


# def _generate_code(model, tokenizer, messages: List[Dict[str, str]], logger: Logger, stage: str = "UNKNOWN") -> Optional[str]:
#     """Common code generation logic"""
#     assert messages[0]["role"] == "system", "First prompt must be system prompt"
#     assert messages[1]["role"] == "user", "Second prompt must be a user prompt"
    
#     logger.log(f"🚀 [{stage}] Starting code generation...")
#     logger.log(f"   System prompt length: {len(messages[0]['content'])} chars")
#     logger.log(f"   User content length: {len(messages[1]['content'])} chars")
    
#     prompt = ""
#     if hasattr(tokenizer, "apply_chat_template"):
#         prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#     else:
#         role_start = "<|im_start|>"
#         role_end = "<|im_end|>"
#         for m in messages:
#             prompt += f"{role_start}{m['role']}\n{m['content']}\n{role_end}\n"
#         prompt += f"{role_start}assistant\n"

#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#     device = next(model.parameters()).device
#     input_ids = input_ids.to(device)
    
#     logger.log(f"   Input tokens shape: {input_ids.shape}")
#     logger.log(f"   Total prompt length: {len(prompt)} chars")

#     with torch.no_grad():
#         output_ids = model.generate(
#             input_ids,
#             max_new_tokens=4096,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     generated_ids = output_ids[0][input_ids.shape[-1]:]
#     code_str = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
#     logger.log(f"✅ [{stage}] Generation complete")
#     logger.log(f"   Generated code length: {len(code_str)} chars")
#     logger.log(f"   First 200 chars: '{code_str[:200]}'")
    
#     return code_str


# def save_generated_code(code_str: str, paper_name: str, stage: str, logger: Logger, output_dir: str = "./codekgc_output"):
#     """Save the generated Python code to a file."""
#     import os
    
#     try:
#         os.makedirs(output_dir, exist_ok=True)
        
#         now = datetime.now()
#         timestamp = now.strftime("%Y%m%d_%H%M%S")
        
#         safe_paper_name = re.sub(r'[<>:"/\\|?*]', '_', paper_name)
        
#         filename = f"{safe_paper_name}_{stage}_{timestamp}.py"
#         filepath = os.path.join(output_dir, filename)
        
#         with open(filepath, 'w', encoding='utf-8') as f:
#             f.write(code_str)
        
#         logger.log(f"💾 Generated code saved to: {filepath}")
#         return filepath
    
#     except Exception as e:
#         logger.log(f"❌ Error saving generated code: {str(e)}")
#         exceptions.append({
#             'error': str(e),
#             'type': 'save_code_error',
#             'paper': paper_name,
#             'stage': stage,
#             'timestamp': datetime.now().isoformat()
#         })
#         return None


# def merge_stage_codes(stage1_code: str, stage2_code: str, logger: Logger) -> Optional[str]:
#     """Merge Stage 1 and Stage 2 codes into a single file."""
    
#     if stage1_code is None:
#         logger.log("⚠️  Stage 1 is NULL, cannot merge")
#         return None
    
#     if stage2_code is None:
#         logger.log("⚠️  Stage 2 is NULL, returning Stage 1 only")
#         merged_code = "# ==========================================\n"
#         merged_code += "# Stage 1: Sample Identification\n"
#         merged_code += "# ==========================================\n\n"
#         merged_code += stage1_code.strip()
#         merged_code += "\n\n# Note: Stage 2 returned NULL (no properties found)\n"
#         return merged_code
    
#     logger.log(f"🔗 Merging Stage 1 ({len(stage1_code)} chars) and Stage 2 ({len(stage2_code)} chars)")
    
#     merged_code = "# ==========================================\n"
#     merged_code += "# Stage 1: Sample Identification\n"
#     merged_code += "# ==========================================\n\n"
#     merged_code += stage1_code.strip()
#     merged_code += "\n\n"
    
#     merged_code += "# ==========================================\n"
#     merged_code += "# Stage 2: Properties Extraction\n"
#     merged_code += "# ==========================================\n\n"
    
#     stage2_lines = stage2_code.strip().split('\n')
#     filtered_lines = []
#     skip_mode = False
#     bracket_count = 0
    
#     for line in stage2_lines:
#         stripped = line.strip()
        
#         if 'TitaniumAlloy(' in line:
#             skip_mode = True
#             bracket_count = line.count('(') - line.count(')')
            
#             if bracket_count == 0:
#                 skip_mode = False
#             continue
        
#         if skip_mode:
#             bracket_count += line.count('(') - line.count(')')
            
#             if bracket_count <= 0:
#                 skip_mode = False
#             continue
        
#         if stripped and not stripped.startswith('#'):
#             filtered_lines.append(line)
#         elif stripped.startswith('#'):
#             filtered_lines.append(line)
    
#     stage2_clean = '\n'.join(filtered_lines)
#     merged_code += stage2_clean
    
#     logger.log(f"✅ Merge complete: {len(merged_code)} chars")
    
#     return merged_code


# def process_paper_two_stage(model, tokenizer, paper_text: str, logger: Logger, paper_name: str = "unknown", output_dir: str = "./codekgc_output"):
#     """Two-stage processing pipeline"""
#     try:
#         logger.log("\n" + "=" * 80)
#         logger.log(f"📄 PROCESSING PAPER: {paper_name}")
#         logger.log("=" * 80)
        
#         if not paper_text or len(paper_text.strip()) == 0:
#             logger.log(f"❌ ERROR: Empty paper_text provided for paper: {paper_name}")
#             return None
        
#         logger.log(f"✅ Paper text loaded: {len(paper_text)} characters")
#         logger.log(f"   First 100 chars: '{paper_text[:100]}'")
        
#         logger.log(f"\n{'='*80}")
#         logger.log(f"[STAGE 1] Identifying samples in paper: {paper_name}")
#         logger.log(f"{'='*80}")
#         stage1_code = parse_paper_stage1(model, tokenizer, paper_text, logger)
        
#         if stage1_code is None:
#             logger.log(f"⚠️  [Stage 1] No samples found in paper: {paper_name}")
#             null_code = "# NULL - No titanium alloy samples found in this paper\n"
#             null_filepath = save_generated_code(null_code, paper_name, "final_null", logger, output_dir)
            
#             all_data.append({
#                 'paper': paper_name,
#                 'stage1_code': None,
#                 'stage2_code': None,
#                 'merged_code': null_code,
#                 'filepath': null_filepath,
#                 'status': 'null',
#                 'timestamp': datetime.now().isoformat()
#             })
            
#             return null_code, null_filepath
        
#         logger.log(f"✅ [Stage 1] Successfully identified samples ({len(stage1_code)} characters)")
        
#         logger.log(f"\n{'='*80}")
#         logger.log(f"[STAGE 2] Extracting properties for samples in paper: {paper_name}")
#         logger.log(f"{'='*80}")
#         stage2_code = parse_paper_stage2(model, tokenizer, paper_text, stage1_code, logger)
        
#         if stage2_code is None:
#             logger.log(f"⚠️  [Stage 2] No properties found for paper: {paper_name}")
#         else:
#             logger.log(f"✅ [Stage 2] Successfully extracted properties ({len(stage2_code)} characters)")
        
#         logger.log(f"\n{'='*80}")
#         logger.log(f"[MERGE] Combining Stage 1 and Stage 2 results")
#         logger.log(f"{'='*80}")
#         merged_code = merge_stage_codes(stage1_code, stage2_code, logger)
        
#         if merged_code is None:
#             logger.log(f"❌ [Merge] Failed to merge codes for paper: {paper_name}")
#             return None
        
#         logger.log(f"✅ [Merge] Successfully merged ({len(merged_code)} characters)")
        
#         merged_filepath = save_generated_code(merged_code, paper_name, "final", logger, output_dir)
        
#         all_data.append({
#             'paper': paper_name,
#             'stage1_code': stage1_code,
#             'stage2_code': stage2_code,
#             'merged_code': merged_code,
#             'filepath': merged_filepath,
#             'status': 'success',
#             'timestamp': datetime.now().isoformat()
#         })
        
#         logger.log(f"\n{'='*80}")
#         logger.log(f"✅ COMPLETED: {paper_name}")
#         logger.log(f"{'='*80}\n")
        
#         return merged_code, merged_filepath
    
#     except Exception as e:
#         logger.log(f"❌ Error processing paper {paper_name}: {str(e)}")
#         exceptions.append({
#             'error': str(e),
#             'paper': paper_name,
#             'timestamp': datetime.now().isoformat()
#         })
#         return None


# def parse_paper(model, tokenizer, messages: List[Dict[str, str]], logger: Logger) -> Optional[str]:
#     """Main entry point for external calls (backward compatible)."""
#     logger.log("\n" + "=" * 80)
#     logger.log("🔍 DEBUG - parse_paper() ENTRY POINT")
#     logger.log("=" * 80)
    
#     if len(messages) < 2:
#         logger.log("❌ ERROR: messages must contain at least 2 elements (system and user)")
#         return None
    
#     paper_text = messages[1]["content"]
    
#     logger.log(f"✅ Messages received:")
#     logger.log(f"   Number of messages: {len(messages)}")
#     logger.log(f"   Message[0] role: {messages[0]['role']}")
#     logger.log(f"   Message[0] content length: {len(messages[0]['content'])} chars")
#     logger.log(f"   Message[1] role: {messages[1]['role']}")
#     logger.log(f"   Message[1] content length: {len(messages[1]['content'])} chars")
#     logger.log(f"   Paper text (first 100 chars): '{paper_text[:100]}'")
    
#     if len(paper_text) < 100:
#         logger.log(f"⚠️  WARNING: paper_text seems too short!")
#         logger.log(f"   Full content: '{paper_text[:500]}'")
    
#     logger.log("=" * 80)
    
#     try:
#         logger.log(f"\n[Stage 1] Identifying samples")
#         stage1_code = parse_paper_stage1(model, tokenizer, paper_text, logger)
        
#         if stage1_code is None:
#             logger.log(f"⚠️  [Stage 1] No samples found")
#             return "# NULL\n"
        
#         logger.log(f"✅ [Stage 1] Successfully identified samples ({len(stage1_code)} characters)")
        
#         logger.log(f"\n[Stage 2] Extracting properties for samples")
#         stage2_code = parse_paper_stage2(model, tokenizer, paper_text, stage1_code, logger)
        
#         if stage2_code is None:
#             logger.log(f"⚠️  [Stage 2] No properties found")
#         else:
#             logger.log(f"✅ [Stage 2] Successfully extracted properties ({len(stage2_code)} characters)")
        
#         logger.log(f"\n[Merge] Combining Stage 1 and Stage 2 results")
#         merged_code = merge_stage_codes(stage1_code, stage2_code, logger)
        
#         if merged_code is None:
#             logger.log(f"❌ [Merge] Failed to merge codes")
#             return None
        
#         logger.log(f"✅ [Merge] Successfully merged ({len(merged_code)} characters)")
        
#         return merged_code
    
#     except Exception as e:
#         logger.log(f"❌ Error in two-stage extraction: {str(e)}")
#         exceptions.append({
#             'error': str(e),
#             'type': 'two_stage_extraction_error',
#             'timestamp': datetime.now().isoformat()
#         })
#         return None


# def process_paper(model, tokenizer, messages: List[Dict[str, str]], logger: Logger, paper_name: str = "unknown", output_dir: str = "./codekgc_output"):
#     """Main processing pipeline (backward compatible)."""
#     try:
#         logger.log("\n" + "=" * 80)
#         logger.log(f"🔍 DEBUG - process_paper() ENTRY POINT")
#         logger.log("=" * 80)
        
#         if len(messages) < 2:
#             logger.log(f"❌ ERROR: messages must contain at least 2 elements for paper: {paper_name}")
#             return None
        
#         paper_text = messages[1]["content"]
        
#         logger.log(f"✅ Processing paper: {paper_name}")
#         logger.log(f"   Messages count: {len(messages)}")
#         logger.log(f"   Paper text length: {len(paper_text)} characters")
#         logger.log(f"   First 100 chars: '{paper_text[:100]}'")
        
#         if len(paper_text) < 100:
#             logger.log(f"⚠️  WARNING: paper_text seems too short!")
#             logger.log(f"   Full content: '{paper_text[:500]}'")
        
#         logger.log("=" * 80)
        
#         merged_code = parse_paper(model, tokenizer, messages, logger)
        
#         if merged_code is None or len(merged_code.strip()) == 0:
#             logger.log(f"❌ Failed to process paper: {paper_name}")
#             return None
        
#         if merged_code.strip() == "# NULL" or "# NULL" in merged_code:
#             logger.log(f"⚠️  Paper {paper_name} returned NULL")
#             null_filepath = save_generated_code(merged_code, paper_name, "final_null", logger, output_dir)
            
#             all_data.append({
#                 'paper': paper_name,
#                 'code': merged_code,
#                 'filepath': null_filepath,
#                 'status': 'null',
#                 'timestamp': datetime.now().isoformat()
#             })
            
#             return merged_code, null_filepath
        
#         logger.log(f"✅ Successfully generated CodeKGC code ({len(merged_code)} characters)")
        
#         filepath = save_generated_code(merged_code, paper_name, "final", logger, output_dir)
        
#         if filepath is None:
#             return None
        
#         all_data.append({
#             'paper': paper_name,
#             'code': merged_code,
#             'filepath': filepath,
#             'status': 'success',
#             'timestamp': datetime.now().isoformat()
#         })
        
#         return merged_code, filepath
    
#     except Exception as e:
#         logger.log(f"❌ Error processing paper {paper_name}: {str(e)}")
#         exceptions.append({
#             'error': str(e),
#             'paper': paper_name,
#             'timestamp': datetime.now().isoformat()
#         })
#         return None


# def build_df(all_data, df_to_continue=None):
#     """Build a pandas DataFrame from parsed CodeKGC data."""
#     if not all_data:
#         if df_to_continue is not None:
#             return df_to_continue
#         return pd.DataFrame()
    
#     flattened_data = []
#     for batch in all_data:
#         if isinstance(batch, list):
#             flattened_data.extend(batch)
#         else:
#             flattened_data.append(batch)
    
#     df_new = pd.DataFrame(flattened_data)
    
#     if df_to_continue is not None and len(df_to_continue) > 0:
#         df = pd.concat([df_to_continue, df_new], ignore_index=True)
#     else:
#         df = df_new
    
#     return df


import re
import sys
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Optional

from src.logging import Logger

# 全局变量
all_data = []
exceptions = []
SYSTEM_PROMPT=""

# ==========================================
# 第一阶段：样品抽取 PROMPT
# ==========================================
STAGE1_SYSTEM_PROMPT = '''
You are an expert Materials Science Information Extraction engine based on the GoLLIE (Guideline-following) architecture.

## STAGE 1: SAMPLE IDENTIFICATION

Your task is to identify all distinct **samples** mentioned in the paper and instantiate the following Python class:

# ===== Entity Types =====
class TitaniumAlloy:
    """Represents a titanium alloy sample with its chemical formula/composition.
    
    GUIDELINES for STAGE 1:
    - Each sample is a distinct experimental specimen
    - Samples may differ by: processing conditions, heat treatment, composition variations
    - Extract explicit labels from text (e.g., "Sample A", "S1", "As-received", "Heat-treated")
    - If no explicit label exists, create descriptive labels (e.g., "sample_1", "sample_annealed")
    """
    def __init__(self, name: str, formula: str, label: str, description: str = None):
        """
         Args:
             name: Common name of the titanium alloy (e.g., "Ti-5Al-2.5Sn", "Ti-10V-2Fe-3Al")
             formula: Chemical formula representation (e.g., "Ti-5Al-2.5Sn" or "Ti-10V-2Fe-3Al")
             label: Unique identifier for this sample (e.g., "S1", "Sample-A", "as-cast", "annealed")
             description: Brief description of sample condition (e.g., "annealed at 850C for 3h", "as-received")
        """
        self.name = name
        self.formula = formula
        self.label = label
        self.description = description


### GLOBAL RULES
# 1. **NO Class Definitions**: Class is already defined. DO NOT redefine it.
# 2. **Keyword Arguments ONLY**: Use keyword arguments explicitly.
# 3. **Unique Variable Names**: Assign every object to a distinct variable (e.g., `alloy_1`, `alloy_2`).
# 4. **Format**: Output valid Python code wrapped in ```python and ``` markers.
# 5. **One sample = One TitaniumAlloy instance**: Each distinct sample gets its own alloy object.
# 6. **CRITICAL**: If the paper contains NO titanium alloy samples, or if the input is empty/invalid, output ONLY: ```python\n# NULL\n```

### ONE-SHOT EXAMPLE(YOU SHOULD NOT ADD THIS EXAMPLE INTO RESULT!):

Input Text:
"""
We investigated Ti-5Al-2.5Sn alloy under two different processing routes:
- Specimen X: as-fabricated condition
- Specimen Y: solution treated at 850°C for 3 hours

Table 2 presents mechanical test data for both specimens.
"""

Output Code:
```python
# Define Specimen X
alloy_1 = TitaniumAlloy(
    name="Ti-5Al-2.5Sn", 
    formula="Ti-5Al-2.5Sn", 
    label="Specimen X", 
    description="as-fabricated condition"
)

# Define Specimen Y
alloy_2 = TitaniumAlloy(
    name="Ti-5Al-2.5Sn", 
    formula="Ti-5Al-2.5Sn", 
    label="Specimen Y", 
    description="solution treated at 850°C for 3 hours"
)
```

### EXAMPLE: No Samples Found

Input Text:
"""
This paper reviews computational methods for predicting crystal structures.
"""

Output Code:
```python
# NULL
```

### YOUR TASK
Identify all samples in the user's input paper and generate Python code wrapped in ```python and ```.
If no samples exist, output: ```python\n# NULL\n```
'''


# ==========================================
# 第二阶段：样品属性抽取 PROMPT
# ==========================================
# ==========================================
# 第二阶段：样品属性抽取 PROMPT
# ==========================================
STAGE2_SYSTEM_PROMPT = '''
You are an expert Materials Science Information Extraction engine based on the GoLLIE (Guideline-following) architecture.

## STAGE 2: SAMPLE PROPERTIES EXTRACTION

You will receive:
1. The original paper text
2. A list of identified samples from Stage 1 with their labels

Your task is to extract properties (element composition, heat treatment, mechanical properties) for each sample.

**CRITICAL: FOCUS ON TABLE DATA**
- **Priority 1**: Information enclosed in <table> </table> tags contains structured experimental data. Extract from tables FIRST.
- **Priority 2**: Text descriptions that reference table data (e.g., "as shown in Table 2").
- **Priority 3**: Other textual descriptions of properties.
- Tables typically contain: chemical compositions, mechanical test results, heat treatment parameters.
- Always cross-reference sample labels between tables and text to ensure correct attribution.

**CRITICAL OUTPUT RULES**: 
1. Output ONLY valid Python code wrapped in ```python and ``` markers.
2. NO explanations. NO markdown headers. NO conversational filler. 
3. Start directly with class instantiations inside the code block.
4. If Stage 1 returned NULL (no samples found), output: ```python\n# NULL\n```
5. If you cannot find any properties for the samples, output: ```python\n# NULL\n```

# ===== Entity Types =====
class TitaniumAlloy:
    """Represents a titanium alloy sample (already identified in Stage 1)."""
    def __init__(self, name: str, formula: str, label: str, description: str = None):
        self.name = name
        self.formula = formula
        self.label = label
        self.description = description
        self.elements = []
        self.properties = []
        self.treatment_steps = []


class ElementComposition:
    """Represents a single element in the alloy composition.
    
    GUIDELINES:
    - Extract measured values if available; otherwise use nominal values.
    - Associate with the correct sample label
    - **Look for composition data in tables first** (often labeled as "Chemical Composition", "Composition (wt%)", etc.)
    """
    def __init__(self, 
                 element: str,
                 content: float = None,
                 unit: str = "wt%"):
        self.element = element
        self.content = content
        self.unit = unit


class MechanicalProperty:
    """Represents mechanical properties of a titanium alloy."""
    def __init__(self,
                 UTS_MPa: float = None,
                 elongation_percent: float = None,
                 youngs_modulus_GPa: float = None,
                 vickers_hardness_HV: float = None):
        """
        CRITICAL EXTRACTION GUIDELINES:
        1. **Scope**: Extract the following properties:
           - Ultimate Tensile Strength (UTS) from Tensile Tests
           - Elongation from Tensile Tests
           - Young's Modulus (E) from Tensile Tests or other mechanical testing
           - Vickers Hardness (HV) from Hardness Tests
           
        2. **Exclusions**: 
           - DO NOT extract Yield Strength (YS).
           - DO NOT extract values from Compression tests or Shear tests for UTS/elongation.
           - DO NOT extract 'Reduction of Area' (RA) for elongation.
           - DO NOT extract other hardness scales (Rockwell, Brinell) for Vickers hardness.
           
        3. **Data Source**: 
           - Extract ONLY data belonging to "This Work" (experimental results of the authors).
           - IGNORE values cited from references or literature reviews.
           - IGNORE values derived from Simulations (FEM/DFT).
           - **PRIORITIZE data from <table> tags** - mechanical properties are usually tabulated.
        
        4. **Numeric Handling (CRITICAL)**:
           - **Ranges**: If a range is provided (e.g., "950-1000 MPa"), calculate and extract the **AVERAGE** (975). Do NOT return a string.
           - **Statistics**: If "Mean ± SD" is provided (e.g., "960 ± 15 MPa"), extract the **MEAN** (960). Ignore the standard deviation.
           - **Qualifiers**: Remove symbols like "approx.", "~", ">", "<". Extract only the numeric value.
           
        5. **Units**:
           - UTS_MPa: Always convert to MPa
           - elongation_percent: Always express as percentage (%)
           - youngs_modulus_GPa: Always convert to GPa (if given in MPa, divide by 1000)
           - vickers_hardness_HV: Extract the HV value directly (no unit conversion needed)
           
        6. **Physical Constraints** (IMPORTANT - violations will trigger re-extraction):
           - UTS must be in range [200, 2200] MPa
           - Elongation must be in range [1, 60] %
           - UTS × Elongation product must be ≤ 50,000 MPa·%
           - If extracted values violate these constraints, double-check the table data and units.
        """
        self.UTS_MPa = UTS_MPa
        self.elongation_percent = elongation_percent
        self.youngs_modulus_GPa = youngs_modulus_GPa
        self.vickers_hardness_HV = vickers_hardness_HV


class HeatTreatmentStep:
    """Represents a single heat treatment step.
    
    GUIDELINES:
    1. **Sequence**: Maintain the chronological order of steps.
    2. **Units**: 
       - 'temperature_unit' must be explicitly extracted as 'C' (Celsius) or 'K' (Kelvin).
       - 'time_unit' must be explicitly extracted (e.g., 'h', 'min', 's'). Do NOT convert units yourself.
    3. **Cooling**: Distinguish between 'air_cool', 'furnace_cool', 'water_quench'. If unknown, use 'unspecified'.
    4. **Look for heat treatment tables** - process parameters are often tabulated.
    5. **Physical Constraints** (IMPORTANT - violations will trigger re-extraction):
       - Solution treatment temperature: [600, 1300]°C
       - Aging temperature: [300, 800]°C
       - All temperatures must be < 2000°C (below melting point)
       - In solution+aging processes: aging temperature < solution temperature
       - If extracted values violate these constraints, verify the temperature values and units from tables.
    """
    def __init__(self,
                 step_type: str,
                 temperature: float = None,
                 temperature_unit: str = None,
                 time_value: float = None,
                 time_unit: str = None,
                 cooling_method: str = None,
                 cooling_medium: str = None,
                 cooling_temperature: float = None,
                 cooling_temperature_unit: str = None,
                 atmosphere: str = None,
                 other_parameters: str = None):
        self.step_type = step_type
        self.temperature = temperature
        self.temperature_unit = temperature_unit
        self.time_value = time_value
        self.time_unit = time_unit
        self.cooling_method = cooling_method
        self.cooling_medium = cooling_medium
        self.cooling_temperature = cooling_temperature
        self.cooling_temperature_unit = cooling_temperature_unit
        self.atmosphere = atmosphere
        self.other_parameters = other_parameters
        self.next_step = None


# ===== Relations =====
def has_element(alloy: TitaniumAlloy, element: ElementComposition):
    """Link a titanium alloy with one of its elemental compositions."""
    alloy.elements.append(element)


def has_property(alloy: TitaniumAlloy, prop: MechanicalProperty):
    """Link a titanium alloy with its mechanical properties."""
    alloy.properties.append(prop)


def has_treatment_step(alloy: TitaniumAlloy, step: HeatTreatmentStep):
    """Add one heat treatment step to the alloy's processing route."""
    alloy.treatment_steps.append(step)


def next_step(previous_step: HeatTreatmentStep, next_step_obj: HeatTreatmentStep):
    """Define the processing order: next_step happens after previous_step."""
    previous_step.next_step = next_step_obj


### GLOBAL RULES
# 1. **NO Class Definitions**: Classes are already defined. DO NOT redefine them.
# 2. **Keyword Arguments ONLY**: Use keyword arguments explicitly.
# 3. **Unique Variable Names**: Assign every object to a distinct variable.
# 4. **Relations as Functions**: Use provided relation functions to link entities.
# 5. **Format**: Output valid Python code wrapped in ```python and ``` markers.
# 6. **NEVER** merge properties from different samples into a single object.
# 7. **Match Sample Labels**: Use the EXACT sample labels provided from Stage 1.
# 8. **If Stage 1 is NULL or no properties found**: Output ```python\n# NULL\n```

### ONE-SHOT EXAMPLE(YOU SHOULD NOT ADD THIS EXAMPLE INTO RESULT!):

Stage 1 Output (Samples identified):
"""
```python
alloy_1 = TitaniumAlloy(name="Ti-5Al-2.5Sn", formula="Ti-5Al-2.5Sn", label="Specimen X", description="as-fabricated condition")
alloy_2 = TitaniumAlloy(name="Ti-5Al-2.5Sn", formula="Ti-5Al-2.5Sn", label="Specimen Y", description="solution treated at 850°C for 3 hours")
```
"""

Input Text:
"""
<table>
Table 1: Chemical composition of Ti-5Al-2.5Sn (wt%).
| Element | Ti   | Al  | Sn  |
|---------|------|-----|-----|
| Content | 92.5 | 5.0 | 2.5 |
</table>

<table>
Table 2: Mechanical properties
| Sample | UTS (MPa) | Elongation (%) | Young's Modulus (GPa) | Vickers Hardness (HV) |
|--------|-----------|----------------|----------------------|----------------------|
| Specimen X | 880 | 10 | 110 | 320 |
| Specimen Y | 920 | 13 | 115 | 340 |
</table>

Specimen X was tested in as-fabricated condition.
Specimen Y was solution treated at 850°C for 3 hours and air cooled.
"""

Output Code:
```python
# Recreate Specimen X (from Stage 1)
alloy_1 = TitaniumAlloy(
    name="Ti-5Al-2.5Sn", 
    formula="Ti-5Al-2.5Sn", 
    label="Specimen X", 
    description="as-fabricated condition"
)

# Define Elements for Specimen X (from Table 1)
elem_1_1 = ElementComposition(element="Ti", content=92.5, unit="wt%")
elem_1_2 = ElementComposition(element="Al", content=5.0, unit="wt%")
elem_1_3 = ElementComposition(element="Sn", content=2.5, unit="wt%")
has_element(alloy=alloy_1, element=elem_1_1)
has_element(alloy=alloy_1, element=elem_1_2)
has_element(alloy=alloy_1, element=elem_1_3)

# Define Properties for Specimen X (from Table 2)
prop_1 = MechanicalProperty(
    UTS_MPa=880.0, 
    elongation_percent=10.0,
    youngs_modulus_GPa=110.0,
    vickers_hardness_HV=320.0
)
has_property(alloy=alloy_1, prop=prop_1)

# Recreate Specimen Y (from Stage 1)
alloy_2 = TitaniumAlloy(
    name="Ti-5Al-2.5Sn", 
    formula="Ti-5Al-2.5Sn", 
    label="Specimen Y", 
    description="solution treated at 850°C for 3 hours"
)

# Define Elements for Specimen Y (from Table 1)
elem_2_1 = ElementComposition(element="Ti", content=92.5, unit="wt%")
elem_2_2 = ElementComposition(element="Al", content=5.0, unit="wt%")
elem_2_3 = ElementComposition(element="Sn", content=2.5, unit="wt%")
has_element(alloy=alloy_2, element=elem_2_1)
has_element(alloy=alloy_2, element=elem_2_2)
has_element(alloy=alloy_2, element=elem_2_3)

# Define Heat Treatment for Specimen Y
step_2 = HeatTreatmentStep(
    step_type="solution_treatment",
    temperature=850.0,
    temperature_unit="C",
    time_value=3.0,
    time_unit="h",
    cooling_method="air_cool"
)
has_treatment_step(alloy=alloy_2, step=step_2)

# Define Properties for Specimen Y (from Table 2)
prop_2 = MechanicalProperty(
    UTS_MPa=920.0, 
    elongation_percent=13.0,
    youngs_modulus_GPa=115.0,
    vickers_hardness_HV=340.0
)
has_property(alloy=alloy_2, prop=prop_2)
```

### EXAMPLE: Stage 1 returned NULL

Stage 1 Output:
"""
```python
# NULL
```
"""

Output Code:
```python
# NULL
```

### YOUR TASK
Extract properties for each sample identified in Stage 1. Generate Python code wrapped in ```python and ```.
**Remember to prioritize information from <table> tags as they contain structured experimental data.**
If Stage 1 is NULL or no properties found, output: ```python\n# NULL\n```
'''


# The validator is defined once and is used both as the runtime executed by
# Python and as the exact runtime description shown to Stage 2.  Keeping one
# source of truth prevents the prompt rules and the executable rules drifting
# apart over time.
class DPGSRValidationError(AssertionError):
    """Raised when a generated candidate violates a domain constraint."""

    def __init__(self, errors):
        self.errors = [str(error) for error in errors]
        super().__init__("\n".join(self.errors))


class GeneratedCodeExecutionError(DPGSRValidationError):
    """Raised when generated code cannot be executed in the DPG-SR runtime."""


DOMAIN_VALIDATION_RUNTIME = r'''
class DomainConstraintViolation(AssertionError):
    def __init__(self, errors):
        self.errors = [str(error) for error in errors]
        AssertionError.__init__(self, "\n".join(self.errors))


class TitaniumAlloy:
    def __init__(self, name, formula, label, description=None):
        self.name = name
        self.formula = formula
        self.label = label
        self.description = description
        self.elements = []
        self.properties = []
        self.treatment_steps = []


class ElementComposition:
    def __init__(self, element, content=None, unit="wt%"):
        self.element = element
        self.content = content
        self.unit = unit


class MechanicalProperty:
    def __init__(self, UTS_MPa=None, elongation_percent=None,
                 youngs_modulus_GPa=None, vickers_hardness_HV=None):
        self.UTS_MPa = UTS_MPa
        self.elongation_percent = elongation_percent
        self.youngs_modulus_GPa = youngs_modulus_GPa
        self.vickers_hardness_HV = vickers_hardness_HV


class HeatTreatmentStep:
    def __init__(self, step_type, temperature=None, temperature_unit=None,
                 time_value=None, time_unit=None, cooling_method=None,
                 cooling_medium=None, cooling_temperature=None,
                 cooling_temperature_unit=None, atmosphere=None,
                 other_parameters=None):
        self.step_type = step_type
        self.temperature = temperature
        self.temperature_unit = temperature_unit
        self.time_value = time_value
        self.time_unit = time_unit
        self.cooling_method = cooling_method
        self.cooling_medium = cooling_medium
        self.cooling_temperature = cooling_temperature
        self.cooling_temperature_unit = cooling_temperature_unit
        self.atmosphere = atmosphere
        self.other_parameters = other_parameters
        self.next_step = None


def _check_alloy_constraints(alloy):
    errors = []
    sample_label = getattr(alloy, "label", "unknown_sample")

    # Composition constraints: preserve the existing thresholds and ppm rule.
    if alloy.elements:
        total_mass = 0
        interstitial_elements = {"C", "N", "O", "H"}
        for elem in alloy.elements:
            if elem.content is None:
                continue
            content_wt = elem.content / 10000.0 if elem.unit == "ppm" else elem.content
            total_mass += content_wt
            if elem.element in interstitial_elements and content_wt > 0.5:
                errors.append(
                    f"[{sample_label}] Composition: interstitial {elem.element}="
                    f"{content_wt:.3f} wt% exceeds 0.5 wt% (possible ppm conversion error)."
                )
        if total_mass > 100.01:
            errors.append(
                f"[{sample_label}] Composition: elemental total={total_mass:.2f} wt% exceeds 100 wt%."
            )

    # Mechanical feasibility and strength-ductility constraint.
    for prop in alloy.properties:
        if prop.UTS_MPa is not None and not (200 <= prop.UTS_MPa <= 2200):
            errors.append(
                f"[{sample_label}] Mechanical: UTS={prop.UTS_MPa} MPa is outside [200, 2200]."
            )
        if prop.elongation_percent is not None and not (1 <= prop.elongation_percent <= 60):
            errors.append(
                f"[{sample_label}] Mechanical: elongation={prop.elongation_percent}% is outside [1, 60]."
            )
        if prop.UTS_MPa is not None and prop.elongation_percent is not None:
            product = prop.UTS_MPa * prop.elongation_percent
            if product > 50000:
                errors.append(
                    f"[{sample_label}] Mechanical: UTS×El={product:.0f} exceeds 50000 "
                    f"(UTS={prop.UTS_MPa} MPa, El={prop.elongation_percent}%)."
                )

    # Process constraints: preserve the existing ranges, unit conversion, and
    # solution-treatment/aging ordering rule.
    previous_solution_temp = None
    for step_index, step in enumerate(alloy.treatment_steps):
        if step.temperature is not None:
            temp_c = step.temperature - 273.15 if step.temperature_unit == "K" else step.temperature
            if step.step_type in ["solution_treatment", "solution", "solutionizing"]:
                if not (600 <= temp_c <= 1300):
                    errors.append(
                        f"[{sample_label}] Process: solution temperature={temp_c:.0f}°C is outside [600, 1300]."
                    )
                if step_index == 0:
                    previous_solution_temp = temp_c
            elif step.step_type in ["aging", "age", "ageing"]:
                if not (300 <= temp_c <= 800):
                    errors.append(
                        f"[{sample_label}] Process: aging temperature={temp_c:.0f}°C is outside [300, 800]."
                    )
                if step_index == 1 and previous_solution_temp is not None and temp_c >= previous_solution_temp:
                    errors.append(
                        f"[{sample_label}] Process: aging temperature={temp_c:.0f}°C is not lower than "
                        f"solution temperature={previous_solution_temp:.0f}°C."
                    )
            if temp_c >= 2000:
                errors.append(
                    f"[{sample_label}] Process: temperature={temp_c:.0f}°C is at or above 2000°C."
                )
        if step.time_value is not None:
            if step.time_unit == "h" and step.time_value > 100:
                errors.append(
                    f"[{sample_label}] Process: holding time={step.time_value} h exceeds 100 h (possible unit error)."
                )
            elif step.time_unit == "s" and step.time_value < 0.1:
                errors.append(
                    f"[{sample_label}] Process: holding time={step.time_value} s is below 0.1 s (possible unit error)."
                )
    return errors


def _assert_alloy_constraints(alloy):
    errors = _check_alloy_constraints(alloy)
    if errors:
        raise DomainConstraintViolation(errors)


def has_element(alloy, element):
    alloy.elements.append(element)
    _assert_alloy_constraints(alloy)


def has_property(alloy, prop):
    alloy.properties.append(prop)
    _assert_alloy_constraints(alloy)


def has_treatment_step(alloy, step):
    alloy.treatment_steps.append(step)
    _assert_alloy_constraints(alloy)


def next_step(previous_step, next_step_obj):
    previous_step.next_step = next_step_obj


def validate_material_graph(namespace):
    alloys = [value for value in namespace.values() if isinstance(value, TitaniumAlloy)]
    if not alloys:
        raise DomainConstraintViolation(["No TitaniumAlloy object was created by the consolidated candidate."])
    for alloy in alloys:
        _assert_alloy_constraints(alloy)
    return alloys
'''

STAGE2_SYSTEM_PROMPT += """

### EXECUTABLE DOMAIN-VALIDATION RUNTIME
The following runtime is preloaded verbatim when your Stage 1 and Stage 2
scripts are executed together. Do not redefine these classes or functions.
Use only the constructors and relation functions below. A violation raises an
AssertionError with the exact diagnostic that will be returned to you on the
next retry.

```python
""" + DOMAIN_VALIDATION_RUNTIME + """
```
"""


# ==========================================
# 校验模块：基于钛合金物理约束的规则校验
# ==========================================

class ValidationResult:
    """校验结果类"""
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False


def validate_extracted_code(code_str: str, logger: Logger) -> ValidationResult:
    """
    Execute a consolidated Stage-1 + Stage-2 candidate in the same runtime
    advertised to the model. Domain failures and execution failures are raised
    with machine-readable diagnostics so the retry loop can feed them back.
    """
    if not code_str or code_str.strip() == "# NULL":
        raise DPGSRValidationError(["Candidate code is NULL; there is no material graph to validate."])

    logger.log("🔍 Starting validation...")

    safe_builtins = {
        "__build_class__": __build_class__,
        "AssertionError": AssertionError,
        "Exception": Exception,
        "object": object,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "getattr": getattr,
        "hasattr": hasattr,
        "isinstance": isinstance,
    }
    sandbox = {
        "__builtins__": safe_builtins,
        "__name__": "dpgsr_sandbox",
    }

    try:
        exec(DOMAIN_VALIDATION_RUNTIME, sandbox, sandbox)
        sandbox.update({
            "unknown": None,
            "Unknown": None,
            "bal": None,
            "Bal": None,
            "NaN": None,
            "nan": None,
            "NULL": None,
        })
        exec(code_str, sandbox, sandbox)
        alloys = sandbox["validate_material_graph"](sandbox)
        logger.log(f"✅ Validation passed for {len(alloys)} alloy sample(s)")
        return ValidationResult(is_valid=True)
    except AssertionError as error:
        errors = getattr(error, "errors", None) or [str(error) or error.__class__.__name__]
        logger.log(f"❌ Domain validation failed with {len(errors)} error(s):")
        for item in errors:
            logger.log(f"   - {item}")
        raise DPGSRValidationError(errors) from error
    except Exception as error:
        reason = f"Generated-code execution error ({error.__class__.__name__}): {error}"
        logger.log(f"❌ {reason}")
        raise GeneratedCodeExecutionError([reason]) from error


# ==========================================
# Pipeline Functions
# ==========================================

def extract_code_from_markdown(text: str, logger: Logger) -> Optional[str]:
    """
    Extract code from markdown code blocks.
    Supports ```python and ``` markers.
    Returns None if # NULL is found.
    """
    if not text or len(text.strip()) == 0:
        logger.log("⚠️  extract_code_from_markdown: Empty input")
        return None
    
    # Check for NULL marker
    if "# NULL" in text:
        logger.log("🔍 Found NULL marker in generated code")
        return None
    
    # Try to extract code from ```python ... ``` blocks
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        code = match.group(1).strip()
        logger.log(f"✅ Extracted code from ```python block ({len(code)} chars)")
        
        # Double check for NULL
        if code.strip() == "# NULL" or code.strip() == "#NULL":
            logger.log("🔍 Code block contains only NULL marker")
            return None
        
        return code
    
    # Try generic ``` ... ``` blocks
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    
    if match_generic:
        code = match_generic.group(1).strip()
        logger.log(f"✅ Extracted code from generic ``` block ({len(code)} chars)")
        
        # Double check for NULL
        if code.strip() == "# NULL" or code.strip() == "#NULL":
            logger.log("🔍 Code block contains only NULL marker")
            return None
        
        return code
    
    # No code blocks found, return the text as-is (fallback)
    logger.log("⚠️  No code blocks found, using raw text as fallback")
    return text.strip()


def parse_paper_stage1(model, tokenizer, paper_text: str, logger: Logger) -> Optional[str]:
    """Stage 1: Extract samples"""
    
    logger.log("=" * 80)
    logger.log("🔍 DEBUG STAGE 1 - INPUT CHECK")
    logger.log("=" * 80)
    
    if not paper_text or len(paper_text.strip()) == 0:
        logger.log("❌ ERROR: paper_text is EMPTY in parse_paper_stage1!")
        return None
    
    logger.log(f"✅ Stage 1 - Paper text received")
    logger.log(f"   Length: {len(paper_text)} characters")
    logger.log(f"   First 100 chars: '{paper_text[:100]}'")
    logger.log(f"   Last 100 chars: '{paper_text[-100:]}'")
    logger.log("=" * 80)
    
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": paper_text}
    ]
    
    raw_output = _generate_code(model, tokenizer, messages, logger, stage="STAGE1")
    
    if raw_output is None:
        return None
    
    # Extract code from markdown blocks
    extracted_code = extract_code_from_markdown(raw_output, logger)
    
    return extracted_code


def parse_paper_stage2(model, tokenizer, paper_text: str, stage1_samples: str, logger: Logger) -> Optional[str]:
    """Stage 2: Extract properties for identified samples"""
    
    logger.log("=" * 80)
    logger.log("🔍 DEBUG STAGE 2 - INPUT CHECK")
    logger.log("=" * 80)
    
    # Check if Stage 1 is NULL
    if stage1_samples is None:
        logger.log("⚠️  Stage 1 returned NULL, skipping Stage 2")
        return None
    
    # Check paper text
    if not paper_text or len(paper_text.strip()) == 0:
        logger.log("❌ ERROR: paper_text is EMPTY in parse_paper_stage2!")
        return None
    else:
        logger.log(f"✅ Stage 2 - Paper text received")
        logger.log(f"   Length: {len(paper_text)} characters")
        logger.log(f"   First 100 chars: '{paper_text[:100]}'")
    
    # Check Stage 1 results
    if len(stage1_samples.strip()) == 0:
        logger.log("❌ ERROR: stage1_samples is EMPTY in parse_paper_stage2!")
        return None
    else:
        logger.log(f"✅ Stage 2 - Stage 1 samples received")
        logger.log(f"   Length: {len(stage1_samples)} characters")
        logger.log(f"   First 100 chars: '{stage1_samples[:100]}'")
    
    logger.log("=" * 80)
    
    # Combine paper text with Stage 1 results
    combined_input = f"""## Stage 1 Identified Samples:
```python
{stage1_samples}
```

## Original Paper Text:
{paper_text}
"""
    
    logger.log(f"📦 Combined input for Stage 2:")
    logger.log(f"   Total length: {len(combined_input)} characters")
    logger.log(f"   First 200 chars: '{combined_input[:200]}'")
    logger.log("=" * 80)
    
    messages = [
        {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
        {"role": "user", "content": combined_input}
    ]
    
    raw_output = _generate_code(model, tokenizer, messages, logger, stage="STAGE2")
    
    if raw_output is None:
        return None
    
    # Extract code from markdown blocks
    extracted_code = extract_code_from_markdown(raw_output, logger)
    
    return extracted_code


def build_consolidated_candidate(stage1_code: str, stage2_code: str) -> Optional[str]:
    """Build the exact Stage-1 + Stage-2 program evaluated by DPG-SR."""
    if stage1_code is None or stage2_code is None:
        return None

    return (
        "# ==========================================\n"
        "# Stage 1: Sample Identification\n"
        "# ==========================================\n\n"
        f"{stage1_code.strip()}\n\n"
        "# ==========================================\n"
        "# Stage 2: Properties Extraction\n"
        "# ==========================================\n\n"
        f"{stage2_code.strip()}\n"
    )


def parse_paper_stage2_with_validation(
    model, 
    tokenizer, 
    paper_text: str, 
    stage1_samples: str, 
    logger: Logger, 
    max_retries: int = 3
) -> Optional[str]:
    """
    Stage 2 with validation: 最多重试3次直到通过校验
    
    Args:
        max_retries: 最大重试次数（默认3次）
    
    Returns:
        通过校验的代码，或None（如果3次都失败）
    """
    
    validation_errors_history = []  # 记录历史错误
    
    for attempt in range(max_retries):
        logger.log(f"\n{'='*80}")
        logger.log(f"[STAGE 2 - Attempt {attempt + 1}/{max_retries}] Extracting properties")
        logger.log(f"{'='*80}")
        
        # 构建增强的输入（如果有之前的错误，则添加错误提示）
        if attempt > 0 and validation_errors_history:
            # 构建错误反馈提示
            error_feedback = "\n\n## VALIDATION ERRORS FROM PREVIOUS ATTEMPT:\n"
            error_feedback += "The previous extraction violated the following physical constraints:\n\n"
            
            for i, error in enumerate(validation_errors_history[-1], 1):
                error_feedback += f"{i}. {error}\n"
            
            error_feedback += "\n**CRITICAL INSTRUCTIONS FOR RETRY**:\n"
            error_feedback += "- Double-check all numeric values extracted from tables\n"
            error_feedback += "- Verify unit conversions (especially for temperatures: C vs K, times: h vs min vs s)\n"
            error_feedback += "- For UTS and Elongation: ensure values are realistic for titanium alloys\n"
            error_feedback += "- For heat treatment: verify temperature ranges and sequence (solution > aging)\n"
            error_feedback += "- For composition: check if ppm values need conversion to wt%\n"
            error_feedback += "- Re-examine the <table> data carefully to ensure correct extraction\n\n"
            
            # 将错误反馈添加到paper_text
            enhanced_paper_text = error_feedback + paper_text
        else:
            enhanced_paper_text = paper_text
        
        # 调用Stage 2提取函数（使用增强的输入）
        stage2_code = parse_paper_stage2(model, tokenizer, enhanced_paper_text, stage1_samples, logger)
        
        if stage2_code is None:
            logger.log(f"⚠️  [Attempt {attempt + 1}] No properties found")
            return None
        
        # Validate the consolidated candidate, not Stage 2 in isolation.  This
        # preserves the paper's c1 ⊕ c2 execution semantics and lets Python's
        # ordinary last-write-wins assignment behavior resolve duplicate names.
        logger.log(f"\n[VALIDATION - Attempt {attempt + 1}] Checking extracted properties")
        candidate_code = build_consolidated_candidate(stage1_samples, stage2_code)
        try:
            validate_extracted_code(candidate_code, logger)
        except DPGSRValidationError as error:
            validation_errors_history.append(error.errors)
            logger.log(f"❌ [Attempt {attempt + 1}] Validation failed with {len(error.errors)} error(s):")
            for item in error.errors:
                logger.log(f"   {item}")

            if attempt < max_retries - 1:
                logger.log(f"🔄 Retrying extraction with error feedback (attempt {attempt + 2}/{max_retries})...")
                continue

            logger.log(f"⚠️  Max retries ({max_retries}) reached. Discarding this extraction.")
            logger.log(f"   Final diagnostics: {error.errors}")
            return None
        else:
            logger.log(f"✅ [Attempt {attempt + 1}] Validation passed!")
            return stage2_code
    
    return None


def _generate_code(model, tokenizer, messages: List[Dict[str, str]], logger: Logger, stage: str = "UNKNOWN") -> Optional[str]:
    """Common code generation logic"""
    assert messages[0]["role"] == "system", "First prompt must be system prompt"
    assert messages[1]["role"] == "user", "Second prompt must be a user prompt"
    
    logger.log(f"🚀 [{stage}] Starting code generation...")
    logger.log(f"   System prompt length: {len(messages[0]['content'])} chars")
    logger.log(f"   User content length: {len(messages[1]['content'])} chars")
    
    # Build the prompt string
    prompt = ""
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        role_start = "<|im_start|>"
        role_end = "<|im_end|>"
        for m in messages:
            prompt += f"{role_start}{m['role']}\n{m['content']}\n{role_end}\n"
        prompt += f"{role_start}assistant\n"

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    logger.log(f"   Input tokens shape: {input_ids.shape}")
    logger.log(f"   Total prompt length: {len(prompt)} chars")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[-1]:]
    code_str = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    logger.log(f"✅ [{stage}] Generation complete")
    logger.log(f"   Generated code length: {len(code_str)} chars")
    logger.log(f"   First 200 chars: '{code_str[:200]}'")
    
    return code_str


def save_generated_code(code_str: str, paper_name: str, stage: str, logger: Logger, output_dir: str = "./codekgc_output"):
    """Save the generated Python code to a file."""
    import os
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        # Sanitize paper name for filename
        safe_paper_name = re.sub(r'[<>:"/\\|?*]', '_', paper_name)
        
        filename = f"{safe_paper_name}_{stage}_{timestamp}.py"
        filepath = os.path.join(output_dir, filename)
        
        # Write code to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code_str)
        
        logger.log(f"💾 Generated code saved to: {filepath}")
        return filepath
    
    except Exception as e:
        logger.log(f"❌ Error saving generated code: {str(e)}")
        exceptions.append({
            'error': str(e),
            'type': 'save_code_error',
            'paper': paper_name,
            'stage': stage,
            'timestamp': datetime.now().isoformat()
        })
        return None


def merge_stage_codes(stage1_code: str, stage2_code: str, logger: Logger) -> Optional[str]:
    """
    Merge Stage 1 and Stage 2 codes into a single file.
    
    The merged result is deliberately identical to the candidate that passed
    DPG-SR validation.  Stage-2 anchor reassignments are retained so Python
    resolves them with ordinary last-write-wins semantics, as specified by the
    paper.
    """
    
    # Handle NULL cases
    if stage1_code is None:
        logger.log("⚠️  Stage 1 is NULL, cannot merge")
        return None
    
    if stage2_code is None:
        logger.log("⚠️  Stage 2 is NULL or failed DPG-SR validation; discarding the extraction")
        return None
    
    logger.log(f"🔗 Merging Stage 1 ({len(stage1_code)} chars) and Stage 2 ({len(stage2_code)} chars)")
    
    merged_code = build_consolidated_candidate(stage1_code, stage2_code)
    
    logger.log(f"✅ Merge complete: {len(merged_code)} chars")
    
    return merged_code


def process_paper_two_stage(model, tokenizer, paper_text: str, logger: Logger, paper_name: str = "unknown", output_dir: str = "./codekgc_output"):
    """
    Two-stage processing pipeline with validation:
    Stage 1: Identify samples
    Stage 2: Extract properties for each sample (with validation and retry)
    Final: Merge into a single file
    """
    try:
        logger.log("\n" + "=" * 80)
        logger.log(f"📄 PROCESSING PAPER: {paper_name}")
        logger.log("=" * 80)
        
        # Check if paper_text is empty
        if not paper_text or len(paper_text.strip()) == 0:
            logger.log(f"❌ ERROR: Empty paper_text provided for paper: {paper_name}")
            return None
        
        logger.log(f"✅ Paper text loaded: {len(paper_text)} characters")
        logger.log(f"   First 100 chars: '{paper_text[:100]}'")
        
        # Stage 1: Sample Identification
        logger.log(f"\n{'='*80}")
        logger.log(f"[STAGE 1] Identifying samples in paper: {paper_name}")
        logger.log(f"{'='*80}")
        stage1_code = parse_paper_stage1(model, tokenizer, paper_text, logger)
        
        if stage1_code is None:
            logger.log(f"⚠️  [Stage 1] No samples found in paper: {paper_name}")
            # Save NULL result
            null_code = "# NULL - No titanium alloy samples found in this paper\n"
            null_filepath = save_generated_code(null_code, paper_name, "final_null", logger, output_dir)
            
            all_data.append({
                'paper': paper_name,
                'stage1_code': None,
                'stage2_code': None,
                'merged_code': null_code,
                'filepath': null_filepath,
                'status': 'null',
                'timestamp': datetime.now().isoformat()
            })
            
            return null_code, null_filepath
        
        logger.log(f"✅ [Stage 1] Successfully identified samples ({len(stage1_code)} characters)")
        
        # Stage 2: Property Extraction WITH VALIDATION (最多重试3次)
        logger.log(f"\n{'='*80}")
        logger.log(f"[STAGE 2] Extracting properties with validation for paper: {paper_name}")
        logger.log(f"{'='*80}")
        stage2_code = parse_paper_stage2_with_validation(
            model, tokenizer, paper_text, stage1_code, logger, max_retries=3
        )
        
        if stage2_code is None:
            logger.log(f"⚠️  [Stage 2] No properties found for paper: {paper_name}")
        else:
            logger.log(f"✅ [Stage 2] Successfully extracted and validated properties ({len(stage2_code)} characters)")
        
        # Merge Stage 1 and Stage 2
        logger.log(f"\n{'='*80}")
        logger.log(f"[MERGE] Combining Stage 1 and Stage 2 results")
        logger.log(f"{'='*80}")
        merged_code = merge_stage_codes(stage1_code, stage2_code, logger)
        
        if merged_code is None:
            logger.log(f"❌ [Merge] Failed to merge codes for paper: {paper_name}")
            return None
        
        logger.log(f"✅ [Merge] Successfully merged ({len(merged_code)} characters)")
        
        # Save merged output (final single file)
        merged_filepath = save_generated_code(merged_code, paper_name, "final", logger, output_dir)
        
        # Store results
        all_data.append({
            'paper': paper_name,
            'stage1_code': stage1_code,
            'stage2_code': stage2_code,
            'merged_code': merged_code,
            'filepath': merged_filepath,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.log(f"\n{'='*80}")
        logger.log(f"✅ COMPLETED: {paper_name}")
        logger.log(f"{'='*80}\n")
        
        return merged_code, merged_filepath
    
    except Exception as e:
        logger.log(f"❌ Error processing paper {paper_name}: {str(e)}")
        exceptions.append({
            'error': str(e),
            'paper': paper_name,
            'timestamp': datetime.now().isoformat()
        })
        return None


def parse_paper(model, tokenizer, messages: List[Dict[str, str]], logger: Logger) -> Optional[str]:
    """
    Main entry point for external calls (backward compatible).
    """
    logger.log("\n" + "=" * 80)
    logger.log("🔍 DEBUG - parse_paper() ENTRY POINT")
    logger.log("=" * 80)
    
    # Extract paper text
    if len(messages) < 2:
        logger.log("❌ ERROR: messages must contain at least 2 elements (system and user)")
        return None
    
    paper_text = messages[1]["content"]
    
    # Debug logs
    logger.log(f"✅ Messages received:")
    logger.log(f"   Number of messages: {len(messages)}")
    logger.log(f"   Message[0] role: {messages[0]['role']}")
    logger.log(f"   Message[0] content length: {len(messages[0]['content'])} chars")
    logger.log(f"   Message[1] role: {messages[1]['role']}")
    logger.log(f"   Message[1] content length: {len(messages[1]['content'])} chars")
    logger.log(f"   Paper text (first 100 chars): '{paper_text[:100]}'")
    
    if len(paper_text) < 100:
        logger.log(f"⚠️  WARNING: paper_text seems too short!")
        logger.log(f"   Full content: '{paper_text[:500]}'")
    
    logger.log("=" * 80)
    
    # Run two-stage extraction with validation
    try:
        # Stage 1: Sample Identification
        logger.log(f"\n[Stage 1] Identifying samples")
        stage1_code = parse_paper_stage1(model, tokenizer, paper_text, logger)
        
        if stage1_code is None:
            logger.log(f"⚠️  [Stage 1] No samples found")
            return "# NULL\n"
        
        logger.log(f"✅ [Stage 1] Successfully identified samples ({len(stage1_code)} characters)")
        
        # Stage 2: Property Extraction WITH VALIDATION
        logger.log(f"\n[Stage 2] Extracting properties for samples with validation")
        stage2_code = parse_paper_stage2_with_validation(
            model, tokenizer, paper_text, stage1_code, logger, max_retries=3
        )
        
        if stage2_code is None:
            logger.log(f"⚠️  [Stage 2] No properties found")
        else:
            logger.log(f"✅ [Stage 2] Successfully extracted and validated properties ({len(stage2_code)} characters)")
        
        # Merge Stage 1 and Stage 2
        logger.log(f"\n[Merge] Combining Stage 1 and Stage 2 results")
        merged_code = merge_stage_codes(stage1_code, stage2_code, logger)
        
        if merged_code is None:
            logger.log(f"❌ [Merge] Failed to merge codes")
            return None
        
        logger.log(f"✅ [Merge] Successfully merged ({len(merged_code)} characters)")
        
        return merged_code
    
    except Exception as e:
        logger.log(f"❌ Error in two-stage extraction: {str(e)}")
        exceptions.append({
            'error': str(e),
            'type': 'two_stage_extraction_error',
            'timestamp': datetime.now().isoformat()
        })
        return None


def process_paper(model, tokenizer, messages: List[Dict[str, str]], logger: Logger, paper_name: str = "unknown", output_dir: str = "./codekgc_output"):
    """
    Main processing pipeline (backward compatible).
    Uses two-stage extraction with validation and saves the merged result.
    """
    try:
        logger.log("\n" + "=" * 80)
        logger.log(f"🔍 DEBUG - process_paper() ENTRY POINT")
        logger.log("=" * 80)
        
        # Extract paper text
        if len(messages) < 2:
            logger.log(f"❌ ERROR: messages must contain at least 2 elements for paper: {paper_name}")
            return None
        
        paper_text = messages[1]["content"]
        
        # Debug logs
        logger.log(f"✅ Processing paper: {paper_name}")
        logger.log(f"   Messages count: {len(messages)}")
        logger.log(f"   Paper text length: {len(paper_text)} characters")
        logger.log(f"   First 100 chars: '{paper_text[:100]}'")
        
        if len(paper_text) < 100:
            logger.log(f"⚠️  WARNING: paper_text seems too short!")
            logger.log(f"   Full content: '{paper_text[:500]}'")
        
        logger.log("=" * 80)
        
        # Use the parse_paper function (which does two-stage extraction with validation)
        merged_code = parse_paper(model, tokenizer, messages, logger)
        
        if merged_code is None or len(merged_code.strip()) == 0:
            logger.log(f"❌ Failed to process paper: {paper_name}")
            return None
        
        # Check if it's NULL
        if merged_code.strip() == "# NULL" or "# NULL" in merged_code:
            logger.log(f"⚠️  Paper {paper_name} returned NULL")
            null_filepath = save_generated_code(merged_code, paper_name, "final_null", logger, output_dir)
            
            all_data.append({
                'paper': paper_name,
                'code': merged_code,
                'filepath': null_filepath,
                'status': 'null',
                'timestamp': datetime.now().isoformat()
            })
            
            return merged_code, null_filepath
        
        logger.log(f"✅ Successfully generated CodeKGC code ({len(merged_code)} characters)")
        
        # Save merged output
        filepath = save_generated_code(merged_code, paper_name, "final", logger, output_dir)
        
        if filepath is None:
            return None
        
        all_data.append({
            'paper': paper_name,
            'code': merged_code,
            'filepath': filepath,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
        return merged_code, filepath
    
    except Exception as e:
        logger.log(f"❌ Error processing paper {paper_name}: {str(e)}")
        exceptions.append({
            'error': str(e),
            'paper': paper_name,
            'timestamp': datetime.now().isoformat()
        })
        return None


def build_df(all_data, df_to_continue=None):
    """Build a pandas DataFrame from parsed CodeKGC data."""
    if not all_data:
        if df_to_continue is not None:
            return df_to_continue
        return pd.DataFrame()
    
    # Flatten the all_data list
    flattened_data = []
    for batch in all_data:
        if isinstance(batch, list):
            flattened_data.extend(batch)
        else:
            flattened_data.append(batch)
    
    df_new = pd.DataFrame(flattened_data)
    
    if df_to_continue is not None and len(df_to_continue) > 0:
        df = pd.concat([df_to_continue, df_new], ignore_index=True)
    else:
        df = df_new
    
    return df