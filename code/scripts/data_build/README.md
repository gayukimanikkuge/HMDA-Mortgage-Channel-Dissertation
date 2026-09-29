# Early data-construction scripts 

The final HMDA channel dataset was constructed and audited with the following local scripts before the numbered analysis pipeline in the repository:

1. `01_build_channel_analysis.py`  
   Main raw HMDA construction script. Built the eligible channel-analysis base dataset and lender-year support informationn. Key final outputs included `channel_analysis_base_parts/` and later `channel_analysis_v1_parts/`.

2. `02_recover_second_pass.py`  
   Recovery/enrichment script used after a disk space failure during the second pass.

3. `03_audit_final_dataset.py`  
   Final-dataset audit script used to verify the rebuilt dataset after recovery.

4. `04_verify_hmda_codes.py`  
   Initial code-verification script. 

5. `05_extract_cleaning_functions.py`  
   Extracted the cleaning functions from `01_build_channel_analysis.py` for direct review, including geography, sex, co-applicant, age and DTI classification functions.

6. `06_raw_hmda_code_audit.py`  
   Audited the construction mappings against raw annual HMDA records.

7. `07_geography_final_audit.py`  
   Geography-code audit after cleaning changes.

8. `07b_geography_main_sample_impact.py`  
   Checked the effect of geography handling on the analysis sample.

9. `08_recovery_integrity_audit.py`  
   Verified the recovery process: 500 base parts and 500 V1 parts, preserved base content/order, enrichment columns added, and no loss/duplication/truncation.

