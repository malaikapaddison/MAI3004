# MAI3004

# Research plan: 
# Objective -> 
# The segmentation of H&N tumours and affected lymph nodes with the help of machine learning models to predict recurrence free survival (RFS) on PET and CT scans. The RFS then helps stratify the patients into three main groups: low, medium and high risk, which will help doctors decide what course of action is best for the patient.

# Defining the problem ->
# Increasing incidence of H&N cancers worldwide, there is a need to develop these automatic models to aid clinicians, reduce their workload and improve inter-/intra-observer variability (more precise segmentations -> better treatment outcomes)

# Participants/dataset ->
# Our research group got access to the HECKTOR 2022 (HEad and neCK tumOR segmentation and outcome prediction in PET/CT images) dataset and will be using this data to make our prediction models.

# Time frame ->
# We have four weeks to complete the project:
# Week 1: data cleaning and curating, learn how the various tools work (ITK-SNAP, PyRadiomics, etc.), start extracting some features.
# Week 2: begin building the first models (only clinical features, only tumour_all features, and tumour_all and lymph_node_all features), using the C-index as an evaluation metric 
# Week 3: extract the features from the rest of the regions of interest (ROI’s), volume related features?, combining the features to get the best outcomes for the models (C-index → 1.0)
# Week 4: similar to week three, find the best models with combining different ROI features, try out different ML models, refine the code and prepare to present outcomes.
