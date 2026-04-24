Associate Editor Comments:
Associate Editor
REQUIRED: Comments to the Author:
Although the paper tackles an important problem in high-resolution remote sensing change detection and the reviewers acknowledge that the proposed FACDNet shows some promise, the three reports converge on several substantial weaknesses that justify a recommendation of "Reject with invitation to resubmit". In particular, the reviewers question whether the method offers sufficient novelty beyond a combination of existing ideas, noting that its distinctions from closely related frequency-aware and Transformer-based approaches are not clearly established. They also highlight important shortcomings in methodological clarity and reproducibility, including missing implementation details, incomplete mathematical formulation, limited explanation of key design choices, and unclear visualization of the overall architecture. In addition, the experimental validation is considered insufficiently rigorous, with outdated or incomplete baselines, limited ablation studies concentrated on a single dataset, and missing efficiency analyses. Reviewer 3 also raises concerns about the conceptual soundness of the feature alignment and frequency reconstruction components, including the risk of incorrectly aligning true semantic changes and the lack of explicit consistency constraints or direct supervision for filtered features. Finally, all three reviewers note issues with presentation, such as figure clarity, language quality, formatting, and references. Taken together, these concerns suggest that the manuscript requires substantial revision in terms of novelty justification, technical detail, experimental support, and presentation before it can be considered for publication.

Reviewer(s) Comments:
Reviewer: 1

Comments to the Author
The paper addresses the core challenge of distinguishing between "phenological pseudo-changes" and "real semantic changes" in high-resolution remote sensing change detection. It innovatively proposes the FACDNet network, which combines feature distribution alignment with frequency-domain aware filtering. Experiments on three mainstream datasets fully demonstrate the method's effectiveness and advancement. However, some details can be improved to make the work more rigorous and polished.
(1) Regarding WHU-CD, the paper only mentions its resolution. It is suggested to supplement its sample size or main characteristics to balance the description of the three datasets.
(2) When analyzing the results on SYSU-CD, it is mentioned that "our method outperforms the second-best ELGCNet by significant margins in both F1-score and Recall." From the table, the improvement of FACDNet is evident: F1 (80.99) vs. ELGCNet (79.46), Recall (77.14) vs. ELGCNet (74.39). However, the Precision is slightly lower (85.24 vs. 85.28). It is suggested to add an explanation here, indicating that the improvement in recall is more valuable, or that suppressing pseudo-changes in complex scenes may lead to a slight decrease in precision while the overall F1 improves, which aligns with the design goal.
(3) In the "Configuration" column of Table III, the term "Base" is ambiguous. It is suggested to clearly define "Base" in a table footnote or in the main text as "plain Siamese Transformer without any proposed modules."
(4) What is the reason for choosing the Haar wavelet in the paper? Is it due to its computational simplicity and effectiveness in capturing horizontal and vertical edges, or does it perform better in this task compared to other wavelet bases (e.g., Daubechies)? It is suggested to add a brief explanation or citation.
(5) In formulas (6) and (7), a local neighborhood Ω (e.g., 3x3) is mentioned. Is this filter size a fixed hyperparameter? Were other sizes attempted? Why was 3x3 chosen? Was using different filter sizes for feature maps at different scales considered? It is suggested to add an analysis of the filter size in the main text or ablation experiments, or to explicitly state that choosing 3x3 is empirical and consistent with the receptive field of standard convolutions.
(6) There are a few spelling or grammatical errors in the paper, for example:
"leading mere to false alarms" -> "leading merely to false alarms" or "leading to false alarms."
"To enhance feature interactions and restore the original dimension" – Does "original dimension" here refer to the number of channels? It is suggested to clarify as "restore the channel dimension."
"greg change detection" -> "robust change detection" ("greg" appears to be a typo for "robust").
Punctuation marks (such as commas and periods) in all formulas should be unified to English punctuation.
(7) The format of the references is incorrect. For example:
The conference name abbreviation in reference [4] "IEEE International Conference on Image Processing (ICIP)" is standard.
The conference name in reference [8] "IGARSS" also needs to provide its full name "IEEE International Geoscience and Remote Sensing Symposium."
Reference [12] "IEEE Geoscience and Remote Sensing Letters, 2025" lacks the volume number, issue number, and page numbers. Please provide the complete information.


Reviewer: 2

Comments to the Author
This paper addresses the issues of pseudo-change interference and blurred boundaries in high-resolution remote sensing change detection by proposing a Feature-Aligned and Frequency-Aware Network (FACDNet), which consists of a Feature Distribution Alignment Module (FDAM), a Frequency-Adaptive Filtering Module (FAFM), and a Change-Aware Guidance Module (CAGM). I have the following comments regarding this manuscript:
1.The three proposed modules correspond to feature alignment, frequency-domain filtering, and attention-based fusion, respectively. However, these ideas can all be found in closely related prior works. The current manuscript appears to be more of a compositional improvement built upon existing methods. It is recommended that the authors more clearly justify the novelty of this combination and explicitly highlight the differences from Transformer-based change detection approaches.
2.Although the overall framework is described, critical implementation details are missing. For example, the specific backbone used in the Siamese encoder, the configuration of the Transformer blocks, feature dimensionality at each stage, the handling of DWT/IDWT across different scales, and the detailed architecture of the decoder are not sufficiently specified. These details should be provided to ensure reproducibility.
3.The distinctions between the proposed method and the most closely related works are not clearly articulated. In particular, it is unclear what limitations of existing frequency-aware or change detection methods remain unresolved, what additional problems are addressed by this work, and why the proposed design is more effective. The related work section lacks targeted and in-depth comparative analysis.
4.Although experiments are conducted on three datasets, the comparison lacks strong recent baselines. It is recommended to include more state-of-the-art methods and to provide efficiency analyses such as parameter count and FLOPs.
5.The current ablation experiments only analyze module combinations and the loss parameter λ. It is suggested to further investigate the contribution of each module in more detail and to validate their effectiveness on more complex datasets.
6.Some English expressions are not sufficiently natural, and the sentence structures are occasionally awkward. It is recommended that the authors carefully revise the manuscript for language quality and formatting.



Reviewer: 3

Comments to the Author
To address the feature distribution shift in bi-temporal remote sensing images, this paper proposes a Feature-Aligned and Frequency-Aware Network (FACDNet). The manuscript is well written and demonstrates acceptable innovation in general. However, several issues need to be revised to further improve quality and clarity:
1. I believe Figure 1 cannot clearly illustrate the proposed method. The overall network structure is not clearly presented. In addition, the meanings of (a), (b), (c) and their positions in the full pipeline are not explained. Key details such as “we integrate a FDAM with a transformer block in each encoding stage” are also not visualized in the figure.
2. The paper mentions “suppressing non-semantic style discrepancies”, but fails to explain how semantic and non-semantic components are distinguished. These two concepts are easily confused in real remote sensing scenarios. There is a lack of analysis and discussion on whether FDAM may incorrectly align real semantic changes.
3. Consistency constraints for frequency-domain reconstruction are missing. After filtering, features are reconstructed via IDWT without ensuring consistency in energy or gradient between the reconstructed features and original features, which may cause information distortion.
4. The mathematical formulation of Wavelet Loss is not provided, which may bring challenges to reproducibility.
5. The output of FAFM does not directly participate in loss computation and only affects the final prediction indirectly through CAGM. This design may lead to insufficient training of the frequency-domain filtering module.
6. Although the paper mentions “three benchmark datasets”, ablation experiments are only provided on the LEVIR-CD dataset (Table III).
7. The comparison methods are relatively outdated, and comparisons with recent state-of-the-art methods are insufficient.
8. There are many minor spelling and grammatical errors throughout the manuscript. A careful proofread and revision is suggested.