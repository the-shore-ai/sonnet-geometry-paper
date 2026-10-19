# Prior work on depth, order disruption and alignment

Targeted check: **2026-10-18**, before inspecting v4 geometric outcomes. This
addendum supplements [the literature review](literature-review.md) and
[novelty assessment](novelty-assessment.md). It is a bounded primary-source
check, not an exhaustive priority search. Public queries contained general
research terms and published titles, without private results or project details.

**Overlap is substantial.** Order perturbations across transformer depth,
lexical shortcuts under shuffling, tokenizer-dependent controls, cross-model
alignment, and misleading alignment under corruption all have precedents.
The useful remaining question is whether the precisely specified relationships
hold in this population and measurement setup, with their numerical and
provenance limits exposed.

## Five close references

**Matteo Alleman, Jonathan Mamou, Miguel A Del Rio, Hanlin Tang, Yoon Kim and
SueYeon Chung (2021), [Syntactic Perturbations Reveal Representational Correlates
of Hierarchical Phrase Structure in Pretrained Language Models](https://arxiv.org/abs/2104.07578).**
The [full text](https://arxiv.org/html/2104.07578v1), especially §§2–4 and
Figures 2–5, directly studies representational change across layers under
n-gram shuffling, phrase swaps and adjacent-word swaps. Larger preserved
contexts delay distortion to deeper layers; sensitivity to disrupted phrase
structure also develops across depth. Several transformer architectures are
examined. This is the closest methodological predecessor for the depth/order
question. Its main measure realigns corresponding words and compares
standardized within-sentence representation matrices; it also examines
CCA/PWCCA and a cosine-related measure. Our across-poem cosine-RDM Spearman
comparison is a different estimand. The paper does not establish our particular
between-model change under shuffling. The arXiv preprint and methods were
verified; a corresponding released implementation and publication venue were
not verified in this pass.

**Thang Pham, Trung Bui, Long Mai and Anh Nguyen (2021), [Out of Order: How
important is the sequential order of words in a sentence in Natural Language
Understanding tasks?](https://aclanthology.org/2021.findings-acl.98/)**
*Findings of ACL-IJCNLP*, pp.1145–1160; DOI 10.18653/v1/2021.findings-acl.98.
The [full text](https://arxiv.org/html/2012.15180v2), §§3.1–3.4, shows that many
correct classifier predictions survive word shuffling, with lexical
attribution and early attention analyses explaining some shortcuts. This
already challenges the inference from apparently successful behavior to
preserved sentence meaning. It concerns fine-tuned classification and
attention/attribution evidence, rather than the pooled cross-model RDM test.
Our geometric stability or agreement cannot be substituted for its task
accuracy—or for independent literary judgments. Proceedings metadata and
methods were checked; no code was executed.

**Mostafa Abdou, Vinit Ravishankar, Artur Kulmizev and Anders Søgaard (2022),
[Word Order Does Matter and Shuffled Language Models Know It](https://aclanthology.org/2022.acl-long.476/).**
*ACL*, pp.6907–6919; DOI 10.18653/v1/2022.acl-long.476. The
[paper](https://aclanthology.org/anthology-files/pdf/acl/2022.acl-long.476.pdf),
particularly §§4–6, demonstrates that shuffling before versus after subword
segmentation leaves different information available. Sentence-length and
unigram dependencies can retain cues even after subword shuffling. This is
direct precedent for distinguishing word and token controls, rather than
treating either as removal of all order-related information. Their altered
pretraining experiments differ from our frozen-model inference interventions.
Our retained endpoints, punctuation and line boundaries further constrain
what disruption means. Exact inventories and fixed padding do not by
themselves remove corpus-level lexical or length relationships. Proceedings
and relevant full-text sections were checked; no implementation was reproduced.

**Minyoung Huh, Brian Cheung, Tongzhou Wang and Phillip Isola (2024),
[The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987).**
The [full text](https://arxiv.org/html/2405.07987v5), §2 and Appendices A–B,
establishes the broader alignment question and explicitly discusses metric
dependence. Its principal experiments use mutual nearest-neighbor agreement;
the appendix shows that changing local-versus-global emphasis changes the
observed alignment trend. This is not the same quantity as Spearman correlation
over all poem-pair distances. Our correlations cannot inherit their
competence interpretation merely because both are called alignment. Their
reported convergence is not a demonstrated prediction of increasing
between-model RSA under order disruption. This previously cited paper was
rechecked at method level; its [author code](https://github.com/minyoungg/platonic-rep)
was not executed in this pass.

**Hong-Han Wang, Yuntao Wang and Hu Ding (2026), [The Alignment Illusion in
Multimodal Large Language Models](https://arxiv.org/abs/2609.30210).**
The [full text](https://arxiv.org/html/2609.30210v1), §3 and Appendix J,
reports that replacing projected visual tokens with Gaussian noise can leave
alignment scores high or increase them despite poorer task accuracy. The
authors investigate shared decoder projections as an explanation and compare
CKA, SVCCA, MIR and principal-angle measures. This directly precedes the broad
warning that stronger geometric alignment under corruption need not mean
better content preservation. The setting is visual/text streams **inside one
shared decoder**, not poem RDMs from separately parameterized text models.
Their proposed shared-weight mechanism therefore cannot be assigned to our
between-model comparison without additional evidence. The September 24 arXiv
version was checked; its metadata reports NeurIPS 2026 acceptance, but that
venue status was not independently checked against proceedings. A released
implementation was not verified.

## Implications for the present study

Separate three questions: preservation of each model's original geometry,
agreement between models after perturbation, and performance on an independent
literary task. None implies either of the others. A rising alignment score
also does not by itself establish representational collapse: rank agreement
does not measure the absolute spread, effective dimensionality or retained
task information of the representations. These are methodological deductions,
not new results from v4.

The present design's concrete distinctions are a fixed new sonnet population,
five predetermined perturbations, identical BF16/FP32 input pairs, separate
raw/final-normalized states, multiple readouts and retained unfavorable or
undefined comparisons. Those controls improve the test's interpretability;
they are not a priority claim. Muse's missing final runtime records also
remain a limit on strict confirmation, irrespective of any numerical outcome;
see [the recovery decision](confirmation-v4-interrupted-finalization.md).

This pass did not verify an exact prior experiment combining the same corpus
unit, pooled-RDM estimand, separate model families and order controls. That
limited search outcome does not establish absence. In particular, the five
papers above already cover much of the motivation and several key cautions.
No claim of first discovery or verified novelty follows.

## Search and evidence boundary

Eight targeted queries covered transformer representations and word-order
perturbations, representational similarity under shuffling, cross-model
alignment under corrupted or random inputs, and exact titles for the two ACL
papers. Primary arXiv HTML/PDF and ACL proceedings were used for the claims
above; search snippets, aggregators and forum commentary were not treated as
paper evidence. The initial ACL PDF endpoint timed out and the corresponding
Anthology file endpoint was read successfully. No cited experiment was rerun,
no code was downloaded for execution, and this pass does not exhaust citation
chains, preprints, author notebooks or unindexed work.
