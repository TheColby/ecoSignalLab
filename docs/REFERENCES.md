# References and Citations

Quick links: [Docs Index](INDEX.md) | [Getting Started](GETTING_STARTED.md) | [Task Recipes](TASK_RECIPES.md) | [Troubleshooting](TROUBLESHOOTING.md) | [Schema](SCHEMA.md) | [Metrics](METRICS_REFERENCE.md) | [References](REFERENCES.md)

This bibliography is the canonical citation list for `esl` algorithms, metrics, and documentation.

## Citation Policy

- All algorithmic implementations should cite the primary paper, standard, or authoritative technical note.
- If implementation details are inspired by open-source projects, both the project and its license must be cited in `docs/ATTRIBUTION.md`.
- When in doubt, cite both the standards document and one high-quality implementation reference.
- Novelty/segmentation citations should also be reflected in workflow docs such as [`docs/MOMENTS_EXTRACTION.md`](MOMENTS_EXTRACTION.md).

## Standards

- [S1] [ITU-R BS.1770-4: Algorithms to measure audio programme loudness and true-peak audio level](https://www.itu.int/rec/R-REC-BS.1770/en)
- [S2] [EBU Tech 3341: Loudness Metering: EBU Mode](https://tech.ebu.ch/publications/tech3341)
- [S3] [ISO 3382-1:2009 — Acoustics — Measurement of room acoustic parameters — Part 1: Performance spaces](https://www.iso.org/standard/40979.html)
- [S4] [ISO 3382-2:2008 — Acoustics — Measurement of room acoustic parameters — Part 2: Reverberation time in ordinary rooms](https://www.iso.org/standard/36205.html)
- [S5] [IEC 61672-1: Electroacoustics — Sound level meters](https://webstore.iec.ch/publication/5708)
- [S6] [EBU Tech 3306: RF64 — An extended File Format for Audio](https://tech.ebu.ch/publications/tech3306)

## Foundational DSP and Signal Analysis

- [D1] [Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time Fourier analysis and synthesis](https://ieeexplore.ieee.org/document/1455106)
- [D2] [Oppenheim, A. V., & Schafer, R. W. — Discrete-Time Signal Processing](https://www.pearson.com/en-us/subject-catalog/p/discrete-time-signal-processing/P200000003480/9780134440147)
- [D3] [Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420)
- [D4] [Serra, X. (1989). A system for sound analysis/transformation/synthesis based on a deterministic plus stochastic decomposition](https://www.upf.edu/web/mtg/sms-tools)
- [D5] [Harris, F. J. (1978). On the use of windows for harmonic analysis with the DFT](https://ieeexplore.ieee.org/document/1455106)
- [D6] [Smith, J. O. (2011). Spectral Audio Signal Processing](https://ccrma.stanford.edu/~jos/sasp/)
- [D7] [Klapuri, A., & Davy, M. (Eds.). (2006). Signal Processing Methods for Music Transcription](https://link.springer.com/book/10.1007/0-387-32845-9)
- [D8] [Virtanen, T., Plumbley, M. D., & Ellis, D. (Eds.). (2018). Computational Analysis of Sound Scenes and Events](https://link.springer.com/book/10.1007/978-3-319-63450-0)
- [D9] [Bregman, A. S. (1990). Auditory Scene Analysis](https://mitpress.mit.edu/9780262521956/auditory-scene-analysis/)
- [D10] [Wang, D., & Brown, G. J. (Eds.). (2006). Computational Auditory Scene Analysis](https://ieeexplore.ieee.org/book/5264093)
- [D11] [McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python](https://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf)
- [D12] [Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python](https://doi.org/10.1038/s41592-019-0686-2)

## Novelty, Segmentation, Similarity

- [N1] [Foote, J. (2000). Automatic audio segmentation using a measure of audio novelty](https://dl.acm.org/doi/10.1145/336597.336612)
- [N2] [Müller, M. — Fundamentals of Music Processing (book + resources)](https://www.audiolabs-erlangen.de/resources/MIR/FMP)
- [N3] [FMP Notebook: Novelty-based segmentation](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html)
- [N4] [Dixon, S. (2006). Onset detection revisited](https://www.dafx.de/paper-archive/2006/papers/p_133.pdf)
- [N5] [Bello, J. P., et al. (2005). A tutorial on onset detection in music signals](https://doi.org/10.1109/TSA.2005.851998)
- [N6] [Serrà, J., Serra, X., & Andrzejak, R. G. (2009). Cross recurrence quantification for cover song identification](https://doi.org/10.1016/j.newar.2008.07.020)
- [N7] [Casey, M. A., et al. (2008). Content-based music information retrieval: Current directions and future challenges](https://doi.org/10.1109/JPROC.2008.916370)
- [N8] [Müller, M., Kurth, F., & Clausen, M. (2007). Audio matching via chroma-based statistical features](https://doi.org/10.1007/s00530-006-0045-3)

## Room Acoustics and Architectural Metrics

- [A1] [Schroeder, M. R. (1965). New method of measuring reverberation time](https://asa.scitation.org/doi/10.1121/1.1909343)
- [A2] [Barron, M. (2010). Auditorium Acoustics and Architectural Design](https://www.routledge.com/Auditorium-Acoustics-and-Architectural-Design/Barron/p/book/9780419191603)
- [A3] [Kuttruff, H. — Room Acoustics](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781138902129)
- [A4] [IEC 60268-16 / STI intelligibility framework](https://webstore.iec.ch/publication/6022)
- [A5] [Kuttruff, H. (1991). A simple iteration scheme for the computation of decay constants in rooms](https://doi.org/10.1121/1.401051)
- [A6] [Lundeby, A., Vigran, T. E., Bietz, H., & Vorländer, M. (1995). Uncertainties of measurements in room acoustics](https://doi.org/10.1121/1.411923)
- [A7] [Vorlander, M. (2008). Auralization: Fundamentals of Acoustics, Modelling, Simulation, Algorithms and Acoustic Virtual Reality](https://link.springer.com/book/10.1007/978-3-540-48830-9)
- [A8] [Long, M. (2014). Architectural Acoustics](https://www.elsevier.com/books/architectural-acoustics/long/978-0-12-398258-2)

## Spatial Audio and TDOA

- [P1] [Knapp, C., & Carter, G. (1976). The generalized correlation method for estimation of time delay](https://ieeexplore.ieee.org/document/1162830)
- [P2] [Blauert, J. — Spatial Hearing](https://mitpress.mit.edu/9780262523545/spatial-hearing/)
- [P3] [Merimaa, J., & Pulkki, V. (2005). Spatial impulse response rendering I: Analysis and synthesis](https://pubmed.ncbi.nlm.nih.gov/16334689/)
- [P4] [Farina, A. (2000). Simultaneous measurement of impulse response and distortion with a swept-sine technique](https://www.researchgate.net/publication/2456363)
- [P5] [Gerzon, M. A. (1973). Periphony: With-height sound reproduction](https://doi.org/10.1121/1.1914257)
- [P6] [Pulkki, V. (1997). Virtual sound source positioning using vector base amplitude panning](https://doi.org/10.1109/ASPAA.1997.608784)
- [P7] [Zotter, F., & Frank, M. (2019). Ambisonics: A Practical 3D Audio Theory for Recording, Studio Production, Sound Reinforcement, and Virtual Reality](https://link.springer.com/book/10.1007/978-3-030-17207-7)
- [P8] [Daniel, J. (2000). Représentation de champs acoustiques, application à la transmission et à la reproduction de scènes sonores complexes](https://theses.hal.science/tel-00005603)

## Acoustic Forensics and Surveillance (Rob Maher and Collaborators)

- [RMA1] [Maher, R. C. (2009). Audio forensic examination: authenticity, enhancement, and interpretation. IEEE Signal Processing Magazine, 26(2), 84-94](https://www.montana.edu/rmaher/publications/maher_ieeespmag_0309_84-94.pdf)
- [RMA2] [Chen, Z., & Maher, R. C. (2006). Semi-automatic classification of bird vocalizations using spectral peak tracks. Journal of the Acoustical Society of America, 120(5), 2974-2984](https://www.montana.edu/rmaher/publications/chen_maher_jasa_0611_2974-2984.pdf)
- [RMA3] [Maher, R. C., & Studniarz, J. (2012). Automatic search and classification of sound sources in long-term surveillance recordings. Proc. AES 46th Conference: Audio Forensics—Recording, Recovery, Analysis, and Interpretation](https://www.montana.edu/rmaher/publications/maher_aesconf_0612_1-4.pdf)
- [RMA4] [Maher, R. C., & Hoerr, E. R. (2018). Audio forensic gunshot analysis and multilateration. Proc. AES 145th Convention, Paper 10100](https://www.montana.edu/rmaher/publications/maher_aes_1018_10100.pdf)
- [RMA5] [Maher, R. C., & Hoerr, E. R. (2019). Forensic comparison of simultaneous recordings of gunshots from a crime scene. Proc. AES 147th Convention, Preprint 10281](https://www.montana.edu/rmaher/publications/maher_aes_1019_10281.pdf)

## Ecoacoustics Indices

- [E1] [Pieretti, N., Farina, A., & Morri, D. (2011). A new methodology to infer the singing activity of an avian community: The Acoustic Complexity Index (ACI)](https://link.springer.com/article/10.1007/s10336-011-0680-8)
- [E2] [Kasten, E. P., et al. (2012). The remote environmental assessment laboratory's acoustic library: An archive for studying soundscape ecology](https://www.sciencedirect.com/science/article/pii/S1574954112000204)
- [E3] [Sueur, J., et al. (2008). Rapid acoustic survey for biodiversity appraisal](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2664.2008.01553.x)
- [E4] [Villanueva-Rivera, L. J., et al. Soundscape Ecology resources](https://soundscapeecology.org/)
- [E5] [Towsey, M., et al. (2014). Visualization of long-duration acoustic recordings of the environment](https://www.mdpi.com/1424-8220/14/6/10339)
- [E6] [Pijanowski, B. C., et al. (2011). Soundscape ecology: The science of sound in the landscape](https://doi.org/10.1093/biosci/biq060)
- [E7] [Buxton, R. T., et al. (2018). A synthesis of health benefits of natural sounds and their distribution in national parks](https://doi.org/10.1073/pnas.1717419115)
- [E8] [Bradfer-Lawrence, T., et al. (2019). Guidelines for the use of acoustic indices in environmental research](https://doi.org/10.1111/2041-210X.13254)
- [E9] [Alcocer, I., et al. (2022). Acoustic indices as proxies for biodiversity: a meta-analysis](https://doi.org/10.1111/brv.12890)
- [E10] [Gasc, A., et al. (2013). Assessing biodiversity with sound: Do acoustic diversity indices reflect phylogenetic diversity of bird communities?](https://doi.org/10.1111/2041-210X.12097)
- [E11] [Fuller, S., et al. (2015). Connecting soundscape to landscape: Which acoustic index best describes landscape configuration?](https://doi.org/10.1111/2041-210X.12397)

## Long-Term Audio Analysis, Calmness, and Diversity

- [LT1] [ISO 12913-1:2014 — Acoustics — Soundscape — Part 1: Definition and conceptual framework](https://www.iso.org/standard/52161.html)
- [LT2] [ISO/TS 12913-2:2018 — Acoustics — Soundscape — Part 2: Data collection and reporting requirements](https://www.iso.org/standard/75267.html)
- [LT3] [ISO/TS 12913-3:2025 — Acoustics — Soundscape — Part 3: Data analysis](https://www.iso.org/standard/86955.html)
- [LT4] [Axelsson, Ö., Nilsson, M. E., & Berglund, B. (2010). A principal components model of soundscape perception. *JASA*, 128(5), 2836-2846](https://doi.org/10.1121/1.3493436)
- [LT5] [Aletta, F., Kang, J., & Axelsson, Ö. (2016). Soundscape descriptors and a conceptual framework for developing predictive soundscape models. *Landscape and Urban Planning*, 149, 65-74](https://doi.org/10.1016/j.landurbplan.2016.02.001)
- [LT6] [Villanueva-Rivera, L. J., Pijanowski, B. C., Doucette, J., & Pekin, B. (2011). A primer of acoustic analysis for landscape ecologists. *Landscape Ecology*, 26, 1233-1246](https://doi.org/10.1007/s10980-011-9636-9)
- [LT7] [Sueur, J., Farina, A., Gasc, A., Pieretti, N., & Pavoine, S. (2014). Acoustic indices for biodiversity assessment and landscape investigation. *Acta Acustica united with Acustica*, 100(4), 772-781](https://doi.org/10.3813/AAA.918757)
- [LT8] [Towsey, M., Wimmer, J., Williamson, I., & Roe, P. (2014). The use of acoustic indices to determine avian species richness in audio-recordings of the environment. *Ecological Informatics*, 21, 110-119](https://doi.org/10.1016/j.ecoinf.2013.11.007)
- [LT9] [Phillips, Y. F., Towsey, M., & Roe, P. (2018). Revealing the ecological content of long-duration audio-recordings of the environment through clustering and visualisation. *PLOS ONE*, 13(3), e0193345](https://doi.org/10.1371/journal.pone.0193345)
- [LT10] [Merchant, N. D., Fristrup, K. M., Johnson, M. P., Tyack, P. L., Witt, M. J., Blondel, P., & Parks, S. E. (2015). Measuring acoustic habitats. *Methods in Ecology and Evolution*, 6(3), 257-265](https://doi.org/10.1111/2041-210X.12330)

## Everglades Monitoring Systems

- [EV1] [Dickey, S. (2011). Near-Real-Time Internet Streaming of Audio from the Florida Everglades](https://fiord.com/images/embedded_syst/compulab/GladeBox.pdf) (GladesBox v2 paper; includes reference to the pilot deployment)
- [EV2] Leider, C., Mann, D., & Dickinson, D. (2010). *Wireless multisensor monitoring of the Florida Everglades: A pilot project*. In *Audio Engineering Society Convention 129*. (Cited in [EV1](https://fiord.com/images/embedded_syst/compulab/GladeBox.pdf), References [1])

## NSF-Supported Ecology and Ecoacoustics Papers

### Ecoacoustics and Acoustic Monitoring

- [NSE1] [Oliver, R. Y., et al. (2018). Eavesdropping on the Arctic: Automated bioacoustics reveal dynamics in songbird breeding phenology](https://doi.org/10.1126/sciadv.aaq1084) (NSF support noted in acknowledgments: ARC 0908444, ARC 0908602, ARC 0909133, GRFP awards)
- [NSE2] [Symes, L. B., et al. (2022). Analytical approaches for evaluating passive acoustic monitoring data: A case study of avian vocalizations](https://doi.org/10.1002/ece3.8797) (NSF Long-Term Ecological Research award 1637685)
- [NSE3] [Gomes, D. G. E., et al. (2021). Phantom rivers filter birds and bats by acoustic niche](https://doi.org/10.1038/s41467-021-22390-y) (NSF support noted in acknowledgments: GRFP 2018268606, DEB 1556192, DEB 1556177, IOS 1920936)
- [NSE4] [Ryan, J. P., et al. (2022). Oceanic giants dance to atmospheric rhythms: Ephemeral wind-driven resource tracking by blue whales](https://doi.org/10.1111/ele.14116) (NSF-supported acoustic infrastructure and IOS-1656676 support noted in acknowledgments)
- [NSE5] [Miller-Struttmann, N. E., et al. (2017). Flight of the bumble bee: Buzzes predict pollination services](https://doi.org/10.1371/journal.pone.0179273) (NSF awards and NSF-supported LTER access noted in funding statement)
- [NSE6] [Lapp, S., et al. (2023). OpenSoundscape: An open-source bioacoustics analysis package for Python](https://doi.org/10.1111/2041-210X.14196) (NSF support in [NSF PAR record](https://par.nsf.gov/biblio/10441480): 1935507, 2120084)
- [NSE7] [Tolkova, I., & Klinck, H. (2022). Source separation with an acoustic vector sensor for terrestrial bioacoustics](https://doi.org/10.1121/10.0013505) (NSF support in [NSF PAR record](https://par.nsf.gov/biblio/10470622-source-separation-acoustic-vector-sensor-terrestrial-bioacoustics): 1764269)
- [NSE8] [Myers, H. J., et al. (2021). Passive acoustic monitoring of killer whales (Orcinus orca) reveals year-round distribution and residency patterns in the Gulf of Alaska](https://doi.org/10.1038/s41598-021-99668-0) (NSF support in [NSF PAR record](https://par.nsf.gov/biblio/10336369): 1757348)

## Anomaly Detection and ML

- [M1] [Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest](https://ieeexplore.ieee.org/document/4781136)
- [M2] [Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution](https://www.mitpressjournals.org/doi/10.1162/089976601750264965)
- [M3] [Pimentel, M. A. F., et al. (2014). A review of novelty detection](https://www.sciencedirect.com/science/article/pii/S0165168414000787)
- [M4] [Goodfellow, I., et al. (2016). Deep Learning](https://www.deeplearningbook.org/)
- [M5] [Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey](https://doi.org/10.1145/1541880.1541882)
- [M6] [Ruff, L., et al. (2021). Unifying review of deep and shallow anomaly detection](https://doi.org/10.1109/PROC.2021.3052449)
- [M7] [Gemmeke, J. F., et al. (2017). Audio Set: An ontology and human-labeled dataset for audio events](https://doi.org/10.1109/ICASSP.2017.7952261)
- [M8] [Hershey, S., et al. (2017). CNN architectures for large-scale audio classification](https://doi.org/10.1109/ICASSP.2017.7952132)
- [M9] [Kong, Q., et al. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition](https://doi.org/10.1109/TASLP.2020.3030497)
- [M10] [Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations](https://arxiv.org/abs/2006.11477)
- [M11] [Hsu, W.-N., et al. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units](https://doi.org/10.1109/TASLP.2021.3122291)
- [M12] [Gong, Y., Chung, Y.-A., & Glass, J. (2021). AST: Audio Spectrogram Transformer](https://doi.org/10.48550/arXiv.2104.01778)
- [M13] [Mesaros, A., et al. (2021). Sound event detection in real life: DCASE 2020 challenge](https://doi.org/10.1109/TASLP.2021.3065333)

## Measurement, Calibration, and Reproducibility

- [C1] [ISO 1996-1:2016 - Acoustics - Description, measurement and assessment of environmental noise](https://www.iso.org/standard/66937.html)
- [C2] [ISO 1996-2:2017 - Acoustics - Description, measurement and assessment of environmental noise](https://www.iso.org/standard/66939.html)
- [C3] [IEC 60942:2017 - Electroacoustics - Sound calibrators](https://webstore.iec.ch/publication/29068)
- [C4] [AES17-2020 - AES standard for digital audio engineering - Measurement of digital audio equipment](https://www.aes.org/standards/?id=standards-documents)
- [C5] [Sturm, B. L. (2014). The state of the art ten years after a state of the art: Future research in music information retrieval](https://doi.org/10.1016/j.jnca.2013.11.008)
- [C6] [Sandve, G. K., et al. (2013). Ten simple rules for reproducible computational research](https://doi.org/10.1371/journal.pcbi.1003285)
- [C7] [Peng, R. D. (2011). Reproducible research in computational science](https://doi.org/10.1126/science.1213847)
- [C8] [Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship](https://doi.org/10.1038/sdata.2016.18)

## Datasets, Evaluation, and Open Science

- [O1] [Salamon, J., Jacoby, C., & Bello, J. P. (2014). A dataset and taxonomy for urban sound research](https://doi.org/10.1145/2647868.2655045)
- [O2] [Fonseca, E., et al. (2022). FSD50K: An open dataset of human-labeled sound events](https://doi.org/10.1109/TASLP.2021.3133208)
- [O3] [Stowell, D., et al. (2019). Automatic acoustic detection of birds through deep learning: The first Bird Audio Detection challenge](https://doi.org/10.1111/2041-210X.13103)
- [O4] [Lapp, S., et al. (2023). OpenSoundscape: An open-source bioacoustics analysis package for Python](https://doi.org/10.1111/2041-210X.14196)

## Libraries and Project Documentation (Implementation Context)

- [L1] [NumPy documentation](https://numpy.org/doc/)
- [L2] [SciPy signal processing documentation](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [L3] [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [L4] [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [L5] [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets)
- [L6] [Librosa documentation](https://librosa.org/doc/latest/index.html)
- [L7] [torchaudio documentation](https://pytorch.org/audio/stable/index.html)
- [L8] [Essentia documentation](https://essentia.upf.edu/documentation/)
- [L9] [pyloudnorm repository](https://github.com/csteinmetz1/pyloudnorm)
- [L10] [scikit-maad documentation](https://scikit-maad.github.io/)
- [L11] [FFmpeg documentation](https://ffmpeg.org/documentation.html)
- [L12] [SoX manual](http://sox.sourceforge.net/sox.html)
- [L13] [Playwright Python docs](https://playwright.dev/python/docs/intro)
- [L14] [Mermaid documentation](https://mermaid.js.org/)

## Guidance for Contributors

- Cite standards when implementing regulatory or measurement-compatible metrics.
- Cite at least one paper for each nontrivial algorithm family.
- Update `docs/ATTRIBUTION.md` when code is adapted or re-implemented from an existing open-source project.

## Citation Workflow

```mermaid
flowchart LR
    A["New Algorithm"] --> B["Find Primary Paper or Standard"]
    B --> C["Add Code Comment Citation"]
    C --> D["Add Entry in REFERENCES.md"]
    D --> E["Add Attribution Note if OSS-Inspired"]
    E --> F["Review Before Merge"]
```
