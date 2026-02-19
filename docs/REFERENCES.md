# References and Citations

This bibliography is the canonical citation list for `esl` algorithms, metrics, and documentation.

## Citation Policy

- All algorithmic implementations should cite the primary paper, standard, or authoritative technical note.
- If implementation details are inspired by open-source projects, both the project and its license must be cited in `/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md`.
- When in doubt, cite both the standards document and one high-quality implementation reference.

## Standards

- [S1] [ITU-R BS.1770-4: Algorithms to measure audio programme loudness and true-peak audio level](https://www.itu.int/rec/R-REC-BS.1770/en)
- [S2] [EBU Tech 3341: Loudness Metering: EBU Mode](https://tech.ebu.ch/publications/tech3341)
- [S3] [ISO 3382-1:2009 — Acoustics — Measurement of room acoustic parameters — Part 1: Performance spaces](https://www.iso.org/standard/40979.html)
- [S4] [ISO 3382-2:2008 — Acoustics — Measurement of room acoustic parameters — Part 2: Reverberation time in ordinary rooms](https://www.iso.org/standard/36205.html)
- [S5] [IEC 61672-1: Electroacoustics — Sound level meters](https://webstore.iec.ch/publication/5708)

## Foundational DSP and Signal Analysis

- [D1] [Allen, J. B., & Rabiner, L. R. (1977). A unified approach to short-time Fourier analysis and synthesis](https://ieeexplore.ieee.org/document/1455106)
- [D2] [Oppenheim, A. V., & Schafer, R. W. — Discrete-Time Signal Processing](https://www.pearson.com/en-us/subject-catalog/p/discrete-time-signal-processing/P200000003480/9780134440147)
- [D3] [Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420)
- [D4] [Serra, X. (1989). A system for sound analysis/transformation/synthesis based on a deterministic plus stochastic decomposition](https://www.upf.edu/web/mtg/sms-tools)
- [D5] [Harris, F. J. (1978). On the use of windows for harmonic analysis with the DFT](https://ieeexplore.ieee.org/document/1455106)

## Novelty, Segmentation, Similarity

- [N1] [Foote, J. (2000). Automatic audio segmentation using a measure of audio novelty](https://dl.acm.org/doi/10.1145/336597.336612)
- [N2] [Müller, M. — Fundamentals of Music Processing (book + resources)](https://www.audiolabs-erlangen.de/resources/MIR/FMP)
- [N3] [FMP Notebook: Novelty-based segmentation](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html)
- [N4] [Dixon, S. (2006). Onset detection revisited](https://www.dafx.de/paper-archive/2006/papers/p_133.pdf)

## Room Acoustics and Architectural Metrics

- [A1] [Schroeder, M. R. (1965). New method of measuring reverberation time](https://asa.scitation.org/doi/10.1121/1.1909343)
- [A2] [Barron, M. (2010). Auditorium Acoustics and Architectural Design](https://www.routledge.com/Auditorium-Acoustics-and-Architectural-Design/Barron/p/book/9780419191603)
- [A3] [Kuttruff, H. — Room Acoustics](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781138902129)
- [A4] [IEC 60268-16 / STI intelligibility framework](https://webstore.iec.ch/publication/6022)

## Spatial Audio and TDOA

- [P1] [Knapp, C., & Carter, G. (1976). The generalized correlation method for estimation of time delay](https://ieeexplore.ieee.org/document/1162830)
- [P2] [Blauert, J. — Spatial Hearing](https://mitpress.mit.edu/9780262523545/spatial-hearing/)
- [P3] [Merimaa, J., & Pulkki, V. (2005). Spatial impulse response rendering I: Analysis and synthesis](https://pubmed.ncbi.nlm.nih.gov/16334689/)
- [P4] [Farina, A. (2000). Simultaneous measurement of impulse response and distortion with a swept-sine technique](https://www.researchgate.net/publication/2456363)

## Ecoacoustics Indices

- [E1] [Pieretti, N., Farina, A., & Morri, D. (2011). A new methodology to infer the singing activity of an avian community: The Acoustic Complexity Index (ACI)](https://link.springer.com/article/10.1007/s10336-011-0680-8)
- [E2] [Kasten, E. P., et al. (2012). The remote environmental assessment laboratory's acoustic library: An archive for studying soundscape ecology](https://www.sciencedirect.com/science/article/pii/S1574954112000204)
- [E3] [Sueur, J., et al. (2008). Rapid acoustic survey for biodiversity appraisal](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2664.2008.01553.x)
- [E4] [Villanueva-Rivera, L. J., et al. Soundscape Ecology resources](https://soundscapeecology.org/)
- [E5] [Towsey, M., et al. (2014). Visualization of long-duration acoustic recordings of the environment](https://www.mdpi.com/1424-8220/14/6/10339)

## Anomaly Detection and ML

- [M1] [Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest](https://ieeexplore.ieee.org/document/4781136)
- [M2] [Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution](https://www.mitpressjournals.org/doi/10.1162/089976601750264965)
- [M3] [Pimentel, M. A. F., et al. (2014). A review of novelty detection](https://www.sciencedirect.com/science/article/pii/S0165168414000787)
- [M4] [Goodfellow, I., et al. (2016). Deep Learning](https://www.deeplearningbook.org/)

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
- Update `/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md` when code is adapted or re-implemented from an existing open-source project.

## Citation Workflow

```mermaid
flowchart LR
    A["New Algorithm"] --> B["Find Primary Paper or Standard"]
    B --> C["Add Code Comment Citation"]
    C --> D["Add Entry in REFERENCES.md"]
    D --> E["Add Attribution Note if OSS-Inspired"]
    E --> F["Review Before Merge"]
```
