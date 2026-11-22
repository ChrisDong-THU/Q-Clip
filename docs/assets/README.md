# Paper figures

These images are raster exports of the original vector figures from **Deduplicate Before You Aggregate: Efficient Clip Encoding for Scalable Video Retrieval**, by Jiajun Dong, Yuan Liang, and Gengyuan Liu, ACM Multimedia 2026. [Paper and publication record](https://doi.org/10.1145/3767308.3835060).

| Asset | Original source | Content |
| --- | --- | --- |
| `framework.png` | `figures/framework.pdf` | End-to-end deduplicate-before-aggregate framework for database indexing and query retrieval. |
| `model.png` | `figures/model.pdf` | Q-Clip architecture: learnable queries, iterative attention, and normalized clip descriptors. |

The PNGs preserve the original figure content and are rendered with a maximum dimension of 2,000 pixels for clear GitHub display. The camera-ready paper declares Creative Commons Attribution (CC BY); retain attribution to the paper when reusing its figures. This figure attribution does not establish a license for the repository's source code.

To reproduce an export with Poppler:

```bash
pdftoppm -f 1 -singlefile -scale-to 2000 -png figures/framework.pdf docs/assets/framework
```
