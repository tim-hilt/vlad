# Vector of Locally Aggregated Descriptors (VLAD)

---
## Description

This repository is an implementation of VLAD, which was originally formulated by Hervé Jégou in [1].
The implementation is part of my bachelors-thesis termed "Computer Vision and Machine Learning for
marker-free product identification".

VLAD is an algorithm that allows a user to aggregate local descriptors into a compact, global representation.
It's derived from the Bag of Features approach [2] and related to the Fisher vector [3].

This repository is a WIP and should not be considered production-ready. I took a bit of inspiration from
[Jorjassos implementation](https://github.com/jorjasso/VLAD) and tried to make it better. Improved versions 
of the original formulation are also implemented. I took them from [1, 4, 5] and inserted references in the code.

## Dependencies

- Numpy
- Scikit-Learn

## Install-Instructions

**TODO**

## Usage

**TODO**

## Documentation

Documentation can be found at [...] **TODO**

## Roadmap

| **Task**                                       | **Status** |
|------------------------------------------------|------------|
| Original formulation (SSR, L2) [1]             | Done       |
| Use RootSIFT-descriptors [6]                   | TODO       |
| Try with more descriptors                      | TODO       |
| Try with dense descriptors                     | TODO       |
| Intra-Normalization [4]                        | Done       |
| Residual-Normalization (RN) [5]                | Done       |
| Local Coordinate System (LCS) [5]              | TODO       |
| Dimensionality-Reduction [7,8]                 | TODO       |
| Quantization [9]                               | TODO       |
| Generalization using multiple vocabularies [7] | TODO       |
| Make documentation                             | TODO       |
| Include Tests                                  | TODO       |
| Include Install-Instructions                   | TODO       |
| Include Usage-Examples                         | TODO       |
| Publish to PyPi                                | TODO       |

## References

[1]: Jégou, H., Douze, M., Schmid, C., & Pérez, P. (2010, June). Aggregating
local descriptors into a compact image representation. In 2010 IEEE computer
society conference on computer vision and pattern recognition (pp. 3304-3311). IEEE.

[2]: Sivic, J., & Zisserman, A. (2003, October). Video Google: A text retrieval 
approach to object matching in videos. In null (p. 1470). IEEE.

[3]: Perronnin, F., Sánchez, J., & Mensink, T. (2010, September). Improving the fisher kernel for
large-scale image classification. In European conference on computer vision (pp. 143-156). Springer,
Berlin, Heidelberg.

[4]: Arandjelovic, R., & Zisserman, A. (2013). All about VLAD. In Proceedings of theIEEE conference on 
Computer Vision and Pattern Recognition (pp. 1578-1585).

[5]: Delhumeau, J., Gosselin, P. H., Jégou, H., & Pérez, P. (2013, October).Revisiting the VLAD image
representation. In Proceedings of the 21st ACM international conference on Multimedia (pp. 653-656).

[6]: Arandjelović, R., & Zisserman, A. (2012, June). Three things everyone should know to improve object
retrieval. In 2012 IEEE Conference on Computer Vision and Pattern Recognition (pp. 2911-2918). IEEE.

[7]: Jégou, H., & Chum, O. (2012, October). Negative evidences and co-occurences in image retrieval: The
benefit of PCA and whitening. In European conference on computer vision (pp. 774-787). Springer, Berlin,
Heidelberg.

[8]: Jegou, H., Perronnin, F., Douze, M., Sánchez, J., Perez, P., & Schmid, C. (2011). Aggregating local
image descriptors into compact codes. IEEE transactions on pattern analysis and machine intelligence,
34(9), 1704-1716.

[9]: Jegou, H., Douze, M., & Schmid, C. (2010). Product quantization for nearest neighbor search. IEEE
transactions on pattern analysis and machine intelligence, 33(1), 117-128.