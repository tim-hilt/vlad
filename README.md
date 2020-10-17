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
- Progressbar2
- OpenCV (for the examples)

## Install-Instructions

**TODO**

## Usage

The API is based on the wonderful Scikit-Learn API, which uses the basic notion of `fit`/ `predict`/ `transform`. So far
I didn't have a reason to transform anything, so `fit` and `predict` (along with `predict_proba`) are the functions
comparable to sklearns API.

To include `VLAD` in the current file just write:

```python
from vlad import VLAD
```

On initialization, the number of visual words (`k`) and the norming-scheme are given. Norming is a crucial difference
in between different implementations [1, 4, 5] with [5] containing the preferable one. To instantiate a VLAD-object
write:

```python
vlad = VLAD(k=16, norming="RN")  # Defaults are k=256 and norming="original"
```

After having instantiated the object you can fit the visual vocabulary with:

```python
vlad.fit(X)
```

The `fit`-function also returns the instance (again, sklearn-style), so the following two are equivalent:

```python
vlad = VLAD(k=16, norming="RN")
vlad.fit(X)
# ...
vlad = VLAD(k=16, norming="RN").fit(X)
```

`X` is a tensor of image-descriptors ($m \times d \times n$), where $m$ is the number of descriptors per image, $d$
is the number of dimensions per descriptor and $n$ is the total number of image-descriptors. It's best to use
image-descriptors in euclidean space (Such as SIFT or RootSIFT [6]), rather than in hamming space, as the
KMeans-clustering won't work properly with hamming-descriptors.

Whenever a visual dictionary is fitted, the dictionary is saved to disc and can be loaded manually to bypass training.

To check for an image one can write:

```python
vlad.predict(imdesc)  # imdesc is a (m x d) descriptor-matrix
```

to get the image-index with maximum similarity. Alternatively

```python
vlad.predict_proba(imdesc)
```

can be used to obtain a Numpy-array with all similarity scores. 

## Documentation

Documentation can be found at [...] **TODO**

## Roadmap

| **Task**                                       | **Status** |
|------------------------------------------------|------------|
| Original formulation (SSR, L2) [1]             | Done       |
| Use RootSIFT-descriptors [6]                   | Done       |
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
| Include Usage-Examples                         | Done       |
| Publish to PyPi                                | TODO       |
| Provide example notebooks                      | TODO       |

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