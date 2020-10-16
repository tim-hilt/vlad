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

## Usage

## Documentation

## References

[1]: Jégou, H., Douze, M., Schmid, C., & Pérez, P. (2010, June). Aggregating
local descriptors into a compact image representation. In 2010 IEEE computer
society conference on computer vision and pattern recognition (pp. 3304-3311). IEEE.

[2]: Sivic, J., & Zisserman, A. (2003, October). Video Google: A text retrieval 
approach to object matching in videos. In null (p. 1470). IEEE.

[3]: Perronnin, F., Sánchez, J., & Mensink, T. (2010, September). Improving the fisher kernel for
large-scale image classification. In European conference on computer vision (pp. 143-156). Springer,
Berlin, Heidelberg.

[4] Arandjelovic, R., & Zisserman, A. (2013). All about VLAD. In Proceedings of theIEEE conference on 
Computer Vision and Pattern Recognition (pp. 1578-1585).

[5] Delhumeau, J., Gosselin, P. H., Jégou, H., & Pérez, P. (2013, October).Revisiting the VLAD image
representation. In Proceedings of the 21st ACM international conference on Multimedia (pp. 653-656).