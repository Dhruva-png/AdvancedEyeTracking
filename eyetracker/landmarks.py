"""MediaPipe Face Mesh landmark index groups used for eye/iris geometry.

Two coordinate references are used:

- The 6-point EAR contour drives blink detection (unchanged).
- The corner/lid anchors drive the head-invariant gaze feature: measuring the
  iris relative to the eye *corners* (and normalizing by the corner-to-corner
  distance) cancels out camera distance and head roll, which a raw bounding
  box around jittery contour points does not.

Corner indices are named by their side *in the image*, so the same left->right
axis convention holds for both eyes and their gaze features can be averaged.
"""

# 6-point eye contour, ordered to match the classic EAR formula (Soukupova & Cech).
LEFT_EYE = [33, 133, 160, 159, 158, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 373]

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Eye corner (canthus) + lid anchors for the head-invariant gaze feature.
# *_LEFT_CORNER / *_RIGHT_CORNER refer to image-left / image-right.
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133
LEFT_EYE_TOP_LID = 159
LEFT_EYE_BOTTOM_LID = 145

RIGHT_EYE_LEFT_CORNER = 362
RIGHT_EYE_RIGHT_CORNER = 263
RIGHT_EYE_TOP_LID = 386
RIGHT_EYE_BOTTOM_LID = 374
