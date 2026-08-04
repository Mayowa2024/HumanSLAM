# HumanSLAM Methodology

## 1. Research motivation

Visual SLAM systems estimate camera motion and recognise previously visited
locations primarily from geometric and local visual features. This is effective
when viewpoint, illumination, weather, motion blur and scene appearance remain
favourable. It can become unreliable when the same place looks different
(perceptual variation), when different places share similar local structure
(perceptual aliasing), or when insufficient reliable features survive. Humans
do not localise using geometry alone: they also use scene context, stable
objects, written language and the temporal order in which observations occur.

HumanSLAM investigates whether those complementary semantic cues can improve
the recovery behaviour of ORB-SLAM3. It is not a replacement visual odometry
system and it does not independently optimise a metric map. ORB-SLAM3 remains
responsible for feature extraction, tracking, keyframe creation, map-point
triangulation, local mapping, pose optimisation, loop correction and Atlas map
fusion. HumanSLAM is an asynchronous semantic place-recognition assistant. It
stores semantic descriptions of ORB-SLAM3 keyframes, ranks plausible previously
observed places, and returns their ORB map and keyframe identifiers. ORB-SLAM3
then performs geometric verification before using a proposal.

This division of responsibility is deliberate. Semantics is robust to some
forms of appearance variation but can confuse visually or functionally similar
places. Geometry is precise when sufficient consistent features exist but can
be brittle under severe variation. HumanSLAM uses semantics to narrow the
search space and geometry to validate the spatial hypothesis.

## 2. Research question and hypothesis

The principal research question is:

> Does hierarchical semantic-contextual retrieval improve relocalisation and
> map-fragment fusion in ORB-SLAM3 under perceptual variation and tracking
> failure, without causing an unacceptable increase in false place matches or
> runtime latency?

The working hypothesis is that scene context provides broad place retrieval,
static-object layout adds structural discrimination, and object-grounded text
provides highly distinctive evidence. Their fusion should improve candidate
recall under appearance variation, while ORB-SLAM3 geometric verification
should suppress unsafe semantic matches.

## 3. System architecture

The system contains two cooperating ROS 2 nodes.

1. **ORB-SLAM3 node.** Processes stereo images, estimates the trajectory,
   constructs keyframes and maps, and publishes semantic frames containing the
   image, frame identifier, reference keyframe identifier, map identifier,
   pose, tracking state and tracking inlier count.
2. **HumanSLAM node.** Converts each semantic frame into a hierarchical
   descriptor, searches its keyframe memory, and publishes a ranked list of
   candidate map/keyframe identifiers with semantic scores.

Two assistance paths consume the same semantic response:

- **Relocalisation assistance:** when ORB-SLAM3 is lost, returned keyframes are
  tested using ORB descriptor matching, PnP pose estimation and pose
  optimisation. A semantic pose is never directly imposed on the tracker.
- **Cross-map fusion assistance:** when tracking loss has caused ORB-SLAM3 to
  create another Atlas map, cross-map semantic candidates are inserted ahead of
  conventional bag-of-words merge candidates. ORB-SLAM3 still requires feature
  matching, Sim(3) estimation and three temporally consistent detections before
  invoking its standard Atlas merge.

The resulting information flow is:

`stereo images → ORB tracking/keyframes → semantic frame → HumanSLAM inference
→ ranked candidates → ORB geometric verification → relocalisation or map merge`

## 4. Semantic keyframe representation

For keyframe \(K_i\), HumanSLAM stores:

\[
K_i = \{i, t_i, m_i, T_i, s_i, O_i\}
\]

where \(i\) is the ORB keyframe identifier, \(t_i\) is the timestamp, \(m_i\)
is the Atlas map identifier, \(T_i\) is the camera pose supplied by ORB-SLAM3,
\(s_i\) is the scene record and \(O_i\) is the set of selected static objects.
An object contains its class, detector confidence, normalised centroid,
normalised area and any OCR strings detected inside its region.

HumanSLAM stores the ORB identifiers rather than maintaining a second geometric
map. This allows a retrieved semantic record to refer back to the authoritative
keyframe and map held by ORB-SLAM3.

## 5. Scene-context layer

A Places365 ResNet-50 model provides both a scene classification output and a
512-dimensional penultimate-layer embedding. The model is exported to
TensorRT, reducing inference overhead and avoiding a Python deep-learning
forward pass at runtime. Input images are resized and normalised using the
training convention before inference. The embedding is L2-normalised for
cosine comparison.

To add temporal context, the current scene and two preceding scenes are
compared using recency weights \(0.5\), \(0.3\) and \(0.2\):

\[
S_{\text{scene}}(Q,C)=
\sum_{j=0}^{k-1}\alpha_j
\max(0,\hat e^Q_j \cdot \hat e^C_j),
\qquad
\sum_j\alpha_j=1.
\]

The sequence helps distinguish individual frames that appear similar but occur
in different contextual transitions. Class-label confidence is retained for
interpretability, but it does not multiply the retrieval embedding similarity:
classification certainty and embedding suitability are different quantities,
and multiplying confidences suppressed valid identical-place matches.

For additional context, the top three Places365 probabilities are aggregated
by their official class indices into road-localisation groups: urban road,
residential, commercial, parking, major transport, industrial, rural road,
natural and restricted/special. Unmapped classes form an uncertain `other`
group. A compatibility value of 1.0 is used for the same informative group,
0.6 for related groups, 0.3 when `other` is involved and 0 for incompatible
groups. With category weight \(\lambda_c\), the per-frame embedding similarity
is modulated as

\[
\tilde S_e = S_e\frac{1+\lambda_c C_{cat}}{1+\lambda_c}.
\]

The default \(\lambda_c=0.10\) makes category evidence deliberately weak:
matching context preserves the embedding score, while a mismatch can reduce it
by at most about 9.1 percent. It cannot create a match without vector
similarity. `use_scene_category` independently disables this term for ablation.

The scene layer also performs the inexpensive global preselection. Only its top
\(K\) candidates are passed to the more detailed object/text fusion, controlling
latency as the map grows.

## 6. Static-object layout layer

YOLO segmentation detects objects. Only configured stable classes are retained;
dynamic road users should not become persistent place anchors. When a selected
class has a reliable mask, its centroid and area are computed from that mask.
A bounding-box fallback preserves approximate layout evidence if a model emits
boxes without usable masks.

For a query object \(o_q\) and candidate object \(o_c\), normalised layout
distance is:

\[
d_{\text{mask}} =
\sqrt{(x_q-x_c)^2+(y_q-y_c)^2+
\lambda_a(a_q-a_c)^2}.
\]

It is converted into a spatial similarity using a Gaussian kernel:

\[
S_{\text{geom}} =
\exp\left(-\frac{d_{\text{mask}}^2}{2\sigma_m^2}\right).
\]

Objects of different classes receive zero correspondence score. A same-class
pair is weighted by both detector confidences:

\[
M_{\text{obj}} =
\mathbb{1}[c_q=c_c]\,p_qp_cS_{\text{geom}}.
\]

Each query object selects its best candidate correspondence and the resulting
scores are averaged. This directional matching penalises query landmarks that
cannot be explained by the candidate while avoiding an expensive global
assignment algorithm.

## 7. Object-grounded text layer

OCR is applied to selected object crops rather than indiscriminately to the
whole image. Text comparison is permitted only when the supporting objects have
the same class and sufficient spatial similarity. This grounding reduces false
matches caused by unrelated text in different parts of an image.

Strings are lower-cased and stripped of punctuation. Their similarity is the
maximum of:

- normalised Levenshtein similarity;
- token-set Jaccard similarity; and
- substring containment for strings of at least four characters.

OCR confidence weights each comparison. A distinctiveness function gives more
importance to longer strings and strings containing digits, while discounting
common words such as “open”, “road” and “parking”. Missing text is treated as
missing evidence, not negative evidence, because OCR failure is common. When
both sides contain text but disagree, a configurable conflict floor bounds the
penalty so that changing shopfront text cannot erase otherwise strong scene and
object agreement.

OCR is expensive and is therefore scheduled every \(n\) keyframes, limited to a
maximum number of eligible objects, and forced on recovery frames. The current
PaddleOCR configuration requests CUDA; runtime logs must be recorded in the
evaluation to confirm the actual selected device.

## 8. Hierarchical cognitive fusion model

For query \(Q\) and candidate \(C_i\), the unified semantic score is:

\[
\Lambda(Q,C_i)=
\frac{
\tilde w_s S_{\text{scene}}+
\tilde w_o S_{\text{object}}+
\tilde w_t S_{\text{text}}
}{
\tilde w_s+\tilde w_o+\tilde w_t
}.
\]

The configured base weights are \(w_s=0.3\), \(w_o=0.3\) and \(w_t=0.4\).
An effective weight \(\tilde w_l\) is included only if that layer is enabled and
has evidence for both query and candidate. Text weight is additionally scaled
by its evidence distinctiveness. This evidence-aware normalisation is important:
the absence of OCR or objects must not automatically depress an otherwise
valid scene-only candidate.

The architecture is explicitly ablatable. Any one layer or any pair of layers
can be selected; disabling all three is rejected because it would produce no
semantic observation model.

Candidates closer than a configurable keyframe separation are excluded,
preventing trivial retrieval of immediate temporal neighbours. The top-ranked
candidates are accepted only when the best score exceeds the semantic
threshold. A small margin between the two best scores marks the response as
ambiguous for analysis.

## 9. Asynchronous execution and latency control

Semantic inference runs on a worker thread so that ROS image callbacks and the
ORB tracking thread remain responsive. The bounded queue defaults to one item.
If inference falls behind, an unprocessed stale observation is replaced by the
newest observation; recovery needs current evidence more than a long FIFO
backlog.

Latency is further controlled through:

- TensorRT scene and YOLO engines;
- scene-first top-\(K\) retrieval;
- periodic and object-limited OCR;
- lazy OCR initialisation with optional warm-up;
- storage of compact semantic records rather than full images; and
- configurable layer ablation for cost/accuracy measurement.

The evaluation must report per-component median, 95th-percentile and maximum
latency, end-to-end semantic response latency, achieved playback/frame rate and
dropped/replaced semantic tasks.

## 10. Relocalisation protocol

When tracking is lost, ORB-SLAM3 publishes the current image and tracking
metadata. HumanSLAM searches its stored keyframes and returns map/keyframe
identifiers. The tracking thread resolves those identifiers against the live
Atlas and tests them with ORB correspondences and PnP. Pose optimisation and an
adaptive minimum-inlier check determine whether recovery is accepted.

This means the candidate keyframe pose is a hypothesis/reference, not the final
camera pose sent directly into the tracker. ORB-SLAM3 estimates the query pose
from current geometric measurements. Conventional bag-of-words relocalisation
remains available as a fallback.

## 11. Cross-map fusion protocol

Prolonged loss may cause ORB-SLAM3 to create a new Atlas map. HumanSLAM memory
spans map identifiers, so a keyframe in the new map can retrieve a semantically
similar keyframe in an older map. Proposals are stored in a thread-safe bounded
queue and remain applicable for a short five-keyframe window to tolerate
asynchronous inference.

Only candidates belonging to a map different from the current keyframe map are
injected into merge detection. They are prioritised ahead of standard
bag-of-words candidates but do not bypass any safety condition.
`DetectCommonRegionsFromBoW` performs ORB matching, Sim(3) RANSAC, Sim(3)
optimisation and projection validation. ORB-SLAM3 then requires consistency
over three current keyframes. Once verified, its existing `MergeLocal` or
`MergeLocal2` path transforms keyframes and map points, fuses duplicate
observations, updates graph connections and optimises the merged map.

The strict semantic-proposal/geometric-verification boundary is a principal
safety mechanism against catastrophic false map fusion.

## 12. Software verification

Software verification is reported separately from SLAM performance evaluation.
Unit tests check deterministic mathematical behaviour: identity and mismatch
scores, temporal scene weighting, spatial decay, class gating, text handling,
evidence-aware normalisation, recovery thresholds, candidate ranking and all
valid layer combinations. Build tests verify the Python ROS package, message
package, ROS wrapper and modified ORB-SLAM3 library.

Integration tests should additionally check message schema integrity, topic
connectivity, identifier round trips, late-response handling and launch
shutdown. GPU smoke tests should record successful TensorRT deserialisation,
tensor dimensions, YOLO output parsing and PaddleOCR device selection.

These tests establish implementation correctness; they do not establish the
research hypothesis.

## 13. Experimental evaluation design

### 13.1 Conditions

Run the same image sequence and ORB settings under:

1. unmodified ORB-SLAM3 assistance disabled;
2. full HumanSLAM;
3. scene only;
4. objects only;
5. text only;
6. scene + objects;
7. scene + text; and
8. objects + text.

Use repeated runs where nondeterminism is material. Preserve the same frame
range, camera calibration, playback rate and random seeds.

### 13.2 Scenario groups

- nominal traversal;
- illumination/day–night variation;
- weather/seasonal variation;
- viewpoint variation;
- motion blur or short occlusion;
- perceptual aliasing with repeated road/building structure;
- forced tracking-loss intervals;
- re-entry into a previously mapped area after a new map is created; and
- genuinely novel places, where the correct action is to reject all candidates.

The final case is essential: a recovery system must demonstrate calibrated
rejection rather than always returning a place.

### 13.3 Metrics

Report:

- absolute trajectory error (ATE);
- relative pose error (RPE);
- tracking-success proportion and lost-frame count;
- relocalisation success rate, false-relocalisation rate and time to recovery;
- place-recognition Recall@1, Recall@5, precision and rejection performance;
- number of Atlas maps created and remaining;
- true and false map fusions, and time from overlap to fusion;
- semantic score distributions and threshold sensitivity;
- runtime latency, throughput, memory and GPU utilisation.

For map fusion, use a spatial ground-truth rule to label whether two maps truly
overlap. A fusion is correct only if the recovered inter-map transform is
consistent with ground truth and does not materially worsen trajectory error.

### 13.4 Statistical reporting

Report sample counts and distributions, not only a single mean. Use medians and
interquartile ranges for skewed latency and recovery-time data. For paired
sequence comparisons, report the per-sequence difference and an appropriate
paired confidence interval or non-parametric paired test. Include failure cases
and qualitative examples of both successful variation handling and semantic
aliasing.

## 14. Validity, limitations and ethical reporting

Current class selection depends on the labels supported by the deployed YOLO
model. Buildings and billboards are conceptually useful but cannot be claimed
as detected unless the trained model exposes those classes. OCR performance
depends on resolution, viewpoint, language and environmental text changes.
Scene embeddings may encode dataset/domain bias. Semantics cannot recover a
place never previously stored, and geometry may still be insufficient to verify
a correct semantic proposal.

The dissertation should therefore distinguish:

- implemented and build-verified behaviour;
- component accuracy measured on labelled retrieval data;
- end-to-end SLAM improvement measured against ground truth; and
- proposed future work.

At the present stage, HumanSLAM is implemented and its mathematical core is
unit-tested. Whether it improves ORB-SLAM3 must be concluded only after the
controlled baseline, ablation, relocalisation and map-fusion experiments.
