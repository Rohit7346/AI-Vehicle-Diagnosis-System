# Dataset Deduplication Audit Log

**Project:** AI-Powered Multimodal Vehicle Fault Detection and Diagnosis System  
**Stage:** Stage 2.1 – Duplicate Removal and Leakage Resolution  
**Date Executed:** November 4, 2026  
**Action Status:** Completed & Cryptographically Verified  

---

## 1. Incident Overview
During pre-flight cross-split data leakage auditing using SHA-256 content hashing (`AnnotationValidator.detect_data_leakage`), a duplicate image was detected across the `train` and `val` splits of the prototype dataset in `data/prototype_5class/`.

The duplication originated upstream in the public Roboflow archive (`Engine Parts Detector v3 Improved Demo Model`), where the original author uploaded the same iStock photo twice with different randomly generated Roboflow hash suffixes (`.rf.7d2678...` and `.rf.2a0e3d...`).

---

## 2. File Verification & Hash Audit

| Attribute | Training Split Copy (PRESERVED) | Validation Split Copy (REMOVED) |
|---|---|---|
| **Image File Path** | `data/prototype_5class/images/train/istockphoto-953763812-612x612_jpg.rf.7d2678a0a53155decffaf575c9276744.jpg` | `data/prototype_5class/images/val/istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.jpg` |
| **Label File Path** | `data/prototype_5class/labels/train/istockphoto-953763812-612x612_jpg.rf.7d2678a0a53155decffaf575c9276744.txt` | `data/prototype_5class/labels/val/istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.txt` |
| **File Size (Bytes)** | 35,561 bytes | 35,561 bytes |
| **Byte Identity** | Identical (`True`) | Identical (`True`) |
| **SHA-256 Digest** | `e75fee7735d1d9e29c1507adb0a27c7075bc2833ce6bb151ccfcc803db3bffcc` | `e75fee7735d1d9e29c1507adb0a27c7075bc2833ce6bb151ccfcc803db3bffcc` |
| **Annotation Content** | `1 0.908654 0.723558 0.182692 0.427885` | `1 0.909856 0.741587 0.180288 0.358173` |
| **Target Class** | `1` (`battery`) | `1` (`battery`) |

---

## 3. Decision Rationale & Resolution Actions
- **Decision:** Following user approval, the copy in `train` was **retained**, and the copy in `val` was **permanently removed**.
- **Justification:**
  1. The training split preserves sample count and diversity for model feature extraction.
  2. The validation split must evaluate genuine out-of-distribution generalization without data snooping.
  3. The training bounding box ($h = 0.428$) more accurately encapsulates the battery base tray and terminal clamps than the validation box ($h = 0.358$).
- **Actions Executed:**
  1. Removed `data/prototype_5class/images/val/istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.jpg`.
  2. Removed `data/prototype_5class/labels/val/istockphoto-953763812-612x612_jpg.rf.2a0e3dc60a7d4104071c71fdf0efb53b.txt`.

---

## 4. Post-Deduplication Dataset Accounting

| Split | Pre-Deduplication Images | Post-Deduplication Images | Pre-Deduplication Battery Anns | Post-Deduplication Battery Anns | Radiator Anns (ID 3) | Total Retained Annotations |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`train`** | 985 | **985** | 945 | **945** | 587 | **1,532** |
| **`val`** | 102 | **101** | 100 | **99** | 62 | **161** |
| **`test`** | 61 | **61** | 59 | **59** | 31 | **90** |
| **TOTAL** | **1,148** | **1,147** | **1,104** | **1,103** | **680** | **1,783** |

---

## 5. Verification & Test Suite Status
- **Leakage Audit (`AnnotationValidator.detect_data_leakage`):**
  - `has_leakage`: **`False`**
  - `filename_overlaps`: `[]` (0 overlaps)
  - `content_duplicates`: `[]` (0 duplicate SHA-256 hashes)
- **Automated Tests:** 16 / 16 passed (`pytest -v`).
