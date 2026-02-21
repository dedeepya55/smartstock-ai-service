import cv2
from difflib import SequenceMatcher


def _ocr_texts_match(text1_list, text2_list, threshold=0.6):
    """
    Fuzzy match two OCR text lists.
    Returns True if they are likely the same product (>threshold similarity).
    Uses both word-level and character-level matching.
    """
    if not text1_list or not text2_list:
        return False
    
    # Join all text in each list and normalize
    t1 = " ".join(text1_list).lower().strip()
    t2 = " ".join(text2_list).lower().strip()
    
    # Remove common noise chars
    for char in "[](){}\'\".,;:!?-":
        t1 = t1.replace(char, " ")
        t2 = t2.replace(char, " ")
    
    t1_words = [w for w in t1.split() if len(w) > 1]
    t2_words = [w for w in t2.split() if len(w) > 1]
    
    if not t1_words or not t2_words:
        return False
    
    # Word-level Jaccard similarity
    t1_word_set = set(t1_words)
    t2_word_set = set(t2_words)
    intersection = len(t1_word_set & t2_word_set)
    union = len(t1_word_set | t2_word_set)
    word_similarity = intersection / union if union > 0 else 0
    
    # Character-level matching using SequenceMatcher (handles typos)
    char_similarity = SequenceMatcher(None, t1, t2).ratio()
    
    # Use the better of the two similarities
    best_similarity = max(word_similarity, char_similarity)
    
    return best_similarity >= threshold


def _normalize_label_fuzzy(cleaned_label, ocr_texts):
    """
    For fuzzy matching, use a combination of cleaned label and OCR text.
    This helps group similar products even with OCR variations.
    """
    if not ocr_texts or all(len(t.strip()) == 0 for t in ocr_texts):
        return cleaned_label.lower()
    
    # Join OCR text and extract key words
    ocr_str = " ".join(ocr_texts).lower()
    for char in "[](){}\'\".,;:!?-":
        ocr_str = ocr_str.replace(char, " ")
    
    words = [w for w in ocr_str.split() if len(w) > 2]
    
    if words:
        return " ".join(sorted(set(words)))  # canonical form
    
    return cleaned_label.lower()


def check_arrangement(detections, row_thresh=30):
    """
    Improved arrangement checker with fuzzy product matching.

    detections: list of dicts
        [{'label': str, 'ocr_text': [str,], 'x_center': int, 'y_center': int, 'bbox': (x1,y1,x2,y2)}, ...]
    row_thresh: int - vertical threshold (px) to group into same row

    Returns:
        status: "CORRECT" or "INCORRECT"
        messages: list of strings per row
        wrong_boxes: list of bbox tuples for misplaced products
    """

    def _norm_label(s):
        if s is None:
            return ""
        return "".join(ch for ch in s.lower() if ch.isalnum())

    # Step 1: Group products into rows by y_center using clustering
    rows = []  # list of dicts: {'y_mean': float, 'products': [p,...]}
    for p in detections:
        y = p["y_center"]
        placed = False
        for row in rows:
            if abs(row["y_mean"] - y) <= row_thresh:
                row["products"].append(p)
                # update mean
                row["y_mean"] = sum(x["y_center"] for x in row["products"]) / len(row["products"])
                placed = True
                break
        if not placed:
            rows.append({"y_mean": y, "products": [p]})

    # Sort rows top->bottom (small y -> top of image)
    rows.sort(key=lambda r: r["y_mean"])

    overall_status = "CORRECT"
    messages = []
    wrong_boxes = []

    for row_idx, row in enumerate(rows, start=1):
        prods = row["products"]
        # sort by x to get left-to-right order
        prods.sort(key=lambda x: x["x_center"])
        
        # Debug: print original labels before fuzzy matching
        orig_labels = [p.get("label", "unknown") for p in prods]
        print(f"\n[DEBUG Row {row_idx}] Original labels: {orig_labels}")
        
        # Fuzzy grouping: assign each product to a canonical/fuzzy group
        fuzzy_groups = []  # list of (canonical_label, original_labels, indices)
        
        for i, p in enumerate(prods):
            label = p.get("label", "unknown")
            ocr_text = p.get("ocr_text", [])
            fuzzy_norm = _normalize_label_fuzzy(label, ocr_text)
            
            # Try to match with existing groups
            matched = False
            for group in fuzzy_groups:
                # Check if this product matches the group
                if _ocr_texts_match(group["ocr_sample"], ocr_text, threshold=0.5):
                    group["indices"].append(i)
                    group["labels"].append(label)
                    print(f"[DEBUG Row {row_idx}] Product {i} ({label}, OCR:{ocr_text}) → MATCHED group with sample {group['ocr_sample']}")
                    matched = True
                    break
            
            if not matched:
                # Create new group
                fuzzy_groups.append({
                    "fuzzy_norm": fuzzy_norm,
                    "ocr_sample": ocr_text,
                    "labels": [label],
                    "indices": [i]
                })
                print(f"[DEBUG Row {row_idx}] Product {i} ({label}, OCR:{ocr_text}) → NEW group")
        
        # Create label sequence using fuzzy grouping
        labels = ["unknown"] * len(prods)
        for group in fuzzy_groups:
            # Use the first non-"unknown" label in group, or "unknown"
            group_label = next((l for l in group["labels"] if l != "unknown"), "unknown")
            for idx in group["indices"]:
                labels[idx] = group_label
        
        print(f"[DEBUG Row {row_idx}] After fuzzy grouping: {labels}")
        norm_labels = [_norm_label(l) for l in labels]

        # create segments of consecutive identical normalized labels
        segments = []  # list of (norm_label, start_idx, end_idx)
        if norm_labels:
            cur_label = norm_labels[0]
            start = 0
            for i in range(1, len(norm_labels)):
                if norm_labels[i] != cur_label:
                    segments.append((cur_label, start, i - 1))
                    cur_label = norm_labels[i]
                    start = i
            segments.append((cur_label, start, len(norm_labels) - 1))

        print(f"[DEBUG Row {row_idx}] Segments: {segments}")

        # map normalized label -> list of segment indices
        seg_map = {}
        for si, (lab, s, e) in enumerate(segments):
            seg_map.setdefault(lab, []).append((si, s, e))

        misplaced_indices = set()

        # For any normalized label that occurs in multiple non-adjacent segments, 
        # mark ONLY the intruding products (products that interrupt it) as misplaced
        for lab, segs in seg_map.items():
            if len(segs) > 1:
                first = segs[0][1]
                last = segs[-1][2]
                print(f"[DEBUG Row {row_idx}] Label '{lab}' appears in {len(segs)} non-adjacent segments (indices {first} to {last}) → marking intruders")
                
                # Mark only products that are NOT of this label as intruders
                for idx in range(first, last + 1):
                    if norm_labels[idx] != lab:
                        misplaced_indices.add(idx)
                        print(f"[DEBUG Row {row_idx}]   → idx {idx} ({labels[idx]}) is an intruder")

        if misplaced_indices:
            overall_status = "INCORRECT"
            # return original label strings for the misplaced indices (collapse duplicates)
            misplaced_labels = sorted({labels[i] for i in misplaced_indices})
            print(f"[DEBUG Row {row_idx}] MISPLACED indices: {sorted(misplaced_indices)} → labels: {misplaced_labels}")
            messages.append(f"Row {row_idx} misplaced products: {', '.join(misplaced_labels)}")
            # gather bboxes for misplaced indices
            for i in sorted(misplaced_indices):
                wrong_boxes.append(prods[i]["bbox"])
        else:
            print(f"[DEBUG Row {row_idx}] No misplaced products")
            messages.append(f"Row {row_idx} correct")

    # Deduplicate wrong_boxes while preserving order
    seen = set()
    uniq_wrong = []
    for b in wrong_boxes:
        if b not in seen:
            uniq_wrong.append(b)
            seen.add(b)

    return overall_status, messages, uniq_wrong
