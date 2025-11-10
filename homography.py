## Be able to upload two images and get the homography matrix and stitch them together

import argparse
import cv2
import numpy as np
import os

def get_keypoints(image, nfeatures=4000):
    if image is None:
        raise ValueError("Input image is empty (failed to load). Check the path.")
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp = orb.detect(greyscale_image, None)
    print(f"Keypoints found: {len(kp)}")
    kp, des = orb.compute(greyscale_image, kp)
    kp = np.array([kp.pt for kp in kp])
    return kp, des


def match_keypoints(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        raise ValueError("Descriptor extraction failed; no features found in one or both images.")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches_knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append([m.queryIdx, m.trainIdx])
    good = np.array(good, dtype=np.int32)
    return good

## Simplified: using OpenCV's findHomography with RANSAC


def bounds(img1_shape, img2_shape, H):
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]
    corners = np.array([[0,0], [w1-1,0], [w1-1, h1-1], [0, h1-1]], dtype=np.float64)
    corners = np.hstack([corners, np.ones((4,1))])
    warped = (H @ corners.T).T # Map corners to points in new plane
    warped_xy = warped[:, :2]/warped[:, 2:3] ## Normalize by last dimension such that everything is on the same plane again
    all_x = np.hstack([warped_xy[:,0], [0, w2-1]])
    all_y = np.hstack([warped_xy[:,1], [0, h2-1]])

    min_x = int(np.floor(all_x.min()))
    min_y = int(np.floor(all_y.min()))
    max_x = int(np.ceil(all_x.max()))
    max_y = int(np.ceil(all_y.max()))

    W = max_x - min_x + 1
    Hh = max_y - min_y + 1

    # translation to keep everything positive on canvas
    T = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]], dtype=np.float64)
    return (W, Hh), T, (min_x, min_y)




def warp_to_canvas(img_src, H, canvas_size, T):
    H_canvas = T @ H                  # include translation
    warped = cv2.warpPerspective(img_src, H_canvas, canvas_size)
    mask = cv2.warpPerspective(np.ones(img_src.shape[:2], np.uint8)*255, H_canvas, canvas_size)
    return warped, mask
def paste_reference(img_ref, canvas_size, offset):
    W, Hh = canvas_size
    tx, ty = offset  # can be negative
    h_r, w_r = img_ref.shape[:2]

    # 1) intersection on the canvas
    x_start = max(0, tx)
    x_end   = min(W, tx + w_r)
    y_start = max(0, ty)
    y_end   = min(Hh, ty + h_r)

    out  = np.zeros((Hh, W, 3), dtype=np.uint8)
    mask = np.zeros((Hh, W), dtype=np.uint8)

    # 2) empty intersection? nothing to paste
    if x_end <= x_start or y_end <= y_start:
        return out, mask

    # 3) map back to ref-image coordinates
    ref_x0 = x_start - tx
    ref_x1 = x_end   - tx
    ref_y0 = y_start - ty
    ref_y1 = y_end   - ty

    # 4) paste
    out[y_start:y_end, x_start:x_end] = img_ref[ref_y0:ref_y1, ref_x0:ref_x1]
    mask[y_start:y_end, x_start:x_end] = 255
    return out, mask

def simple_blend(warped_img, warped_mask, ref_img_on_canvas, ref_mask):
    # where both valid → average, else take valid
    out = warped_img.copy()
    both = (warped_mask>0) & (ref_mask>0)
    only_ref = (warped_mask==0) & (ref_mask>0)
    out[only_ref] = ref_img_on_canvas[only_ref]
    out[both] = ((out[both].astype(np.float32) + ref_img_on_canvas[both].astype(np.float32)) * 0.5).astype(np.uint8)
    return out


## Removed combine() for simplicity


def stitch_images(img1_bgr, img2_bgr, nfeatures: int = 4000, ratio: float = 0.75, ransac_thresh: float = 3.0):
    """Stitch two BGR images and return the stitched BGR image."""
    kp1, des1 = get_keypoints(img1_bgr, nfeatures=nfeatures)
    kp2, des2 = get_keypoints(img2_bgr, nfeatures=nfeatures)
    matches = match_keypoints(des1, des2, ratio=ratio)
    if len(matches) < 4:
        raise ValueError("Not enough good matches to estimate a homography.")
    src = kp1[matches[:, 0]].astype(np.float32)
    dst = kp2[matches[:, 1]].astype(np.float32)

    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    canvas_size, T, offset = bounds(img1_bgr.shape, img2_bgr.shape, H)
    W, Hh = canvas_size

    warped1, mask1 = warp_to_canvas(img1_bgr, H, (W, Hh), T)
    ref2_on_canvas, mask2 = paste_reference(img2_bgr, (W, Hh), offset)
    stitched = simple_blend(warped1, mask1, ref2_on_canvas, mask2)
    return stitched



def main():
    ap = argparse.ArgumentParser()
    script_dir = os.path.dirname(__file__)
    default_img1 = os.path.join(script_dir, "test_images", "IMG_2636.jpeg")
    default_img2 = os.path.join(script_dir, "test_images", "IMG_2637.jpeg")
    ap.add_argument("image1", nargs="?", default=default_img1, help="Path to first image")
    ap.add_argument("image2", nargs="?", default=default_img2, help="Path to second image")
    ap.add_argument("-o", "--output", default="stitched_output.png", help="Output image path")
    ap.add_argument("--features", type=int, default=4000, help="Number of ORB features")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio for KNN match filtering")
    ap.add_argument("--ransac", type=float, default=5.0, help="RANSAC reprojection threshold (pixels)")
    args = ap.parse_args()

    img1_bgr = cv2.imread(args.image1, cv2.IMREAD_COLOR)
    img2_bgr = cv2.imread(args.image2, cv2.IMREAD_COLOR)
    if img1_bgr is None:
        raise FileNotFoundError(f"Failed to load image1: {args.image1}")
    if img2_bgr is None:
        raise FileNotFoundError(f"Failed to load image2: {args.image2}")

    stitched = stitch_images(img1_bgr, img2_bgr, nfeatures=args.features, ratio=args.ratio, ransac_thresh=args.ransac)
    ok = cv2.imwrite(args.output, stitched)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.output}")
    print(f"Wrote {args.output}")
if __name__ == "__main__":
    main()