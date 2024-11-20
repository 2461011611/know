import os
from scipy.spatial import distance
import cv2
import numpy as np

# 测量电缆直径的函数
def count_dia(tgt, row):
    cols = tgt.shape[1]
    point1, point2 = 0, 0

    for j in range(cols):
        if tgt[row, j] == 255:
            point1 = j
            break

    for j in range(cols - 1, -1, -1):
        if tgt[row, j] == 255:
            point2 = j
            break

    diameter = point2 - point1 + 1
    print(f"Diameter: {diameter}")
    return point1, point2

# 绘制直径线和标注
def draw_lines(org_img, p1, p2, row):
    thickness = 2
    diameter = p2 - p1 + 1

    cv2.line(org_img, (p1, row), (p1 - 20, row), (0, 0, 255), thickness)
    cv2.line(org_img, (p2, row), (p2 + 20, row), (0, 0, 255), thickness)
    cv2.putText(org_img, f"Diameter {diameter}", (p2 + 50, row + 6),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)

# 检测缺陷
def defects(edges_img, img):
    contours, _ = cv2.findContours(edges_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 5
    contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]

    merged_contours = []
    for contour in contours:
        merged = False
        for i, existing_contour in enumerate(merged_contours):
            x1, y1, w1, h1 = cv2.boundingRect(existing_contour)
            x2, y2, w2, h2 = cv2.boundingRect(contour)
            if abs(x1 - x2) < 20 or abs(y1 - y2) < 20 or \
               abs((x1 + w1) - (x2 + w2)) < 20 or abs((y1 + h1) - (y2 + h2)) < 20:
                merged_contours[i] = np.vstack((existing_contour, contour))
                merged = True
                break
        if not merged:
            merged_contours.append(contour)

    centers, radius = [], []
    for contour in merged_contours:
        center, rad = cv2.minEnclosingCircle(contour)
        centers.append(center)
        radius.append(rad)

    max_radius = max(radius) if radius else 0
    filtered_centers, filtered_radius = [], []
    for i in range(len(radius)):
        if radius[i] < 5 or radius[i] >= max_radius or max_radius - radius[i] < 1.0:
            continue
        filtered_centers.append(centers[i])
        filtered_radius.append(radius[i])

    final_centers, final_radius = [], []
    for center1, rad1 in zip(filtered_centers, filtered_radius):
        if not final_centers:
            final_centers.append(center1)
            final_radius.append(rad1)
            continue

        is_close = False
        for j, (center2, rad2) in enumerate(zip(final_centers, final_radius)):
            if distance.euclidean(center1, center2) < rad1 + rad2:
                avg_center = ((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2)
                max_radius = max(rad1, rad2) + 10
                final_centers[j] = avg_center
                final_radius[j] = max_radius
                is_close = True
                break

        if not is_close:
            final_centers.append(center1)
            final_radius.append(rad1)

    for i in range(len(final_radius)):
        cx, cy = int(final_centers[i][0]), int(final_centers[i][1])
        rad = int(final_radius[i] + 15)
        cv2.circle(img, (cx, cy), rad, (0, 255, 0), 2)

        if rad <= 25:
            label = "Defect: Pin Hole"
        elif rad <= 70:
            label = "Defect: Cut"
        else:
            label = "Defect: Scratch"

        cv2.putText(img, label, (cx + 70, cy + 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 100, 255), 2)

# 主函数
def main():
    input_dir = "Input Images"
    output_dir = "Output Images"
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Unable to load image {file_name}. Skipping.")
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, cols = img_gray.shape
        for i in range(rows):
            for j in range(cols):
                if img_gray[i, j] < 40 or j < (cols // 2) - 120 or j > (cols // 2) + 120:
                    img_gray[i, j] = 255

        edges_img = cv2.Canny(img_gray, 150, 300, apertureSize=3, L2gradient=False)

        rows_to_measure = [100, 625]
        for row in rows_to_measure:
            p1, p2 = count_dia(edges_img, row)
            draw_lines(img, p1, p2, row)

        defects(edges_img, img)

        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, img)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    main()
