from scipy.spatial import distance
import cv2
import numpy as np
# 测量电缆直径的函数
def count_dia(tgt, row):
    cols = tgt.shape[1]
    point1, point2 = 0, 0

    # 获取直径的起点
    for j in range(cols):
        if tgt[row, j] == 255:  # 白色像素
            point1 = j
            break

    # 获取直径的终点
    for j in range(cols - 1, -1, -1):
        if tgt[row, j] == 255:  # 白色像素
            point2 = j
            break

    diameter = point2 - point1 + 1
    print(f"Diameter: {diameter}")
    return point1, point2

# 绘制直径线和标注
def draw_lines(org_img, p1, p2, row):
    thickness = 2
    diameter = p2 - p1 + 1

    # 左右线
    cv2.line(org_img, (p1, row), (p1 - 20, row), (0, 0, 255), thickness)
    cv2.line(org_img, (p2, row), (p2 + 20, row), (0, 0, 255), thickness)

    # 标注直径文字
    cv2.putText(org_img, f"Diameter {diameter}", (p2 + 50, row + 6),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
    
    
    
def defects(edges_img, img):
    # 查找轮廓
    contours, _ = cv2.findContours(edges_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤掉像素点数小于5的轮廓
    min_contour_area = 5
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]

    # 合并相邻的轮廓
    merged_contours = []
    for contour in contours:
        merged = False
        for i, existing_contour in enumerate(merged_contours):
            # 如果两个轮廓的边界框有交叠或接近，则合并
            x1, y1, w1, h1 = cv2.boundingRect(existing_contour)
            x2, y2, w2, h2 = cv2.boundingRect(contour)

            if (
                abs(x1 - x2) < 20 or abs(y1 - y2) < 20 or
                abs((x1 + w1) - (x2 + w2)) < 20 or abs((y1 + h1) - (y2 + h2)) < 20
            ):
                merged_contours[i] = np.vstack((existing_contour, contour))
                merged = True
                break

        if not merged:
            merged_contours.append(contour)

    # 获取每个合并后轮廓的最小外接圆
    centers = []
    radius = []
    for contour in merged_contours:
        center, rad = cv2.minEnclosingCircle(contour)
        centers.append(center)
        radius.append(rad)

    max_radius = max(radius) if radius else 0

    # 移除干扰性的小圆
    filtered_centers = []
    filtered_radius = []
    for i in range(len(radius)):
        if radius[i] < 5 or radius[i] >= max_radius or max_radius - radius[i] < 1.0:
            continue
        filtered_centers.append(centers[i])
        filtered_radius.append(radius[i])

    # 合并中心点距离较近的圆，确保每个缺陷只有一个圆
    final_centers = []
    final_radius = []
    for i, (center1, rad1) in enumerate(zip(filtered_centers, filtered_radius)):
        if not final_centers:
            final_centers.append(center1)
            final_radius.append(rad1)
            continue

        is_close = False
        for j, (center2, rad2) in enumerate(zip(final_centers, final_radius)):
            if distance.euclidean(center1, center2) < rad1 + rad2:
                # 如果距离较近，合并为一个更大的圆
                avg_center = (
                    (center1[0] + center2[0]) / 2,
                    (center1[1] + center2[1]) / 2,
                )
                max_radius = max(rad1, rad2) + 10  # 增加 10 像素扩展范围
                final_centers[j] = avg_center
                final_radius[j] = max_radius
                is_close = True
                break

        if not is_close:
            final_centers.append(center1)
            final_radius.append(rad1)

    # 绘制有效的缺陷圆圈并标注分类
    for i in range(len(final_radius)):
        cx, cy = int(final_centers[i][0]), int(final_centers[i][1])
        rad = int(final_radius[i] + 15)  # 再次扩大半径，确保完全覆盖
        cv2.circle(img, (cx, cy), rad, (0, 255, 0), 2)

        if rad <= 25:
            label = "Defect: Pin Hole"
        elif rad <= 70:
            label = "Defect: Cut"
        else:
            label = "Defect: Scratch"

        cv2.putText(img, label, (cx + 70, cy + 30), 
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 100, 255), 2)


def main():
    # 读取图像
    img = cv2.imread("Input Images\Scratches.bmp")
    if img is None:
        print("Error: Unable to load image.")
        return

    # 转灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 噪声处理：将背景转换为白色
    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if img_gray[i, j] < 40 or j < (cols // 2) - 120 or j > (cols // 2) + 120:
                img_gray[i, j] = 255

    # 显示灰度图像
    cv2.imshow("Gray Image", img_gray)

    # 边缘检测
    edges_img = cv2.Canny(img_gray, 150, 300, apertureSize=3, L2gradient=False)
    cv2.imshow("Edges", edges_img)

    # 测量直径并绘制
    rows_to_measure = [100, 625]
    for row in rows_to_measure:
        p1, p2 = count_dia(edges_img, row)
        draw_lines(img, p1, p2, row)

    # 检测缺陷
    defects(edges_img, img)

    # 显示最终结果
    cv2.imshow("Final Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
