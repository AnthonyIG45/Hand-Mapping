import cv2
import numpy as np
from scipy.spatial import distance

class HandBiometricSystem:
    def __init__(self, image_paths, target_width=800):
        self.image_paths = image_paths
        self.target_width = target_width
        self.master_points = []
        self.feature_vectors = []

    def resize_img(self, img):
        h, w = img.shape[:2]
        ratio = self.target_width / float(w)
        return cv2.resize(img, (self.target_width, int(h * ratio)))

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.master_points.append((x, y))
            cv2.circle(params['img'], (x, y), 4, (0, 255, 0), -1)
            cv2.putText(params['img'], f"P{len(self.master_points)}", (x+5, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow('1. Landmarking (Click 12 points)', params['img'])

    def get_thickness_and_axis(self, img, p1, p2, is_f1=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if is_f1:
            # F1 profile is directly along the points
            axis_p1, axis_p2 = p1, p2
            line_pts = np.linspace(p1, p2, 200).astype(int)
        else:
            # Perpendicular axis at midpoint
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2) or 1
            ux, uy = -dy/length, dx/length
            axis_p1 = (int(mid[0] - ux*150), int(mid[1] - uy*150))
            axis_p2 = (int(mid[0] + ux*150), int(mid[1] + uy*150))
            line_pts = np.linspace(axis_p1, axis_p2, 300).astype(int)

        profile = []
        valid_coords = []
        for pt in line_pts:
            if 0 <= pt[1] < gray.shape[0] and 0 <= pt[0] < gray.shape[1]:
                profile.append(gray[pt[1], pt[0]])
                valid_coords.append(pt)
        
        # Find the distance between high-contrast changes
        profile = np.array(profile, dtype=np.float32)
        if len(profile) < 2: return 0, axis_p1, axis_p2

        # Detect where hand meets background
        binary = profile < np.mean(profile)
        diff = np.diff(binary.astype(int))
        edges = np.where(diff != 0)[0]

        if len(edges) >= 2:
            start_pt = np.array(valid_coords[edges[0]])
            end_pt = np.array(valid_coords[edges[-1]])
            dist = np.linalg.norm(start_pt - end_pt)
            return dist, axis_p1, axis_p2
        
        return 0, axis_p1, axis_p2

    def run(self):
        first_img = cv2.imread(self.image_paths[0])
        if first_img is None:
            print("Error: Could not load first image. Check file paths.")
            return

        img1_resized = self.resize_img(first_img)
        clone = img1_resized.copy()
        
        cv2.imshow('1. Landmarking (Click 12 points)', img1_resized)
        cv2.setMouseCallback('1. Landmarking (Click 12 points)', self.click_event, {'img': img1_resized})
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(self.master_points) < 12:
            print(f"Error: Required 12 points, but you clicked {len(self.master_points)}.")
            return

        for i, path in enumerate(self.image_paths):
            raw = cv2.imread(path)
            if raw is None: continue
            
            img = self.resize_img(raw)
            h, w = img.shape[:2]
            vis_img = img.copy()
            vector = []

            for j in range(0, 12, 2):
                p1, p2 = self.master_points[j], self.master_points[j+1]
                is_f1 = (j == 0)
                label = f"F{int(j/2 + 1)}"

                # Calculate the thickness and axes
                dist, ax1, ax2 = self.get_thickness_and_axis(img, p1, p2, is_f1)
                vector.append(dist)

                # Calculate the direction vector
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                
                inf_p1 = (int(p1[0] - 2000 * dx), int(p1[1] - 2000 * dy))
                inf_p2 = (int(p1[0] + 2000 * dx), int(p1[1] + 2000 * dy))

                cv2.line(vis_img, inf_p1, inf_p2, (255, 0, 0), 1) 
                cv2.circle(vis_img, p1, 3, (0, 255, 0), -1)
                cv2.circle(vis_img, p2, 3, (0, 255, 0), -1)
                cv2.line(vis_img, ax1, ax2, (0, 0, 255), 1)
                cv2.putText(vis_img, label, (p1[0] + 10, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            self.feature_vectors.append(vector)

            if i == 0:
                cv2.imshow("Requirement 1b (Infinite Blue) & 1c (Red Axes)", vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        num_imgs = len(self.feature_vectors)
        dist_matrix = np.zeros((num_imgs, num_imgs))

        for r in range(num_imgs):
            for c in range(r, num_imgs):
                d = distance.euclidean(self.feature_vectors[r], self.feature_vectors[c])
                dist_matrix[r, c] = round(d, 4)

        print("\n--- FEATURE VECTORS ---")
        for idx, vec in enumerate(self.feature_vectors):
            print(f"Hand {idx+1}: {[round(v, 2) for v in vec]}")

        print("\n--- MATRIX (5x5) ---")
        print(dist_matrix)

image_files = ['Hand_Image1.jpg', 'Hand_Image2.jpg', 'Hand_Image3.jpg', 'Hand_Image4.jpg', 'Hand_Image5.jpg']

if __name__ == "__main__":
    app = HandBiometricSystem(image_files)
    app.run()