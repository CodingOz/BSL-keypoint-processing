import json
import numpy as np
import os
import shutil
import subprocess

# manages all checks related to keypoint orientation


class OrientationChecker:

    # Extracts rotation metadata from video file using ffprobe.
    # Returns angle needed to return it to original
    def getRotationMetadata(self, filename):
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'v:0', str(filename)
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5)
            metadata = json.loads(result.stdout)

            print(f"  FFprobe metadata for {os.path.basename(filename)}:")

            for stream in metadata.get('streams', []):
                # Check multiple possible rotation fields
                rotation = stream.get('tags', {}).get('rotate')
                if rotation:
                    print(f"  Found rotation needed: {-rotation}")

                    return int(-rotation) % 360

                # Check side_data for rotation
                side_data = stream.get('side_data_list', [])
                for data in side_data:
                    if data.get('rotation'):
                        print(
                            f"rotation needed: {(-data.get('rotation')) % 360}")
                        return int(-data.get('rotation')) % 360

            print("No rotation metadata found")
            return 0

        except Exception as e:
            print(f"  ffprobe failed: {e}")
            return 0

    # finds the percentage of frames point_a set is right/above point_b set
    def getOffset(self, start_frame, end_frame, json_data):
        right_count = 0
        above_count = 0
        right_distance = 0
        above_distance = 0

        total_points_counted = 0

        for frame in range(start_frame, end_frame + 1):
            for p in range(min(len(json_data["frames"][frame]["hands"]["left"]),
                               len(json_data["frames"][frame]["hands"]["right"]))):
                # ignoring z values for now as doesnt reliably corrispond with
                # orientation
                point_p_left = json_data["frames"][frame]["hands"]["left"][p]
                a_x = point_p_left["x"]
                a_y = point_p_left["y"]

                point_p_right = json_data["frames"][frame]["hands"]["right"][p]
                b_x = point_p_right["x"]
                b_y = point_p_right["y"]

                # only count where both points exist
                if point_p_left is None or point_p_right is None:
                    continue
                else:
                    total_points_counted += 1
                    right_distance += a_x - b_x
                    above_distance += a_y - b_y

                    if a_x > b_x:
                        right_count += 1
                    if a_y < b_y:
                        above_count += 1
        if total_points_counted == 0:
            raise ValueError("no points counted\n")

        percentage_right = (right_count / total_points_counted) * 100
        percentage_above = (above_count / total_points_counted) * 100

        return percentage_right, percentage_above, right_distance, above_distance

    # estimates the orientation based on the relitive position of the 2 hand clusters
    # designed for symmetric signs like 'B' or 'or'

    def checkHandOrientation(self, json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)

        frame_num = len(json_data["frames"])

        # center frames have much more predictible behaviour
        start_frame = frame_num // 4
        end_frame = start_frame * 3

        # finds the percentage of points on the left hand
        # that is right of / above of the right hand
        x_percentage, y_percentage, x_distance, y_distance = self.getOffset(
            start_frame, end_frame, json_data)

        Y_is_unclear = (y_percentage > 10) and (y_percentage < 90)
        X_is_unclear = (x_percentage > 10) and (x_percentage < 90)

        # if there is no consitant corrilation
        if X_is_unclear and Y_is_unclear:
            raise ValueError(
                f"Unsure of orientation for: {json_path}\npercentage_x: {x_percentage}\npercentage_y: {y_percentage}"
            )

        # probably upside-down
        elif x_percentage > 90 and Y_is_unclear:
            return 180

        # probably upright
        elif x_percentage < 10 and Y_is_unclear:
            return 0

        # probably clockwise 90 degrees
        elif y_percentage > 90 and X_is_unclear:
            return 90

        # probably anti-clockwise 90 degrees
        elif y_percentage < 10 and X_is_unclear:
            return 270

        # if one hand is concistantly both above/below and left/right of the other
        # then the diciction is based on which has a larger total distance
        elif abs(x_distance) > abs(y_distance):
            if x_distance <= 0:
                return 180
            else:
                return 0
        elif abs(x_distance) < abs(y_distance):
            if y_distance <= 0:
                return 90
            else:
                return 270
        else:
            raise ValueError(
                f"Unsure of orientation for: {json_path}\npercentage_x: as x and y have lenth {x_distance}"
            )

    # Rotate all pose and hand  points in the JSON file around based on center (0.5, 0.5)
    # ignores `z` and `visibility` fields

    @staticmethod
    def rotatePoints(json_path, target_path, angle, center=(0.5, 0.5)):
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        center_arr = np.array(center, dtype=float)

        for frame in json_data.get("frames", []):
            # rotate landmarks in `pose` list
            for lm in frame.get("pose", []):
                x = lm.get("x")
                y = lm.get("y")
                if x is None or y is None:
                    continue
                vec = np.array([x, y], dtype=float) - center_arr
                rotated = R.dot(vec) + center_arr
                lm["x"] = float(rotated[0])
                lm["y"] = float(rotated[1])

            # rotate landmarks in hands (left and right)
            hands = frame.get("hands", {})
            for side in ("left", "right"):
                for lm in hands.get(side, []):
                    x = lm.get("x")
                    y = lm.get("y")
                    if x is None or y is None:
                        continue
                    vec = np.array([x, y], dtype=float) - center_arr
                    rotated = R.dot(vec) + center_arr
                    lm["x"] = float(rotated[0])
                    lm["y"] = float(rotated[1])

        # Write the rotated JSON to the target path
        with open(target_path, "w", encoding="utf-8") as out_f:
            json.dump(json_data, out_f, ensure_ascii=False, indent=2)

    # checks orientation of all keypoints in file
    def fixRotations(self, folder_path, target_path):
        os.makedirs(target_path, exist_ok=True)

        for root, _, files in os.walk(folder_path):
            for fname in files:
                src_path = os.path.join(root, fname)
                rel_path = os.path.relpath(src_path, folder_path)
                dst_path = os.path.join(target_path, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                if not fname.lower().endswith('.json'):
                    shutil.copy2(src_path, dst_path)
                    continue

                try:
                    detected = self.checkHandOrientation(src_path)
                except Exception as e:
                    print(
                        f"[fixRotations] could not determine orientation for {rel_path}: {e}")
                    shutil.copy2(src_path, dst_path)
                    continue

                rotate_by = (360 - int(detected)) % 360
                if rotate_by == 0:
                    shutil.copy2(src_path, dst_path)
                    print(f"[fix_rotations] copied (no rotation): {rel_path}")
                else:
                    try:
                        # rotate and write to destination
                        self.rotatePoints(src_path, dst_path, rotate_by)
                    except Exception as e:
                        print(
                            f"[fix_rotations] failed to rotate {rel_path}: {e}")
                        shutil.copy2(src_path, dst_path)
