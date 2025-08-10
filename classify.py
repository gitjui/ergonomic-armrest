def classify_armrest_height(data):
    """
    Classifies armrest height as 'Optimal', 'Too High', or 'Too Low'.
    Handles:
    - Standing without chair: compare desk vs elbow
    - Sitting with desk: compare armrest vs elbow and desk
    - Sitting without desk: compare armrest vs elbow only
    """
    print(data)

    required_keys = ["isChair", "isDesk", "isPerson", "arm_landmarks_detected", "landmarks"]
    if not all(k in data for k in required_keys):
        return "Insufficient Data"
    if not (data["isPerson"] and data["arm_landmarks_detected"]):
        return "Insufficient Data"

    shoulder = data["landmarks"]["shoulder"]
    elbow = data["landmarks"]["elbow"]

    # Calculate resting elbow y (shoulder + vertical distance from shoulder to elbow)
    dx = shoulder["x"] - elbow["x"]
    dy = shoulder["y"] - elbow["y"]
    elbow_shoulder_dist = (dx**2 + dy**2) ** 0.5
    resting_elbow_y = shoulder["y"] + elbow_shoulder_dist * 1.1  # slight offset

    allowed_margin = resting_elbow_y / 10  # 10% margin

    # ---------------------------
    # CASE 1: Standing, no chair
    # ---------------------------
    if data.get("isStanding"):
        if "desk_y" not in data:
            return "Insufficient Data"
        desk_y = data["desk_y"]
        if desk_y > resting_elbow_y + allowed_margin:
            return "Too High"
        elif desk_y < resting_elbow_y - allowed_margin:
            return "Too Low"
        else:
            return "Optimal"

    # ---------------------------
    # Armrest height calculation
    # ---------------------------
    if not data.get("isChair") or "armrest_box" not in data:
        return "Insufficient Data"
    armrest_box = data["armrest_box"]
    armrest_top_y = float(armrest_box["y"])
    armrest_h = float(armrest_box["h"])
    armrest_avg_y = armrest_top_y + (armrest_h / 2)

    # ---------------------------
    # CASE 2: Sitting with desk
    # ---------------------------
    if data.get("isDesk"):
        desk_y = data["desk_y"]
        if armrest_avg_y > resting_elbow_y + allowed_margin or armrest_avg_y > desk_y + allowed_margin:
            return "Too High"
        elif armrest_avg_y < resting_elbow_y - allowed_margin:
            return "Too Low"
        else:
            return "Optimal"

    # ---------------------------
    # CASE 3: Sitting without desk
    # ---------------------------
    else:
        if armrest_avg_y > resting_elbow_y + allowed_margin:
            return "Too High"
        elif armrest_avg_y < resting_elbow_y - allowed_margin:
            return "Too Low"
        else:
            return "Optimal"
