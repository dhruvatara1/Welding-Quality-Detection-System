import os
import pandas as pd
import numpy as np
import random
import cv2
from PIL import Image, ImageDraw

# --- Configuration ---
NUM_SAMPLES = 50000
IMAGE_SIZE = (64, 64)
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
TABULAR_FILE = os.path.join(DATA_DIR, "welding_data.csv")

# Create directories
os.makedirs(IMAGES_DIR, exist_ok=True)

def generate_tabular_data(n=NUM_SAMPLES):
    """
    Generates synthetic tabular data for welding parameters.
    """
    data = []
    for i in range(n):
        # Random inputs within realistic ranges
        current = np.random.uniform(100, 300)  # Amps
        voltage = np.random.uniform(20, 40)    # Volts
        speed = np.random.uniform(10, 50)      # cm/min
        feed_rate = np.random.uniform(5, 15)   # m/min
        gas_flow = np.random.uniform(10, 25)   # L/min
        temp = np.random.uniform(20, 100)      # Pre-heat temp C
        thickness = np.random.uniform(2, 10)   # mm
        material = random.choice(["Steel", "Aluminum", "Titanium"])

        # Determine Defect & Quality based on heuristic rules
        # Rule 1: High current + Low speed = Burn Through
        # Rule 2: Low current + High speed = Incomplete Fusion
        # Rule 3: Low gas flow = Porosity
        # Rule 4: Optimal range = Good

        defect = "Normal"
        quality_score = np.random.uniform(85, 99) # Default good score
        
        # Inject defects
        if current > 260 and speed < 20:
            defect = "Burn Through"
            quality_score = np.random.uniform(40, 60)
        elif current < 140 and speed > 40:
            defect = "Incomplete Fusion"
            quality_score = np.random.uniform(50, 70)
        elif gas_flow < 12:
            defect = "Porosity"
            quality_score = np.random.uniform(60, 75)
        elif voltage > 35:
            defect = "Spatter"
            quality_score = np.random.uniform(70, 85)
        
        # Pass/Fail logic
        passed = 1 if quality_score > 75 else 0
        
        # Add some noise to score
        quality_score += np.random.normal(0, 2)
        quality_score = np.clip(quality_score, 0, 100)

        data.append({
            "Current": current,
            "Voltage": voltage,
            "TravelSpeed": speed,
            "WireFeedRate": feed_rate,
            "GasFlow": gas_flow,
            "Temperature": temp,
            "PlateThickness": thickness,
            "MaterialType": material,
            "TensileStrength": quality_score * 10, # Mock tensile strength
            "DefectType": defect,
            "Passed": passed,
            "ImageID": f"weld_{i:04d}.png"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(TABULAR_FILE, index=False)
    print(f"Tabular data saved to {TABULAR_FILE}")
    return df

def generate_weld_image(filename, defect_type):
    """
    Generates a synthetic image of a weld bead.
    """
    img = Image.new('L', IMAGE_SIZE, color=0) # Black background
    draw = ImageDraw.Draw(img)
    
    # Draw the weld bead (a noisy white line)
    start_point = (10, IMAGE_SIZE[1]//2)
    end_point = (IMAGE_SIZE[0]-10, IMAGE_SIZE[1]//2)
    width = random.randint(5, 8)
    
    # Draw main bead
    draw.line([start_point, end_point], fill=200, width=width)
    
    # Add noise/texture to bead
    img_np = np.array(img)
    noise = np.random.randint(-20, 20, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    
    # Add specific visual defects
    if defect_type == "Porosity":
        # Draw small black circles on the bead
        for _ in range(random.randint(3, 8)):
            x = random.randint(15, IMAGE_SIZE[0]-15)
            y = random.randint(IMAGE_SIZE[1]//2 - 2, IMAGE_SIZE[1]//2 + 2)
            cv2.circle(img_np, (x, y), 1, 50, -1)
            
    elif defect_type == "Burn Through":
        # Large dark gap in middle
        x = random.randint(20, IMAGE_SIZE[0]-20)
        cv2.circle(img_np, (x, IMAGE_SIZE[1]//2), 4, 20, -1)
        
    elif defect_type == "Crack":
        # Thin dark line across bead
        x = random.randint(20, IMAGE_SIZE[0]-20)
        y = IMAGE_SIZE[1]//2
        cv2.line(img_np, (x, y-4), (x+2, y+4), 50, 1)

    # Save
    cv2.imwrite(os.path.join(IMAGES_DIR, filename), img_np)

if __name__ == "__main__":
    print("Generating data...")
    df = generate_tabular_data()
    print("Generating images...")
    for idx, row in df.iterrows():
        generate_weld_image(row['ImageID'], row['DefectType'])
    print("Done!")
