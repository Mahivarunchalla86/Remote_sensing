import cv2
import pandas as pd
import re
import easyocr
from scipy.spatial import distance
import os
def preprocess(img_path, scale=3.0):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {img_path}. Please check the path.")
    h, w = image.shape[:2]
    image_big = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image_big, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=25)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.5, gray, -0.5, 0)
    return image_big, sharp


reader = easyocr.Reader(['en'], gpu=False, verbose=False)


def extract_text_with_positions(img):
    results = reader.readtext(img)
    entries = []
    for (bbox, text, conf) in results:
        if conf < 0.3:
            continue
        cleaned = re.sub(r'[^0-9A-Za-z ]', '', text).strip()
        if len(cleaned) < 2:
            continue
        x_center = int((bbox[0][0] + bbox[2][0]) / 2)
        y_center = int((bbox[0][1] + bbox[2][1]) / 2)
        width = int(bbox[2][0] - bbox[0][0])
        height = int(bbox[2][1] - bbox[0][1])
        entries.append({
            'text': cleaned,
            'conf': conf,
            'x': x_center,
            'y': y_center,
            'width': width,
            'height': height
        })
    return entries


def classify_and_pair(entries):
    places = []
    numbers = []

    for entry in entries:
        txt = entry['text']
        if txt.isdigit():
            numbers.append(entry)
        elif re.search(r'[A-Za-z]', txt):
            places.append(entry)

    assigned = []
    used_numbers = set()  # Track numbers that have already been assigned

    for place in places:
        available_numbers = [
            n for n in numbers if (n['x'], n['y']) not in used_numbers
        ]
        if available_numbers:
            nearest_number = min(
                available_numbers,
                key=lambda n: distance.euclidean((n['x'], n['y']), (place['x'], place['y']))
            )
            assigned.append({
                'Characters': place['text'].title(),
                'Numbers': nearest_number['text']
            })
            # Mark this number as used
            used_numbers.add((nearest_number['x'], nearest_number['y']))
        else:
            # Assign a default number based on character size
            default_number = int((place['width'] + place['height']) / 2)
            assigned.append({
                'Characters': place['text'].title(),
                'Numbers': str(default_number)
            })

    return pd.DataFrame(assigned)


def export_output(df, output_xlsx="output.xlsx", output_csv="output.csv"):
    df.to_excel(output_xlsx, index=False)
    print(f"Results exported to {output_xlsx}")
    df.to_csv(output_csv, index=False)
    print(f"Results exported to {output_csv}")


def main():
    img_path = input("Enter the full path to the image: ").strip()

    if not os.path.exists(img_path):
        print("File not found. Please check the path and try again.")
        return

    try:
        image_big, processed_img = preprocess(img_path)
        entries = extract_text_with_positions(image_big)
        df_result = classify_and_pair(entries)

        print("Processing completed. Exporting results...")
        export_output(df_result)

        print("Pipeline completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
