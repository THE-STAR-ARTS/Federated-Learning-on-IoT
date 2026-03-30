import csv
import random
import time
from datetime import datetime

CSV_PATH = "data/dataset.csv"

def generate_reading():
    temperature = random.uniform(15, 40)      # �C
    humidity = random.uniform(20, 90)          # %
    light = random.uniform(0, 1000)            # lux

    # Simple environmental danger label
    danger = 1 if temperature > 35 or humidity < 25 else 0
    return temperature, humidity, light, danger

def run_generator(interval=5):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        while True:
            row = [datetime.now().isoformat(), *generate_reading()]
            writer.writerow(row)
            f.flush()
            time.sleep(interval)

if __name__ == "__main__":
    run_generator()

import csv
import random
import time
from datetime import datetime

CSV_PATH = "data/dataset.csv"

def generate_reading():
    temperature = random.uniform(15, 40)      # �C
    humidity = random.uniform(20, 90)          # %
    light = random.uniform(0, 1000)            # lux

    # Simple environmental danger label
    danger = 1 if temperature > 35 or humidity < 25 else 0
    return temperature, humidity, light, danger

def run_generator(interval=5):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        while True:
            row = [datetime.now().isoformat(), *generate_reading()]
            writer.writerow(row)
            f.flush()
            time.sleep(interval)

if __name__ == "__main__":
    run_generator()

