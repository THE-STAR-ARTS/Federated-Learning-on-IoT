Server_IP = input("Enter the IP adress of your client:")

SERVER_ADDRESS = f"{Server_IP}:8080"   # change later
CSV_PATH = "data/dataset.csv"
NUM_FEATURES = 3                   # temp, humidity, light
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.01
