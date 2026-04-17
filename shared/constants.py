AGE_MAP = {
    "teens": 0, "twenties": 1, "thirties": 2, "fourties": 3,
    "fifties": 4, "sixties": 5, "seventies": 6,
}
AGE_LABELS = ["Teens", "Twenties", "Thirties", "Forties", "Fifties", "Sixties", "Seventies"]
NUM_AGE_CLASSES = 7

GENDER_MAP = {"male_masculine": 0, "female_feminine": 1}
GENDER_LABELS = ["Male", "Female"]

TARGET_SR = 16000
MAX_DURATION_SEC = 3.0
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION_SEC)  # 48000
