import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)

    client.admin.command("ping")
    print("MongoDB Atlas connection successful!")

    db = client["pothole_db"]
    collection = db["detections"]

    result = collection.insert_one({
        "test": True,
        "message": "X-PotholeNet MongoDB Atlas test"
    })

    print("Test document inserted successfully!")
    print("Document ID:", result.inserted_id)

except Exception as e:
    print("MongoDB connection failed:")
    print(e)

finally:
    try:
        client.close()
    except:
        pass