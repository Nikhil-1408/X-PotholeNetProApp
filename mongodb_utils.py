import os
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()


def get_config(name, default=None):
    """
    Get configuration from Streamlit secrets first,
    then fall back to environment variables.
    """

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


MONGO_URI = get_config("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = get_config("MONGO_DB_NAME", "pothole_db")
MONGO_COLLECTION_NAME = get_config(
    "MONGO_COLLECTION_NAME",
    "detections"
)


_client = None


def get_collection():
    global _client

    try:
        if _client is None:
            _client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000
            )

        # Test connection
        _client.admin.command("ping")

        db = _client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]

        # Create useful indexes
        collection.create_index([("timestamp", -1)])
        collection.create_index([("source_type", 1)])

        return collection

    except PyMongoError as e:
        print("MongoDB connection error:", e)
        return None


def save_detection(
    source_type,
    filename,
    mode,
    counts,
    risk_score,
    road_status,
    alert,
    detections,
    general_objects_count=0
):
    """
    Save one completed pothole detection to MongoDB.
    """

    collection = get_collection()

    if collection is None:
        return None

    document = {
        "timestamp": datetime.now(timezone.utc),

        "source_type": source_type,

        "filename": filename,

        "mode": mode,

        "pothole_count": int(sum(counts.values())),

        "severity": {
            "low": int(counts.get("Low", 0)),
            "medium": int(counts.get("Medium", 0)),
            "high": int(counts.get("High", 0))
        },

        "risk_score": float(risk_score),

        "road_status": road_status,

        "alert": alert,

        "general_objects_count": int(
            general_objects_count
        ),

        "detections": detections
    }

    try:
        result = collection.insert_one(document)

        return str(result.inserted_id)

    except PyMongoError as e:
        print("MongoDB save error:", e)
        return None


def get_history(limit=100):

    collection = get_collection()

    if collection is None:
        return []

    try:
        return list(
            collection
            .find()
            .sort("timestamp", -1)
            .limit(limit)
        )

    except PyMongoError as e:
        print("MongoDB history error:", e)
        return []


def delete_all_history():

    collection = get_collection()

    if collection is None:
        return False

    try:
        collection.delete_many({})
        return True

    except PyMongoError as e:
        print("MongoDB delete error:", e)
        return False


def mongodb_available():

    return get_collection() is not None