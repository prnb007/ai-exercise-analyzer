import os
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBConfig:
    _client = None
    _async_client = None
    _db = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            # Try MongoDB Atlas first, then fallback to local
            mongo_uri = os.environ.get("MONGO_URI", "mongodb+srv://gargpranab01_db_user:ef88fqwwJMuplaPN@cluster0.q2vrs1i.mongodb.net/")
            db_name = os.environ.get("MONGO_DB_NAME", "exercise_analyzer")
            try:
                # Try Atlas connection first
                cls._client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                cls._db = cls._client[db_name]
                # Test connection
                cls._client.admin.command('ping')
                logger.info(f"Successfully connected to MongoDB Atlas: {db_name}")
            except Exception as e:
                logger.warning(f"Atlas connection failed: {e}")
                try:
                    # Fallback to local MongoDB
                    local_uri = "mongodb://localhost:27017/"
                    cls._client = MongoClient(local_uri, serverSelectionTimeoutMS=2000)
                    cls._db = cls._client[db_name]
                    cls._client.admin.command('ping')
                    logger.info(f"Successfully connected to local MongoDB: {db_name}")
                except Exception as local_e:
                    logger.error(f"Both Atlas and local MongoDB failed: {local_e}")
                    raise Exception("Failed to connect to MongoDB. Please check your connection string and network.")
        return cls._client

    @classmethod
    def get_async_client(cls):
        if cls._async_client is None:
            # Try MongoDB Atlas first, then fallback to local
            mongo_uri = os.environ.get("MONGO_URI", "mongodb+srv://gargpranab01_db_user:ef88fqwwJMuplaPN@cluster0.q2vrs1i.mongodb.net/")
            db_name = os.environ.get("MONGO_DB_NAME", "exercise_analyzer")
            try:
                # Try Atlas connection first
                cls._async_client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=5000)
                cls._db = cls._async_client[db_name]
                logger.info(f"Successfully connected to Async MongoDB Atlas: {db_name}")
            except Exception as e:
                logger.warning(f"Async Atlas connection failed: {e}")
                try:
                    # Fallback to local MongoDB
                    local_uri = "mongodb://localhost:27017/"
                    cls._async_client = AsyncIOMotorClient(local_uri, serverSelectionTimeoutMS=2000)
                    cls._db = cls._async_client[db_name]
                    logger.info(f"Successfully connected to Async local MongoDB: {db_name}")
                except Exception as local_e:
                    logger.error(f"Both async Atlas and local MongoDB failed: {local_e}")
                    raise Exception("Failed to connect to MongoDB. Please check your connection string and network.")
        return cls._async_client

    @classmethod
    def get_db(cls):
        if cls._db is None:
            cls.get_client() # Ensure client is initialized
        return cls._db

    @classmethod
    def close_connection(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("MongoDB connection closed.")
        if cls._async_client:
            cls._async_client.close()
            cls._async_client = None
            logger.info("Async MongoDB connection closed.")

mongodb_config = MongoDBConfig()

def get_mongodb():
    """Get MongoDB database instance"""
    return mongodb_config.get_db()

def get_collection(collection_name):
    """Get a specific collection"""
    return mongodb_config.get_db()[collection_name]