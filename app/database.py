"""MongoDB async database connection."""

import logging
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = "mongodb+srv://shivamchoughule2_db_user:<db_password>@cluster0.lgtcwnk.mongodb.net/?appName=Cluster0"
DB_NAME = "promptops_db"
COLLECTION_HISTORY = "history"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the async client
client = None
db = None
history_collection = None

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
history_collection = db[COLLECTION_HISTORY]

async def ping_db():
    try:
        await client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
    except Exception as e:
        logger.error(f"MongoDB connection error: {e}")
