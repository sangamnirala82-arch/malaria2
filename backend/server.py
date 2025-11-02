from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import uuid
from datetime import datetime, timezone
import tensorflow as tf
import numpy as np
from PIL import Image
import io


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Load the malaria detection model at startup
malaria_model = None
MODEL_PATH = ROOT_DIR / "models" / "best_model.h5"

def load_malaria_model():
    """Load the trained malaria detection model"""
    global malaria_model
    try:
        if MODEL_PATH.exists():
            malaria_model = tf.keras.models.load_model(str(MODEL_PATH))
            logger.info(f"✅ Malaria detection model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Model file not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_malaria_model()


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Ignore MongoDB's _id field
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    # Exclude MongoDB's _id field from the query results
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

# Malaria Detection Endpoint
@api_router.post("/predict-malaria")
async def predict_malaria(file: UploadFile = File(...)):
    """
    Predict whether a cell image is parasitized or uninfected
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with prediction result and confidence
    """
    try:
        # Check if model is loaded
        if malaria_model is None:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please ensure the model file exists."
            )
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224)
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = malaria_model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine class
        if confidence >= 0.5:
            result = "Parasitized"
            confidence_percentage = confidence * 100
        else:
            result = "Uninfected"
            confidence_percentage = (1 - confidence) * 100
        
        # Save prediction to database
        prediction_record = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "prediction": result,
            "confidence": confidence_percentage,
            "raw_score": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await db.predictions.insert_one(prediction_record)
        
        return JSONResponse(content={
            "success": True,
            "prediction": result,
            "confidence": round(confidence_percentage, 2),
            "message": f"The cell image is predicted to be {result} with {confidence_percentage:.2f}% confidence."
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@api_router.get("/prediction-history")
async def get_prediction_history(limit: int = 50):
    """Get recent prediction history"""
    try:
        predictions = await db.predictions.find(
            {}, 
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit).to_list(limit)
        
        return {
            "success": True,
            "predictions": predictions,
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Error fetching prediction history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()