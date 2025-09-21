from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import uvicorn
import os
import json
import logging
from contextlib import asynccontextmanager
from multi1 import EnhancedMultilingualHazardDetector,HazardReport
import torch
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("Initializing Hazard Detection System...")
    

    detector = EnhancedMultilingualHazardDetector(use_gpu=torch.cuda.is_available())
    logger.info(f"System initialized. GPU: {torch.cuda.is_available()}")
    
    yield
    
    logger.info("Shutting down Hazard Detection System...")
    if detector:
        detector.export_reports("json", f"shutdown_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


app = FastAPI(
    title="Multilingual Hazard Detection API",
    description="Advanced AI-powered hazard detection system supporting multiple languages and real-time monitoring",
    version="2.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class LanguageEnum(str, Enum):
    AUTO = "auto"
    ENGLISH = "en"
    HINDI = "hi"
    BENGALI = "bn"
    TAMIL = "ta"
    TELUGU = "te"
    GUJARATI = "gu"
    MALAYALAM = "ml"
    KANNADA = "kn"
    MARATHI = "mr"
    PUNJABI = "pa"
    URDU = "ur"
    ODIA = "or"

class HazardTypeEnum(str, Enum):
    TSUNAMI = "tsunami"
    FLOOD = "flood"
    FLASH_FLOOD = "flash_flood"
    HIGH_WAVES = "high_waves"
    STORM_SURGE = "storm_surge"
    WATER_SPOUT = "water_spout"
    DAM_BURST = "dam_burst"
    RIVER_OVERFLOW = "river_overflow"
    COASTAL_EROSION = "coastal_erosion"
    KING_TIDE = "king_tide"
    SEICHE = "seiche"
    CYCLONE = "cyclone"
    EARTHQUAKE = "earthquake"
    WILDFIRE = "wildfire"
    LANDSLIDE = "landslide"
    UNKNOWN = "unknown"

class SeverityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExportFormatEnum(str, Enum):
    JSON = "json"
    CSV = "csv"

class HazardReportRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze for hazards")
    source: str = Field(default="user_report", description="Source of the report")
    language: Optional[LanguageEnum] = Field(default=LanguageEnum.AUTO, description="Language of the text")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v

class BatchHazardReportRequest(BaseModel):
    reports: List[HazardReportRequest] = Field(..., min_items=1, max_items=100, description="List of reports to process")

class LocationInfo(BaseModel):
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: float
    method: str

class HazardReportResponse(BaseModel):
    id: str
    text: str
    timestamp: datetime
    language: str
    hazard_type: HazardTypeEnum
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: SeverityEnum
    locations: List[LocationInfo]
    sentiment: str
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    source: str
    verified: bool = False
    affected_population: Optional[int] = None
    coordinates: Optional[List[float]] = None
    detection_details: Optional[Dict[str, Any]] = None

class TrendAnalysisResponse(BaseModel):
    total_reports: int
    hazard_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    language_distribution: Dict[str, int]
    average_confidence: float
    average_urgency: float
    time_period_days: int

class AlertSummaryResponse(BaseModel):
    total_alerts: int
    severity_threshold: str
    time_range: Dict[str, str]
    hazard_breakdown: Dict[str, Dict[str, Union[int, float]]]
    location_breakdown: Dict[str, Dict[str, Union[int, List[str]]]]
    most_urgent: Optional[Dict[str, Any]]

class ClusteringResponse(BaseModel):
    total_clusters: int
    noise_reports: int
    clusters: Dict[str, List[Dict[str, Any]]]

class MonitoringConfig(BaseModel):
    data_sources: List[str] = Field(..., min_items=1, description="List of URLs to monitor")
    interval_seconds: int = Field(default=300, ge=60, le=3600, description="Monitoring interval in seconds")
    severity_filter: Optional[SeverityEnum] = Field(default=None, description="Filter for minimum severity")

class TwitterCredentials(BaseModel):
    api_key: str
    api_secret: str
    access_token: str
    access_token_secret: str

class RedditCredentials(BaseModel):
    client_id: str
    client_secret: str
    user_agent: str


monitoring_task = None
monitoring_active = False


@app.get("/", tags=["General"])
async def root():
  
    return {
        "message": "Multilingual Hazard Detection API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Check API and model health status"""
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Hazard detection system not initialized")
    
    try:
        test_text = "Test flood detection"
        report = detector.process_multilingual_report_enhanced(test_text, "health_check")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "gpu_available": torch.cuda.is_available(),
                "gpu_in_use": detector.device.type == "cuda" if detector else False,
                "total_reports_processed": len(detector.historical_reports),
                "models_loaded": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/analyze", response_model=HazardReportResponse, tags=["Hazard Detection"])
async def analyze_hazard_report(request: HazardReportRequest):
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        report = detector.process_multilingual_report_enhanced(
            request.text,
            request.source
        )
        
       
        locations = []
        for loc in report.locations:
            location_data = {
                "name": loc.get("name", "unknown"),
                "latitude": loc.get("latitude"),
                "longitude": loc.get("longitude"),
                "confidence": loc.get("confidence", 0.0),
                "method": loc.get("method", "basic_extraction")
            }
            locations.append(LocationInfo(**location_data))
        
        return HazardReportResponse(
            id=f"report_{datetime.now().timestamp()}",
            text=report.text,
            timestamp=report.timestamp,
            language=report.language,
            hazard_type=report.hazard_type,
            confidence=report.confidence,
            severity=report.severity,
            locations=locations,
            sentiment=report.sentiment,
            urgency_score=report.urgency_score,
            source=report.source,
            verified=report.verified,
            affected_population=report.affected_population,
            coordinates=list(report.coordinates) if report.coordinates else None
        )
        
    except Exception as e:
        logger.error(f"Error analyzing report: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch", response_model=List[HazardReportResponse], tags=["Hazard Detection"])
async def analyze_batch_hazard_reports(request: BatchHazardReportRequest):
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        texts = [r.text for r in request.reports]
        sources = [r.source for r in request.reports]
        
        reports = detector.batch_process_reports(texts, sources)
        
        responses = []
        for i, report in enumerate(reports):
           
            locations = []
            for loc in report.locations:
                location_data = {
                    "name": loc.get("name", "unknown"),
                    "latitude": loc.get("latitude"),
                    "longitude": loc.get("longitude"),
                    "confidence": loc.get("confidence", 0.0),
                    "method": loc.get("method", "basic_extraction")
                }
                locations.append(LocationInfo(**location_data))
            
            responses.append(HazardReportResponse(
                id=f"batch_{datetime.now().timestamp()}_{i}",
                text=report.text,
                timestamp=report.timestamp,
                language=report.language,
                hazard_type=report.hazard_type,
                confidence=report.confidence,
                severity=report.severity,
                locations=locations,
                sentiment=report.sentiment,
                urgency_score=report.urgency_score,
                source=report.source,
                verified=report.verified,
                affected_population=report.affected_population,
                coordinates=list(report.coordinates) if report.coordinates else None
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/trends", response_model=TrendAnalysisResponse, tags=["Analytics"])
async def get_trend_analysis(
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze")
):
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        trends = detector.get_trend_analysis(days=days)
        
        if "message" in trends:
            raise HTTPException(status_code=404, detail=trends["message"])
        
        return TrendAnalysisResponse(
            total_reports=trends["total_reports"],
            hazard_distribution=trends["hazard_distribution"],
            severity_distribution=trends["severity_distribution"],
            language_distribution=trends["language_distribution"],
            average_confidence=trends["average_confidence"],
            average_urgency=trends["average_urgency"],
            time_period_days=days
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating trends: {e}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@app.get("/alerts/summary", response_model=AlertSummaryResponse, tags=["Alerts"])
async def get_alert_summary(
    severity_threshold: SeverityEnum = Query(default=SeverityEnum.MEDIUM, description="Minimum severity level")
):
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        summary = detector.generate_alert_summary(severity_threshold.value)
        
        if "message" in summary:
            raise HTTPException(status_code=404, detail=summary["message"])
        
        return AlertSummaryResponse(**summary)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating alert summary: {e}")
        raise HTTPException(status_code=500, detail=f"Alert summary failed: {str(e)}")

@app.post("/clusters", response_model=ClusteringResponse, tags=["Analytics"])
async def cluster_similar_reports(
    threshold: float = Query(default=0.8, ge=0.5, le=1.0, description="Similarity threshold for clustering")
):
   
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        clusters = detector.cluster_similar_reports(threshold=threshold)
        
        if "message" in clusters:
            raise HTTPException(status_code=404, detail=clusters["message"])
        
        serialized_clusters = {}
        for cluster_id, cluster_data in clusters["clusters"].items():
            serialized_clusters[str(cluster_id)] = [
                {
                    "report_index": item["report_index"],
                    "similarity_score": item["similarity_score"],
                    "hazard_type": item["report"].hazard_type,
                    "severity": item["report"].severity,
                    "text_preview": item["report"].text[:100] + "..." if len(item["report"].text) > 100 else item["report"].text
                }
                for item in cluster_data
            ]
        
        return ClusteringResponse(
            total_clusters=clusters["total_clusters"],
            noise_reports=clusters["noise_reports"],
            clusters=serialized_clusters
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clustering reports: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@app.get("/export", tags=["Data Management"])
async def export_reports(
    format: ExportFormatEnum = Query(default=ExportFormatEnum.JSON, description="Export format"),
    filename: Optional[str] = Query(default=None, description="Custom filename (without extension)")
):
    
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        if filename is None:
            filename = f"hazard_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = detector.export_reports(format=format.value, filename=filename)
        
        # Check if file was created
        file_path = f"{filename}.{format.value}"
        if os.path.exists(file_path):
            return FileResponse(
                file_path,
                media_type="application/octet-stream",
                filename=os.path.basename(file_path)
            )
        else:
            raise HTTPException(status_code=404, detail=result)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting reports: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.post("/monitoring/start", tags=["Real-time Monitoring"])
async def start_monitoring(
    config: MonitoringConfig,
    background_tasks: BackgroundTasks
):
   
    global detector, monitoring_task, monitoring_active
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if monitoring_active:
        raise HTTPException(status_code=409, detail="Monitoring already active")
    
    try:
        monitoring_active = True
        
        
        async def monitor():
            try:
                await detector.real_time_monitoring(
                    config.data_sources,
                    config.interval_seconds
                )
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                monitoring_active = False
        
        monitoring_task = asyncio.create_task(monitor())
        
        return {
            "status": "started",
            "data_sources": config.data_sources,
            "interval": config.interval_seconds,
            "message": "Real-time monitoring started successfully"
        }
        
    except Exception as e:
        monitoring_active = False
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@app.post("/monitoring/stop", tags=["Real-time Monitoring"])
async def stop_monitoring():
    
    global monitoring_task, monitoring_active
    
    if not monitoring_active:
        raise HTTPException(status_code=404, detail="No active monitoring session")
    
    try:
        if monitoring_task:
            monitoring_task.cancel()
            monitoring_task = None
        
        monitoring_active = False
        
        return {
            "status": "stopped",
            "message": "Real-time monitoring stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@app.get("/monitoring/status", tags=["Real-time Monitoring"])
async def monitoring_status():
   
    global monitoring_active
    
    return {
        "active": monitoring_active,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/credentials/twitter", tags=["Configuration"])
async def set_twitter_credentials(credentials: TwitterCredentials):
   
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = detector.set_twitter_credentials(
            credentials.api_key,
            credentials.api_secret,
            credentials.access_token,
            credentials.access_token_secret
        )
        
        if success:
            return {"status": "success", "message": "Twitter credentials configured successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid Twitter credentials")
            
    except Exception as e:
        logger.error(f"Error setting Twitter credentials: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure Twitter: {str(e)}")

@app.post("/credentials/reddit", tags=["Configuration"])
async def set_reddit_credentials(credentials: RedditCredentials):
    
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = detector.set_reddit_credentials(
            credentials.client_id,
            credentials.client_secret,
            credentials.user_agent
        )
        
        if success:
            return {"status": "success", "message": "Reddit credentials configured successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid Reddit credentials")
            
    except Exception as e:
        logger.error(f"Error setting Reddit credentials: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure Reddit: {str(e)}")

@app.get("/statistics", tags=["Analytics"])
async def get_statistics():
   
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        total_reports = len(detector.historical_reports)
        
        if total_reports == 0:
            return {
                "total_reports": 0,
                "message": "No reports processed yet"
            }
        
       
        hazard_types = set(r.hazard_type for r in detector.historical_reports)
        languages = set(r.language for r in detector.historical_reports)
        
        severity_counts = {
            "critical": len([r for r in detector.historical_reports if r.severity == "critical"]),
            "high": len([r for r in detector.historical_reports if r.severity == "high"]),
            "medium": len([r for r in detector.historical_reports if r.severity == "medium"]),
            "low": len([r for r in detector.historical_reports if r.severity == "low"])
        }
        
    
        all_locations = []
        for report in detector.historical_reports:
            for loc in report.locations:
                if loc['name'] != 'unknown':
                    all_locations.append(loc['name'])
        
        unique_locations = list(set(all_locations))
        
        return {
            "total_reports": total_reports,
            "unique_hazard_types": list(hazard_types),
            "unique_languages": list(languages),
            "severity_breakdown": severity_counts,
            "unique_locations_count": len(unique_locations),
            "top_locations": unique_locations[:10] if unique_locations else [],
            "average_confidence": float(sum(r.confidence for r in detector.historical_reports) / total_reports),
            "average_urgency": float(sum(r.urgency_score for r in detector.historical_reports) / total_reports),
            "verified_reports": len([r for r in detector.historical_reports if r.verified]),
            "latest_report_time": max(r.timestamp for r in detector.historical_reports).isoformat() if detector.historical_reports else None
        }
        
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics generation failed: {str(e)}")

@app.delete("/reports/clear", tags=["Data Management"])
async def clear_reports():
   
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        count = len(detector.historical_reports)
        detector.historical_reports = []
        
        return {
            "status": "success",
            "message": f"Cleared {count} reports from memory"
        }
        
    except Exception as e:
        logger.error(f"Error clearing reports: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear reports: {str(e)}")


from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            await websocket.send_text(f"Message received: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
     uvicorn.run(
        "endpint:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )