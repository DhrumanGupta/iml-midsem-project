from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any

from .main import models_dict, load_model, run_test

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelConfig(BaseModel):
    model_name: str
    config: Dict[str, Any]


class SimulationResponse(BaseModel):
    result: List[List[float]]


@app.get("/models", response_model=List[str])
async def get_models():
    """Return a list of available models"""
    return list(models_dict.keys())


@app.post("/run", response_model=SimulationResponse)
async def run_simulation(model_config: ModelConfig):
    """Run a simulation with the specified model and configuration"""
    # Check if model exists
    if model_config.model_name not in models_dict:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_config.model_name}' not found"
        )

    try:
        # Get the model
        model = models_dict[model_config.model_name]

        # Load the model instance
        model_instance = load_model(
            model_config.model_name,
            f"models/{model_config.model_name}/checkpoints/model_1.pth",
        )

        # Run the test
        result = run_test(model, model_instance, model_config.config)

        # Convert numpy array to list for JSON serialization
        return SimulationResponse(result=result.tolist())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error running simulation: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
