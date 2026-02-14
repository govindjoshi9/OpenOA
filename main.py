from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from pathlib import Path
from openoa.plant import PlantData
from openoa.analysis.aep import MonteCarloAEP
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenOA Web API")

@app.get("/")
async def root():
    return FileResponse('index.html')

@app.post("/calculate")
async def calculate():
    try:
        data_path = Path("examples/data/la_haute_borne")
        
        logger.info("Loading project data...")
        
        asset_df = pd.read_csv(data_path / "la-haute-borne_asset_table.csv")


        scada_df = pd.read_csv(
            data_path / "la-haute-borne-data-2014-2015.csv", 
            usecols=["Date_time", "P_avg", "Ws_avg", "Wind_turbine_name"]
        )
        scada_df["Date_time"] = pd.to_datetime(scada_df["Date_time"], utc=True).dt.tz_localize(None)

        meter_cols = ["time_utc", "net_energy_kwh", "availability_kwh", "curtailment_kwh"]
        meter_df = pd.read_csv(data_path / "plant_data.csv", usecols=meter_cols)
        meter_df["time_utc"] = pd.to_datetime(meter_df["time_utc"], utc=True).dt.tz_localize(None)

        curtail_df = meter_df.copy()


        
        era5_cols = ["datetime", "ws_100m", "u_100", "v_100", "dens_100m"]
        era5_df = pd.read_csv(data_path / "era5_wind_la_haute_borne.csv", index_col=0, usecols=["Unnamed: 0"] + era5_cols)
        era5_df["datetime"] = pd.to_datetime(era5_df["datetime"], utc=True).dt.tz_localize(None)
        
        merra2_cols = ["datetime", "ws_50m", "u_50", "v_50", "dens_50m"]
        merra2_df = pd.read_csv(data_path / "merra2_la_haute_borne.csv", index_col=0, usecols=["Unnamed: 0"] + merra2_cols)
        merra2_df["datetime"] = pd.to_datetime(merra2_df["datetime"], utc=True).dt.tz_localize(None)

        metadata = {
            "capacity": 8.2,
            "asset": {
                "asset_id": "Wind_turbine_name",
                "latitude": "Latitude",
                "longitude": "Longitude"
            },
            "meter": {
                "time": "time_utc",
                "MMTR_SupWh": "net_energy_kwh"
            },
            "curtail": {
                "time": "time_utc",
                "IAVL_DnWh": "availability_kwh",
                "IAVL_ExtPwrDnWh": "curtailment_kwh"
            },
            "scada": {
                "time": "Date_time",
                "asset_id": "Wind_turbine_name",
                "WTUR_W": "P_avg",
                "WMET_HorWdSpd": "Ws_avg"
            },
            "reanalysis": {
                "era5": {
                    "time": "datetime",
                    "WMETR_HorWdSpd": "ws_100m",
                    "WMETR_HorWdSpdU": "u_100",
                    "WMETR_HorWdSpdV": "v_100",
                    "WMETR_AirDen": "dens_100m"
                },
                "merra2": {
                    "time": "datetime",
                    "WMETR_HorWdSpd": "ws_50m",
                    "WMETR_HorWdSpdU": "u_50",
                    "WMETR_HorWdSpdV": "v_50",
                    "WMETR_AirDen": "dens_50m"
                }
            }
        }

        logger.info("Initializing PlantData...")
        plant = PlantData(
            metadata=metadata,
            scada=scada_df,
            meter=meter_df,
            curtail=curtail_df,
            asset=asset_df,
            reanalysis={"era5": era5_df, "merra2": merra2_df}
        )

        logger.info("Running AEP Monte Carlo analysis...")
        pa = MonteCarloAEP(plant)
        pa.run(num_sim=5, progress_bar=False)
        aep_mean = float(pa.results['aep_GWh'].mean())
        aep_std = float(pa.results['aep_GWh'].std()) if 'aep_GWh' in pa.results else 0.0

        return {
            "status": "success",
            "aep_GWh": aep_mean,
            "aep_std": aep_std,
            "metrics": pa.results.mean().to_dict()
        }

    except Exception as e:

        logger.exception("Error during OpenOA calculation")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
